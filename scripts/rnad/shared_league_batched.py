"""Experimental batched three-phase RNAD learner, isolated from the default.

Only strictly verified, already accepted full Matches may enter this module.
No optimizer is constructed and no checkpoint is modified here. Each Match's
V-trace/NeuRD computation remains independent and in authoritative event order.
The Battle recurrent state is indexed by (match, seat), with detach after EACH
accepted battle decision exactly as the scalar reference learner does.
"""
from __future__ import annotations

import os
from collections import defaultdict

import torch

from scripts.rnad.shared_league_learner import ETA, VALUE_WEIGHT
from scripts.rnad.two_player import neurd_policy_loss, rnad_targets

PHASES = ("draft", "selection", "battle")
# Protect hosted runner memory when one accepted draft step has huge candidate lists.
MAX_DRAFT_CANDIDATES = 20000
MAX_SELECTION_STATES = 32
_COMPILED_MODULES = {}


def _call(module, *args, **kwargs):
    """Optional TorchInductor on *modules only*: never mutate their state_dict.

    The compiled callable is cached across all 40 updates, and this is an
    explicit experimental opt-in because CPU compilation may take minutes.
    """
    flag = os.environ.get("RNAD_TORCH_COMPILE", "0")
    if flag not in ("0", "1"):
        raise ValueError("RNAD_TORCH_COMPILE must be 0 or 1")
    if flag == "0":
        return module(*args, **kwargs)
    # The complex 60-action selection embedding/scatter graph failed the
    # real Match gradient-parity gate under TorchInductor. Its weights MUST
    # remain eager while smaller independent heads are explored.
    if type(module).__name__ == "DecomposedTeacher":
        return module(*args, **kwargs)
    key = id(module)
    compiled = _COMPILED_MODULES.get(key)
    if compiled is None:
        compiled = torch.compile(module, backend="inductor", dynamic=True)
        _COMPILED_MODULES[key] = compiled
    return compiled(*args, **kwargs)


def _chunk_by_size(rows, size_fn, capacity):
    chunk, count = [], 0
    for row in rows:
        n = size_fn(row)
        if chunk and count + n > capacity:
            yield chunk
            chunk, count = [], 0
        chunk.append(row)
        count += n
    if chunk:
        yield chunk


def _draft_inference(net, rows):
    """One phi for all pools, one rho for all ragged legal candidate teams."""
    pool_parts, code_parts, extra_parts = [], [], []
    pool_sizes, action_sizes = [], []
    offset = 0
    for _, _, payload in rows:
        pool = torch.as_tensor(payload["pool"], dtype=torch.float32)
        codes = torch.as_tensor(payload["codes"], dtype=torch.long)
        extra = torch.as_tensor(payload["extra"], dtype=torch.float32)
        assert pool.ndim == 2 and pool.shape[1] == 44
        assert codes.ndim == 2 and codes.shape[1] == 6
        assert extra.shape == (codes.shape[0], 4)
        assert codes.numel() > 0 and int(codes.min()) >= 0
        assert int(codes.max()) < pool.shape[0]
        pool_parts.append(pool)
        code_parts.append(codes + offset)
        extra_parts.append(extra)
        pool_sizes.append(pool.shape[0])
        action_sizes.append(codes.shape[0])
        offset += pool.shape[0]
    pool_tensor = torch.cat(pool_parts, dim=0)
    codes_tensor = torch.cat(code_parts, dim=0)
    extra_tensor = torch.cat(extra_parts, dim=0)
    embedded = _call(net.draft.phi, pool_tensor)
    selected = embedded[codes_tensor]
    features = torch.cat((selected.mean(dim=1),
                          selected.square().mean(dim=1),
                          selected.amax(dim=1), extra_tensor), dim=1)
    all_logits = _call(net.draft.rho, features).reshape(-1)
    out = {}
    pool_start, action_start = 0, 0
    for (match_id, event_id, _), pool_n, action_n in zip(
            rows, pool_sizes, action_sizes):
        mean_pool = embedded[pool_start:pool_start + pool_n].mean(dim=0)
        mean_extra = extra_tensor[action_start:action_start + action_n].mean(dim=0)
        value = _call(net.draft_value, torch.cat((mean_pool, mean_extra))).reshape(())
        out[(match_id, event_id)] = (
            all_logits[action_start:action_start + action_n], value)
        pool_start += pool_n
        action_start += action_n
    return out


def _selection_inference(net, rows):
    keys = (("own_cat", torch.long), ("foe_cat", torch.long),
            ("own_num", torch.float32), ("foe_num", torch.float32))
    # as_tensor(list_of_tensors) attempts scalar conversion; stack supports
    # both strict-verified JSON arrays and safely decoded tensor caches.
    batched = {key: torch.stack([
        torch.as_tensor(entry[2][key], dtype=dtype) for entry in rows],
        dim=0) for key, dtype in keys}
    logits, _, _, own, foe = _call(
        net.selection, batched, return_embeddings=True)
    inputs = torch.cat((own.mean(dim=1), foe.mean(dim=1)), dim=-1)
    values = _call(net.selection_value, inputs).reshape(len(rows))
    return {(mi, ei): (logits[j], values[j])
            for j, (mi, ei, _) in enumerate(rows)}


def _three_models(model, target, magnet, inference, rows):
    result = {}
    for chunk in rows:
        current = inference(model, chunk)
        with torch.no_grad():
            targets = inference(target, chunk)
            magnets = inference(magnet, chunk)
        for key, (logits, value) in current.items():
            t, tv = targets[key]
            m, _ = magnets[key]
            result[key] = (logits, value, t, tv, m)
    return result


def _battle_inference(model, target, magnet, entries):
    """Wavefront batch independent seat-local GRU histories.

    Every per-seat previous memory is detached after each event. No gradient
    may flow between decisions or Match boundaries; only the current event
    contributes to the shared optimizer.
    """
    sequences = defaultdict(list)
    for match_id, event_id, seat, payload in entries:
        sequences[(match_id, seat)].append((match_id, event_id, payload))
    active = list(sequences)
    states = {
        name: {key: net.initial_memory()[0] for key in sequences}
        for name, net in (("main", model), ("target", target), ("magnet", magnet))
    }
    results = {}
    offset = 0
    # Iterate the kth battle event of each seat independently. Each lane is
    # guaranteed to be sequential even when different Matches have different
    # number of decisions.
    while active:
        wave = [(key, sequences[key][offset]) for key in active
                if len(sequences[key]) > offset]
        if not wave:
            break
        widths = [len(entry[2]["actions"]) for _, entry in wave]
        max_width = max(widths)
        state_input = torch.stack([
            torch.as_tensor(entry[2]["state"], dtype=torch.float32)
            for _, entry in wave], dim=0)
        candidate = torch.zeros(
            (len(wave), max_width, model.battle.action_features),
            dtype=torch.float32)
        for j, (_, entry) in enumerate(wave):
            legal = torch.as_tensor(entry[2]["actions"], dtype=torch.float32)
            assert legal.shape == (widths[j], model.battle.action_features)
            candidate[j, :widths[j]] = legal
        by_net = {}
        for name, net in (("main", model), ("target", target), ("magnet", magnet)):
            memory = torch.stack([states[name][key] for key, _ in wave], dim=0)
            if name == "main":
                logits, values, next_memory = _call(
                    net.battle, state_input, candidate, memory)
            else:
                with torch.no_grad():
                    logits, values, next_memory = _call(
                        net.battle, state_input, candidate, memory)
            by_net[name] = (logits, values)
            for j, (key, _) in enumerate(wave):
                states[name][key] = next_memory[j].detach()
        for j, (_, (mi, ei, _)) in enumerate(wave):
            width = widths[j]
            logits, val = by_net["main"]
            t, tv = by_net["target"]
            m, _ = by_net["magnet"]
            results[(mi, ei)] = (
                logits[j, :width], val[j],
                t[j, :width], tv[j], m[j, :width])
        offset += 1
    return results


def batched_match_losses(model, target, magnet, records_with_roles,
                         *, draft_candidates=MAX_DRAFT_CANDIDATES):
    """Return one unnormalized loss and ratio per Match, as the scalar path.

    A caller must divide summed Match loss by 32 for the SAME 32-Match optimizer
    step, regardless of this function's microbatch size.
    """
    if not records_with_roles:
        raise ValueError("Cannot train on an empty accepted Match microbatch")
    fast_targets = os.environ.get("RNAD_FAST_RNAD_TARGETS", "0")
    if fast_targets not in ("0", "1"):
        raise ValueError("RNAD_FAST_RNAD_TARGETS must be 0 or 1")
    if fast_targets == "1":
        from scripts.rnad.shared_league_fast_targets import rnad_targets_numpy
        target_function = rnad_targets_numpy
    else:
        target_function = rnad_targets
    linear_neurd = os.environ.get("RNAD_LINEARIZED_NEURD", "0")
    if linear_neurd not in ("0", "1"):
        raise ValueError("RNAD_LINEARIZED_NEURD must be 0 or 1")
    if linear_neurd == "1":
        from scripts.rnad.shared_league_linearized_neurd import (
            linearized_neurd_loss_batch,
        )
    draft_rows, selection_rows, battle_rows = [], [], []
    seats_by_match = []
    for mi, (record, role) in enumerate(records_with_roles):
        if role not in ("self", "fixed", "historical", "exploiter"):
            raise ValueError("Unexpected frozen opponent role")
        train_seats = (0, 1) if role == "self" else (record["learner_seat"],)
        seats_by_match.append(train_seats)
        for ei, row in enumerate(record["events"]):
            if row["actor"] not in train_seats:
                continue
            phase = row["phase"]
            if phase == "draft":
                draft_rows.append((mi, ei, row["phase_input"]))
            elif phase == "selection":
                selection_rows.append((mi, ei, row["phase_input"]))
            elif phase == "battle":
                battle_rows.append((mi, ei, row["actor"], row["phase_input"]))
            else:
                raise ValueError("Unexpected authoritative phase")
    predictions = {}
    if draft_rows:
        groups = _chunk_by_size(
            draft_rows, lambda row: len(row[2]["codes"]), draft_candidates)
        predictions.update(_three_models(
            model, target, magnet, _draft_inference, groups))
    if selection_rows:
        groups = _chunk_by_size(
            selection_rows, lambda _: 1, MAX_SELECTION_STATES)
        predictions.update(_three_models(
            model, target, magnet, _selection_inference, groups))
    if battle_rows:
        predictions.update(_battle_inference(model, target, magnet, battle_rows))

    result = []
    for mi, (record, _) in enumerate(records_with_roles):
        terms, inputs = [], []
        for ei, row in enumerate(record["events"]):
            phase = row["phase"]
            if row["actor"] not in seats_by_match[mi]:
                # Frozen opponent rows are public receipts, not learner inputs.
                terms.append({
                    "phase": phase, "actor": row["actor"], "reward": row["reward"],
                    "pi": torch.tensor(row["policy_pi"], dtype=torch.float32),
                    "magnet": torch.tensor(row["magnet_pi"], dtype=torch.float32),
                    "behavior_prob": row["behavior_prob"],
                    "value": row["predicted_value"],
                    "action_index": row["action_index"],
                })
                inputs.append((None, None, phase))
                continue
            logits, value, t, tv, magnet_logits = predictions[(mi, ei)]
            with torch.no_grad():
                target_pi = torch.softmax(t, dim=-1)
                magnet_pi = torch.softmax(magnet_logits, dim=-1)
                original = torch.tensor(row["magnet_pi"], dtype=torch.float32)
                if magnet_pi.shape != original.shape or (
                        torch.max((magnet_pi - original).abs()).item() > 2e-4):
                    raise ValueError("Frozen actor Magnet differs from replayed Magnet")
            terms.append({
                "phase": phase, "actor": row["actor"], "reward": row["reward"],
                "pi": target_pi, "magnet": magnet_pi,
                "behavior_prob": row["behavior_prob"], "value": float(tv),
                "action_index": row["action_index"],
            })
            inputs.append((logits, value, phase))
        targets = target_function(terms, eta=ETA, rho_bar=1.0, c_bar=1.0)
        phase_loss, phase_value = defaultdict(list), defaultdict(list)
        phase_policy = defaultdict(list)
        ratios = []
        for (logits, value, phase), target_row in zip(inputs, targets):
            ratios.append(target_row["ratio"])
            if logits is None:
                continue
            if linear_neurd == "1":
                phase_policy[phase].append((
                    logits, target_row["q_targets"], target_row["c"]))
            else:
                phase_loss[phase].append(neurd_policy_loss(
                    logits, target_row["q_targets"], beta=2.0,
                    q_clip=10.0, is_c=target_row["c"]))
            phase_value[phase].append(
                (value - torch.tensor(
                    target_row["value_target"], dtype=value.dtype)) ** 2)
        if linear_neurd == "1":
            if set(phase_policy) != set(PHASES):
                raise ValueError("Cannot skip a phase when training integrated Main")
            balanced = sum(
                linearized_neurd_loss_batch(phase_policy[p]) +
                VALUE_WEIGHT * sum(phase_value[p]) / len(phase_value[p])
                for p in PHASES
            ) / len(PHASES)
        else:
            if set(phase_loss) != set(PHASES):
                raise ValueError("Cannot skip a phase when training integrated Main")
            balanced = sum(
                sum(phase_loss[p]) / len(phase_loss[p]) +
                VALUE_WEIGHT * sum(phase_value[p]) / len(phase_value[p])
                for p in PHASES
            ) / len(PHASES)
        if not torch.isfinite(balanced):
            raise ValueError("Nonfinite shared NeuRD/V-trace objective")
        result.append((balanced, max(ratios)))
    return result


def vector_batch_size():
    raw = os.environ.get("RNAD_LEARNER_VECTOR_MICROBATCH", "4")
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError("Invalid RNAD_LEARNER_VECTOR_MICROBATCH") from exc
    if size not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Unsafe vector Match microbatch size")
    return size
