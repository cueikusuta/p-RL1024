"""One optimizer over 1280 accepted full four-win Matches, 40 updates.

All 20 actor artifacts are verified before the first backward pass. The two-
player V-trace/NeuRD target is built on accepted chronological seat order.
Only a single shared Main is changed; the old frozen opponents are NEVER
updated. One optimizer step follows gradients from 32 different full Matches.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

from scripts.rnad.integrated_policy_v1 import IntegratedPolicyV1
from scripts.rnad.shared_league_bootstrap import (
    POLICY_ID, digest, export_bundle, write_json,
)
from scripts.rnad.shared_league_contract import (
    OPTIMIZER_BATCH_MATCHES, OPTIMIZER_STEPS, verify_actor,
    verify_manifest, canonical_hash,
)
from scripts.rnad.train import current_git_sha, frozen_engine_sha
from scripts.rnad.two_player import rnad_targets, neurd_policy_loss
from scripts.rnad.train_selection_specialist import TRIPLES

SCHEMA = "rnad-shared-learner-full-state-v1"
ETA = .2
VALUE_WEIGHT = .5
DRAFT_POLICY_GRADIENT_WEIGHT = 3.0  # Bounded surrogate scale; no reward/Q change.
VALUE_TERMINAL_AUXILIARY_WEIGHT = .10  # Post-acceptance training label only.
LR = 3e-5
MAGNET_CADENCE = 4
SHUFFLE_SEED = 11520261118


def _phase_policy_weight(phase):
    if phase not in ("draft", "selection", "battle"):
        raise ValueError("Unknown trainable phase")
    return DRAFT_POLICY_GRADIENT_WEIGHT if phase == "draft" else 1.0


def _critic_loss(value, target, *, winner, actor):
    """R-NaD V-trace plus accepted terminal target, never a policy input."""
    if winner not in (0, 1) or actor not in (0, 1):
        raise ValueError("Invalid winner or actor")
    trace = torch.as_tensor(target, dtype=value.dtype, device=value.device)
    terminal = value.new_tensor(1.0 if actor == winner else -1.0)
    return (value - trace).square() + VALUE_TERMINAL_AUXILIARY_WEIGHT * (value - terminal).square()


def _load(path):
    """Decode immutable actor gzip or strict-verified plain JSON.

    The opt-in orjson path is only used after the strict 20/20 receipt/manifest
    barrier. It preserves Python JSON numbers and never skips verification.
    The reference stdlib parser remains the default and comparison baseline.
    """
    path = Path(path)
    codec = os.environ.get("RNAD_LEARNER_JSON_CODEC", "stdlib")
    if codec == "orjson":
        try:
            import orjson
        except ImportError as exc:
            raise RuntimeError("Pinned orjson required by learner codec") from exc
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as source:
                return orjson.loads(source.read())
        return orjson.loads(path.read_bytes())
    if codec != "stdlib":
        raise ValueError("RNAD_LEARNER_JSON_CODEC must be stdlib or orjson")
    with (gzip.open(path, "rt") if path.suffix == ".gz"
          else path.open("r", encoding="utf-8")) as source:
        return json.load(source)


def _order(rows):
    ids = sorted(rows, key=lambda m: m["match_id"])
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(ids)
    return ids


def _shape_model(actor_dir):
    import json
    source = json.loads((Path(actor_dir) / "selection.json").read_text())
    return IntegratedPolicyV1(source["public_actor_vocabulary"])


def _models(bootstrap, checkpoint):
    actor_dir = Path(bootstrap) / "actor"
    model = _shape_model(actor_dir)
    target = copy.deepcopy(model)
    magnet = copy.deepcopy(model)
    for net, key in ((model, "model"), (target, "target"),
                     (magnet, "magnet")):
        net.load_state_dict(checkpoint[key], strict=True)
    for frozen in (target, magnet):
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda _: 1.0)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["torch_rng"])
    random.setstate(checkpoint["python_rng"])
    return model, target, magnet, optimizer, scheduler


def _inputs(phase, p):
    if phase == "draft":
        return ({
            "pool": torch.tensor(p["pool"], dtype=torch.float32),
            "extra": torch.tensor(p["extra"], dtype=torch.float32),
        }, torch.tensor(p["codes"], dtype=torch.long))
    if phase == "selection":
        d = {k: torch.tensor([p[src]], dtype=dt)
             for k, src, dt in (
                 ("own_cat", "own_cat", torch.long),
                 ("foe_cat", "foe_cat", torch.long),
                 ("own_num", "own_num", torch.float32),
                 ("foe_num", "foe_num", torch.float32),
             )}
        return d, TRIPLES
    return ({"state": torch.tensor([p["state"]], dtype=torch.float32)},
            torch.tensor([p["actions"]], dtype=torch.float32))


def _selection_batch(model, target, magnet, events, train_seats):
    """Batch independent selection-head forwards, never recurrent Battle."""
    indices = [i for i, row in enumerate(events)
               if row["actor"] in train_seats and row["phase"] == "selection"]
    if not indices:
        return {}
    fields = (("own_cat", torch.long), ("foe_cat", torch.long),
              ("own_num", torch.float32), ("foe_num", torch.float32))
    observations = {
        key: torch.tensor([events[i]["phase_input"][key] for i in indices],
                          dtype=dtype)
        for key, dtype in fields
    }
    logits, values = model.selection_forward_batch(observations)
    with torch.no_grad():
        target_logits, target_values = target.selection_forward_batch(observations)
        magnet_logits, _ = magnet.selection_forward_batch(observations)
    return {index: (logits[j], values[j], target_logits[j],
                    target_values[j], magnet_logits[j])
            for j, index in enumerate(indices)}


def _match_loss(model, target, magnet, record, *, role):
    own_seat = record["learner_seat"]
    train_seats = (0, 1) if role == "self" else (own_seat,)
    memories = {key: [net.initial_memory() for _ in (0, 1)]
                for key, net in (("main", model), ("target", target),
                                 ("magnet", magnet))}
    inputs, terms = [], []
    selection_outputs = _selection_batch(
        model, target, magnet, record["events"], train_seats)
    for event_index, row in enumerate(record["events"]):
        seat, phase = row["actor"], row["phase"]
        if seat not in train_seats:
            # Frozen agent probabilities/values were logged only AFTER
            # authoritative Match acceptance and strict identity validation.
            terms.append({
                "phase": phase, "actor": seat, "reward": row["reward"],
                "pi": torch.tensor(row["policy_pi"], dtype=torch.float32),
                "magnet": torch.tensor(row["magnet_pi"], dtype=torch.float32),
                "behavior_prob": row["behavior_prob"],
                "value": row["predicted_value"],
                "action_index": row["action_index"],
            })
            inputs.append((None, None, phase))
            continue
        if phase == "selection":
            logits, value, t, t_value, m = selection_outputs[event_index]
        else:
            observation, legal = _inputs(phase, row["phase_input"])
            logits, value, next_memory = model(
                phase, observation, legal, memories["main"][seat])
            if phase == "battle":
                memories["main"][seat] = next_memory.detach()
            with torch.no_grad():
                t, t_value, next_t = target(
                    phase, observation, legal, memories["target"][seat])
                m, _, next_m = magnet(
                    phase, observation, legal, memories["magnet"][seat])
                if phase == "battle":
                    memories["target"][seat] = next_t
                    memories["magnet"][seat] = next_m
        with torch.no_grad():
            target_pi, magnet_pi = torch.softmax(t, -1), torch.softmax(m, -1)
            original = torch.tensor(row["magnet_pi"], dtype=torch.float32)
            if magnet_pi.shape != original.shape or (
                torch.max((magnet_pi - original).abs()).item() > 2e-4):
                raise ValueError("Frozen actor Magnet differs from replayed Magnet")
        terms.append({
            "phase": phase, "actor": seat, "reward": row["reward"],
            "pi": target_pi, "magnet": magnet_pi,
            "behavior_prob": row["behavior_prob"], "value": float(t_value),
            "action_index": row["action_index"],
        })
        inputs.append((logits, value, phase))
    targets = rnad_targets(terms, eta=ETA, rho_bar=1.0, c_bar=1.0)
    losses = defaultdict(list)
    critic = defaultdict(list)
    ratios = []
    for (logits, value, phase), result, event in zip(inputs, targets, record["events"]):
        ratios.append(result["ratio"])
        if logits is None:
            continue
        losses[phase].append(_phase_policy_weight(phase) * neurd_policy_loss(
            logits, result["q_targets"], beta=2., q_clip=10.,
            is_c=result["c"]))
        critic[phase].append(_critic_loss(
            value, result["value_target"], winner=record["winner"],
            actor=event["actor"]))
    if set(losses) != {"draft", "selection", "battle"}:
        raise ValueError("Cannot skip a phase when training integrated Main")
    balanced = sum(
        sum(losses[p]) / len(losses[p]) +
        VALUE_WEIGHT * sum(critic[p]) / len(critic[p])
        for p in ("draft", "selection", "battle")
    ) / 3.
    if not torch.isfinite(balanced):
        raise ValueError("Nonfinite shared NeuRD/V-trace objective")
    return balanced, max(ratios)



def _verified_phase_tensors(row):
    """Only after verify_actor: convert public/self tensors without labels."""
    phase_dtype = {
        "draft": (("pool", torch.float32), ("codes", torch.long),
                  ("extra", torch.float32)),
        "selection": (("own_cat", torch.long), ("foe_cat", torch.long),
                      ("own_num", torch.float32),
                      ("foe_num", torch.float32)),
        "battle": (("state", torch.float32), ("actions", torch.float32)),
    }
    packed = dict(row)
    packed["events"] = []
    for event in row["events"]:
        phase = event["phase"]
        if phase not in phase_dtype:
            raise ValueError("Invalid phase for strict tensor cache")
        copy_event = dict(event)
        payload = dict(event["phase_input"])
        for key, dtype in phase_dtype[phase]:
            payload[key] = torch.as_tensor(payload[key], dtype=dtype)
        copy_event["phase_input"] = payload
        packed["events"].append(copy_event)
    return packed


def _cached_verified(root, choice, *, tensor_cache):
    name = choice["match_id"]
    if not tensor_cache:
        return _load(root / (name+".json"))
    path = root / (name+".pt")
    if digest(path) != choice["tensor_cache_sha256"]:
        raise ValueError("Corrupt strict tensor cache after acceptance")
    # Safe tensors and primitive metadata only; no arbitrary pickle objects.
    row = torch.load(path, map_location="cpu", weights_only=True)
    if row.get("match_id") != name or row.get("completed") is not True:
        raise ValueError("Tensor cache does not match verified Match")
    return row


def _atomic_checkpoint(path, ckpt):
    path=Path(path)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    index=path.with_suffix(".json")
    body={"schema":SCHEMA,"generation":ckpt["generation"],
          "step":ckpt["step"],"matches":len(ckpt["training_match_ids"]),
          "sha256":digest(path),"is_champion":False}
    index_tmp=index.with_suffix(".json.tmp")
    index_tmp.write_text(json.dumps(body,sort_keys=True)+"\n")
    os.replace(index_tmp,index)


def learn(*, bootstrap, actors, out, original_draft, original_selection_js,
          max_steps=OPTIMIZER_STEPS):
    if type(max_steps) is not int or not 1 <= max_steps <= OPTIMIZER_STEPS:
        raise ValueError("Invalid bounded explicit optimizer budget")
    root, output = Path(bootstrap), Path(out)
    manifest=json.loads((root/"generation-manifest.json").read_text())
    verify_manifest(manifest)
    if (manifest["source_sha"] != current_git_sha() or
        manifest["engine_sha"] != frozen_engine_sha()):
        raise ValueError("Wrong frozen learner code/game engine")
    paths=[Path(actors) / f"actor-{i}" / f"actor-{i}.json.gz"
           for i in range(20)]
    if any(not x.is_file() for x in paths):
        raise ValueError("Cannot train on incomplete 20/20 Actor artifact set")
    # Verify each of the 20 immutable actor shards before any optimization.
    # Write verified Matches back to disk one-by-one; keeping all 1280 giant
    # variable-width draft policy tensors in Python objects can exceed a
    # hosted runner's RAM even though the gzip artifacts fit on disk.
    verified = Path(actors) / "_strict_verified_matches"
    verified.mkdir(exist_ok=True)
    cache_flag = os.environ.get("RNAD_VERIFIED_TENSOR_CACHE", "0")
    if cache_flag not in ("0", "1"):
        raise ValueError("Invalid strict verified tensor cache flag")
    tensor_cache = cache_flag == "1"
    if tensor_cache and os.environ.get("RNAD_LEARNER_BACKEND", "reference") != "vector":
        raise ValueError("Tensor cache needs the experimental vector Learner")
    # All strict receipt/provenance checks finish before the first optimizer
    # update. Opt-in stage variants are compared on exactly the same inputs.
    from scripts.rnad.shared_league_staging import (
        stage_verified_actors, staging_config,
    )
    prep_workers, fast_json, profile_staging, prep_codec = staging_config()
    match_ids, meta, roles = stage_verified_actors(
        manifest, paths, verified, tensor_cache=tensor_cache,
        workers=prep_workers, fast_json=fast_json, codec=prep_codec,
        profile=profile_staging,
    )
    expected_roles = {"self": 8, "historical": 8,
                      "fixed": 2, "exploiter": 2}
    if not manifest["opponents"]["exploiter"]:
        expected_roles["historical"] += expected_roles.pop("exploiter")
    if (len(match_ids) != 1280 or
        set(roles) != set(expected_roles) or
        dict(roles) != expected_roles):
        raise ValueError("Must finish all 20 Actors and 1280 unique paired Matches")
    report = {
        "generation": manifest["generation"],
        "independent_seed_clusters": 640,
        "accepted_complete_four_win_matches": 1280,
        "optimizer_minibatch_matches": 32,
        "optimizer_steps_one_epoch": 40,
        "manifest_sha256": canonical_hash(manifest),
        "opponent_roles": dict(roles),
    }
    ordering = _order(meta)
    ids = [x["match_id"] for x in ordering]
    order_sha=hashlib.sha256(("\n".join(ids)+"\n").encode()).hexdigest()
    checkpoint_file=root/"checkpoint-g0.pt"
    index=json.loads((root/"checkpoint-index.json").read_text())
    if digest(checkpoint_file)!=index["sha256"]:
        raise ValueError("Corrupted genesis full-state shared learner")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path=output/"checkpoint-working.pt"
    if checkpoint_path.exists():
        index=json.loads(checkpoint_path.with_suffix(".json").read_text())
        if digest(checkpoint_path)!=index["sha256"]:
            raise ValueError("Corrupted resumable shared full-state checkpoint")
        ckpt=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    else:
        ckpt=torch.load(checkpoint_file,map_location="cpu",weights_only=True)
        ckpt["order_sha256"]=order_sha
    if (ckpt.get("schema")!=SCHEMA or
        ckpt.get("generation") != manifest["generation"] or
        ckpt.get("source_sha") != current_git_sha() or
        ckpt.get("engine_sha") != manifest["engine_sha"] or
        ckpt.get("policy_id") != manifest["policy_id"] or
        ckpt.get("manifest_sha256") != canonical_hash(manifest) or
        ckpt.get("main_sha256") != manifest["main_sha256"] or
        ckpt.get("order_sha256") != order_sha or
        ckpt.get("training_match_ids") != ids[:ckpt.get("step",0)*32]):
        raise ValueError("Invalid checkpoint model, replay order or training prefix")
    model,target,magnet,optimizer,scheduler=_models(root,ckpt)
    torch.set_num_threads(1)
    backend = os.environ.get("RNAD_LEARNER_BACKEND", "reference")
    if backend not in ("reference", "vector", "parallel"):
        raise ValueError("Unknown research-only Learner backend")
    if backend == "parallel":
        from scripts.rnad.shared_league_parallel import (
            ParallelGradientPool, configured_workers,
        )
        vector_workers = os.environ.get("RNAD_PARALLEL_VECTOR", "0")
        if vector_workers not in ("0","1"):
            raise ValueError("RNAD_PARALLEL_VECTOR must be 0 or 1")
        pool_context = ParallelGradientPool(
            model, target, magnet, workers=configured_workers(),
            vector=vector_workers == "1")
    else:
        pool_context = nullcontext(None)
    with pool_context as parallel_pool:
        for step in range(ckpt["step"],max_steps):
            selected=ordering[32*step:32*(step+1)]
            if len(selected)!=32 or len(set(x["match_id"] for x in selected))!=32:
                raise ValueError("Missing no-reuse minibatch")
            optimizer.zero_grad(set_to_none=True)
            losses,ratios=[],[]
            if backend == "reference":
                for choice in selected:
                    record = _load(verified / (choice["match_id"] + ".json"))
                    loss,ratio=_match_loss(model,target,magnet,record,
                                           role=choice["role"])
                    (loss/32).backward()
                    del record
                    losses.append(float(loss.detach()))
                    ratios.append(ratio)
            elif backend == "vector":
                # Preserve all 32 Match IDs, exact loss normalization and the
                # single optimizer update; bound graph peak memory by microbatch.
                from scripts.rnad.shared_league_batched import (
                    batched_match_losses, vector_batch_size,
                )
                size = vector_batch_size()
                for start in range(0, len(selected), size):
                    micro = selected[start:start+size]
                    loaded = [(_cached_verified(verified, x, tensor_cache=tensor_cache),
                               x["role"]) for x in micro]
                    results = batched_match_losses(model, target, magnet, loaded)
                    (sum(loss for loss, _ in results)/32).backward()
                    losses.extend(float(loss.detach()) for loss, _ in results)
                    ratios.extend(ratio for _, ratio in results)
                    del results, loaded
            elif backend == "parallel":
                tasks = [
                    (str(verified/(x["match_id"]+(".pt" if tensor_cache else ".json"))),
                     x["role"],
                     x.get("tensor_cache_sha256") if tensor_cache else None)
                    for x in selected
                ]
                losses,ratios = parallel_pool.step_gradients(tasks)
            norm=torch.nn.utils.clip_grad_norm_(
                model.parameters(),10.0,error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite minibatch gradient")
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for dst,src in zip(target.parameters(),model.parameters()):
                    dst.mul_(.99).add_(src,alpha=.01)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite integrated Main after optimizer step")
            ckpt.update({
                "step": step+1, "model":model.state_dict(),
                "target":target.state_dict(), "magnet":magnet.state_dict(),
                "optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "torch_rng":torch.get_rng_state(),
                "python_rng":random.getstate(),
                "training_match_ids":ids[:32*(step+1)],
            })
            # snapshot state_dict values before mutating next minibatch.
            ckpt["model"]={k:v.clone() for k,v in model.state_dict().items()}
            ckpt["target"]={k:v.clone() for k,v in target.state_dict().items()}
            _atomic_checkpoint(checkpoint_path,ckpt)
            print(json.dumps({"step":step+1,"steps_total":40,
                              "accepted_matches_applied":32*(step+1),
                              "mean_three_phase_loss":sum(losses)/len(losses),
                              "max_off_policy_ratio":max(ratios),
                              "research_only":True}),flush=True)
    if ckpt["step"] != OPTIMIZER_STEPS:
        return {"complete":False,"step":ckpt["step"]}
    # A complete generation is the ONLY legal boundary for a new Actor
    # Main/Magnet snapshot. The Magnet changes once every four generations;
    # it must remain bit-for-bit frozen for all 20 Actors within a generation.
    completed_gen=manifest["generation"]
    next_gen=completed_gen+1
    if next_gen % MAGNET_CADENCE == 0:
        magnet.load_state_dict(model.state_dict(), strict=True)
    export_dir=output/f"generation{next_gen}-actor"
    out_sha=export_bundle(model,generation=next_gen,
                          original_draft=original_draft,
                          selection_template=original_selection_js,
                          engine_sha=manifest["engine_sha"],
                          code_sha=manifest["source_sha"],
                          out=export_dir)
    next_mag=output/f"generation{next_gen}-magnet"
    magnet_sha=export_bundle(magnet,generation=next_gen,
                             original_draft=original_draft,
                             selection_template=original_selection_js,
                             engine_sha=manifest["engine_sha"],
                             code_sha=manifest["source_sha"],out=next_mag)
    finished=output/f"checkpoint-g{next_gen}.pt"
    ckpt["generation"]=next_gen
    ckpt["step"]=0
    ckpt["main_sha256"]=out_sha
    ckpt["magnet_sha256"]=magnet_sha
    ckpt["completed_generation_match_ids"]=ids
    ckpt["training_match_ids"]=[]
    ckpt.pop("order_sha256",None)
    ckpt["cumulative_optimizer_updates"]=(
        int(ckpt.get("cumulative_optimizer_updates",completed_gen*40))+40)
    ckpt["cumulative_complete_matches"]=(
        int(ckpt.get("cumulative_complete_matches",completed_gen*1280))+1280)
    ckpt["magnet"]={k:v.clone() for k,v in magnet.state_dict().items()}
    ckpt["is_champion"]=False
    torch.save(ckpt,finished)
    write_json(output/f"generation{next_gen}-index.json",{
        "schema":SCHEMA,"generation":next_gen,
        "main_sha256":out_sha,"magnet_sha256":magnet_sha,
        "full_checkpoint_sha256":digest(finished),
        "accepted_complete_matches":len(ids),
        "optimizer_updates":40,
        "cumulative_optimizer_updates":ckpt["cumulative_optimizer_updates"],
        "cumulative_complete_matches":ckpt["cumulative_complete_matches"],
        "research_only":True,
        "champion_promoted":False,
        "source_manifest_sha256":canonical_hash(manifest),
    })
    result={**report,"optimizer_updates_completed":40,
            "next_generation":next_gen,
            "single_writer_main_sha256":out_sha,
            "frozen_magnet_sha256":magnet_sha,
            "magnet_rotated":next_gen%MAGNET_CADENCE==0,
            "cumulative_optimizer_updates":ckpt["cumulative_optimizer_updates"],
            "cumulative_complete_matches":ckpt["cumulative_complete_matches"],
            "research_only":True,"champion_promoted":False,
            "independent_review_completed":False,
            "three_phase_top1_gate_passed":False,
            "real_match_strength_established":False}
    write_json(output/f"generation{next_gen}-report.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ("bootstrap","actors","out","original-draft",
                 "original-selection-js"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--max-steps",type=int,default=OPTIMIZER_STEPS)
    a=vars(parser.parse_args())
    result=learn(
        bootstrap=a["bootstrap"],actors=a["actors"],out=a["out"],
        original_draft=a["original_draft"],
        original_selection_js=a["original_selection_js"],
        max_steps=a["max_steps"])
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
