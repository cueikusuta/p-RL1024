"""Real accepted-Match comparison: scalar vs vector learner.

Strictly reuses four one-use authoritative Match artifacts and a pinned
disposable bootstrap. This is a correctness benchmark, NOT a strength study.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import torch

from scripts.rnad.shared_league_contract import _verify_events, verify_manifest
from scripts.rnad.shared_league_learner import _match_loss, _models
from scripts.rnad.shared_league_batched import batched_match_losses
from scripts.rnad.train import frozen_engine_sha


def _data(root, source):
    manifest = json.loads((root/"generation-manifest.json").read_text())
    verify_manifest(manifest)
    if manifest["engine_sha"] != frozen_engine_sha():
        raise ValueError("Frozen authority engine mismatch")
    rows = []
    for path in sorted(source.rglob("*.json.gz")):
        with gzip.open(path, "rt") as f:
            match = json.load(f)
        if (match["source_sha"] != manifest["source_sha"] or
                match["engine_sha"] != manifest["engine_sha"] or
                match["policy_id"] != manifest["policy_id"] or
                match["completed"] is not True or
                match["score"][match["winner"]] != 4):
            raise ValueError("Not an accepted pinned complete four-win Match")
        _verify_events(match)
        rows.append((match, "fixed" if "frozen" in path.parts else "self"))
    if len(rows) != 4 or len({m["match_id"] for m, _ in rows}) != 4:
        raise ValueError("Require exactly four accepted independent smoke Matches")
    return rows


def _gradients(net):
    return {k: p.grad.detach().clone()
            for k, p in net.named_parameters() if p.grad is not None}


def _reference(root, checkpoint, rows):
    model, target, magnet, optimizer, scheduler = _models(root, checkpoint)
    optimizer.zero_grad(set_to_none=True)
    before = time.perf_counter()
    outputs = []
    for match, role in rows:
        loss, ratio = _match_loss(model, target, magnet, match, role=role)
        (loss / len(rows)).backward()
        outputs.append((float(loss.detach()), ratio))
    return model, optimizer, scheduler, outputs, _gradients(model), time.perf_counter() - before


def _vector(root, checkpoint, rows):
    model, target, magnet, optimizer, scheduler = _models(root, checkpoint)
    optimizer.zero_grad(set_to_none=True)
    before = time.perf_counter()
    outputs = batched_match_losses(model, target, magnet, rows)
    (sum(loss for loss, _ in outputs) / len(rows)).backward()
    elapsed = time.perf_counter() - before
    return (model, optimizer, scheduler,
            [(float(x.detach()), ratio) for x, ratio in outputs],
            _gradients(model), elapsed)


def compare(bootstrap, smoke):
    torch.set_num_threads(1)
    root, source = Path(bootstrap), Path(smoke)
    rows = _data(root, source)
    checkpoint = torch.load(root/"checkpoint-g0.pt",
                            map_location="cpu", weights_only=True)
    a = _reference(root, checkpoint, rows)
    b = _vector(root, checkpoint, rows)
    for (loss0, ratio0), (loss1, ratio1) in zip(a[3], b[3]):
        if abs(loss0-loss1) > 2e-4 or abs(ratio0-ratio1) > 2e-4:
            raise AssertionError(
                f"Full accepted Match loss/ratio parity: {loss0, loss1, ratio0, ratio1}")
    if set(a[4]) != set(b[4]):
        raise AssertionError("Gradient keys differ")
    grad_max = 0.
    for name in a[4]:
        left, right = a[4][name], b[4][name]
        grad_max = max(grad_max, float((left - right).abs().max()))
        torch.testing.assert_close(left, right, rtol=3e-3, atol=3e-5,
                                   msg=lambda msg: f"{name}: {msg}")
    for model, optimizer, scheduler in (a[:3], b[:3]):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.,
                                       error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
    step_max = 0.
    for (name, p), (_, q) in zip(a[0].named_parameters(), b[0].named_parameters()):
        step_max = max(step_max, float((p-q).abs().max()))
        torch.testing.assert_close(p, q, rtol=2e-3, atol=3e-5,
                                   msg=lambda msg: f"{name}: {msg}")
    return {
        "schema":"rnad-i5-real-match-scalar-vs-vector-parity-v1",
        "accepted_matches":len(rows),
        "accepted_decisions":sum(len(match["events"]) for match, _ in rows),
        "mean_loss_scalar":sum(x[0] for x in a[3])/len(rows),
        "mean_loss_vector":sum(x[0] for x in b[3])/len(rows),
        "maximum_absolute_gradient_diff":grad_max,
        "maximum_absolute_optimizer_weight_diff":step_max,
        "scalar_wall_seconds":a[-1],
        "vector_wall_seconds":b[-1],
        "one_trial_speed_ratio_not_evidence":a[-1]/b[-1],
        "champion_or_strength_claim":False,
        "generation_completed":False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--smoke", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.bootstrap, args.smoke), indent=2), flush=True)
