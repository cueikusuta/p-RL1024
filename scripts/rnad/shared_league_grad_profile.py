"""Research-only per-phase CPU profile on strictly accepted G0 Matches.

This profiles ONE unchanged 32-Match optimizer minibatch WITHOUT an update.
It does not estimate full-generation throughput or AI strength. Frozen source
and Match receipts must pass full strict manifest/actor checks before profiling.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.rnad import shared_league_batched as batched
from scripts.rnad.shared_league_bootstrap import digest
from scripts.rnad.shared_league_contract import verify_actor, verify_manifest
from scripts.rnad.shared_league_learner import _models


_TIMED = ("_draft_inference", "_selection_inference",
          "_battle_inference", "rnad_targets")


def profile(bootstrap, actor, *, reuse_tensors=False):
    root = Path(bootstrap)
    manifest = json.loads((root / "generation-manifest.json").read_text())
    verify_manifest(manifest)
    with gzip.open(actor, "rt") as stream:
        payload = json.load(stream)
    seen = verify_actor(manifest, payload)
    if payload["actor_id"] != 0 or len(seen) != 64:
        raise ValueError("Profile requires one fully verified 64-Match Actor0")
    ckpt_path = root / "checkpoint-g0.pt"
    ckpt_index = json.loads((root / "checkpoint-index.json").read_text())
    if digest(ckpt_path) != ckpt_index["sha256"]:
        raise ValueError("Pinned bootstrap checkpoint SHA mismatch")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if checkpoint.get("source_sha") != manifest["source_sha"]:
        raise ValueError("Mismatched frozen training source")
    torch.set_num_threads(1)
    model, target, magnet, _, _ = _models(root, checkpoint)
    model.zero_grad(set_to_none=True)
    rows = sorted(payload["matches"], key=lambda row: row["match_id"])[:32]
    if len(rows) != 32 or len({row["match_id"] for row in rows}) != 32:
        raise ValueError("Need 32 distinct accepted complete Matches")
    elapsed = defaultdict(float)
    counts = defaultdict(int)

    def timed(label, fn):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed[label] += time.perf_counter() - start
                counts[label] += 1
        return wrapped

    losses, ratios = [], []
    start_all = time.perf_counter()
    with ExitStack() as stack:
        if reuse_tensors:
            stack.enter_context(patch.dict(
                os.environ, {"RNAD_REUSE_FROZEN_PHASE_TENSORS": "1"}))
        for label in _TIMED:
            stack.enter_context(patch.object(
                batched, label, timed(label, getattr(batched, label))))
        for start in range(0, 32, 4):
            batch = [(row, payload["role"]) for row in rows[start:start + 4]]
            start_forward = time.perf_counter()
            result = batched.batched_match_losses(model, target, magnet, batch)
            elapsed["forward_total"] += time.perf_counter() - start_forward
            start_backward = time.perf_counter()
            (sum(loss for loss, _ in result) / 32).backward()
            elapsed["backward_total"] += time.perf_counter() - start_backward
            losses.extend(float(loss.detach()) for loss, _ in result)
            ratios.extend(float(ratio) for _, ratio in result)
    elapsed["total"] = time.perf_counter() - start_all
    tensors = [param.grad for param in model.parameters()
               if param.requires_grad]
    if (len(losses) != 32 or len(ratios) != 32 or
            not all(math.isfinite(x) for x in losses + ratios) or
            not tensors or any(g is None or not torch.isfinite(g).all()
                               for g in tensors)):
        raise ValueError("Incomplete or nonfinite original gradient")
    return {
        "schema": "rnad-i5-real-accepted-32-match-gradient-hotspots-v1",
        "original_source_sha": manifest["source_sha"],
        "experimental_code_not_authoritative_source_replay": True,
        "full_verified_actor_matches": len(seen),
        "examined_matches": len(rows),
        "accepted_decisions": sum(len(row["events"]) for row in rows),
        "matched_minibatch_role": payload["role"],
        "microbatch_matches": 4,
        "reuse_frozen_phase_tensors": reuse_tensors,
        "single_optimizer_updates": 0,
        "all_gradients_finite": True,
        "timing_is_single_diagnostic_not_full_generation": True,
        "phase_wall_seconds": {key: round(value, 6)
                               for key, value in sorted(elapsed.items())},
        "calls": dict(sorted(counts.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--reuse-tensors", action="store_true")
    args = parser.parse_args()
    print(json.dumps(profile(args.bootstrap, args.actor,
                             reuse_tensors=args.reuse_tensors), indent=2),
          flush=True)


if __name__ == "__main__":
    main()
