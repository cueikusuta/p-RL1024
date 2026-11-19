"""Single-G0 Learner CPU staging study using existing immutable 1280 Matches.

NO Actor or Match simulation is performed. Strictly accept the 20 historical
Actor artifacts once; stage them using an opt-in bounded CPU fast path, then
execute ONE disposable 40-update G0 Learner replay. Nothing produced here is
a valid new-source training continuation, strength result, or Champion.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from scripts.rnad.shared_league_contract import MATCHES_PER_GENERATION
from scripts.rnad.shared_league_full_gen_benchmark import (
    HISTORIC_SOURCE_SHA, _order, _run,
)
from scripts.rnad.shared_league_staging import stage_verified_actors


def run(bootstrap, actors, *, workers=4):
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("Bounded staging worker count must be 1, 2 or 4")
    bootstrap, actors = Path(bootstrap), Path(actors)
    manifest = json.loads((bootstrap / "generation-manifest.json").read_text())
    if (manifest.get("source_sha") != HISTORIC_SOURCE_SHA
            or manifest.get("generation") != 0):
        raise ValueError("Historical G0 provenance does not match pinned study")
    paths = [actors / f"actor-{i}" / f"actor-{i}.json.gz" for i in range(20)]

    # ONE staging pass and ONE 40-update Learner replay. The historical
    # unmodified G0 log is the external baseline; do not re-run it in CI.
    with tempfile.TemporaryDirectory(prefix="rnad-i5-one-g0-") as temp:
        scratch = Path(temp)
        accepted = scratch / "accepted"
        started = time.perf_counter()
        ids, meta, roles, metrics = stage_verified_actors(
            manifest, paths, accepted, workers=workers, fast_json=True,
            profile=True, return_metrics=True)
        prep_seconds = time.perf_counter() - started
        if len(ids) != MATCHES_PER_GENERATION or len(meta) != len(ids):
            raise ValueError("Incomplete 20/20 strict accepted Match barrier")
        ordered = _order(meta)
        if len(ordered) != 1280:
            raise ValueError("Wrong immutable 40-update ordering")
        compute_start = time.perf_counter()
        result = _run(
            bootstrap, ordered, accepted, "cpu-parallel-vector",
            scratch / "disposable-learner")
        compute_seconds = time.perf_counter() - compute_start
        if result["optimizer_updates"] != 40 or (
                result["accepted_complete_matches"] != 1280):
            raise ValueError("Missing 40-update G0 research replay")
        report = {
            "schema": "rnad-i5-historical-single-g0-learner-profile-v1",
            "original_source_sha": HISTORIC_SOURCE_SHA,
            "generations_replayed": 1,
            "actor_simulations_run": 0,
            "strictly_verified_actors": 20,
            "accepted_complete_four_win_matches": len(ids),
            "optimizer_updates": result["optimizer_updates"],
            "staging_workers": workers,
            "fast_json": True,
            "preparation_wall_seconds": round(prep_seconds, 3),
            "preparation_cpu_phase_sum_seconds": metrics["sum_actor_phase_seconds"],
            "peak_stage_worker_rss_kib": metrics["max_worker_rss_kib"],
            "forty_update_wall_seconds": round(compute_seconds, 3),
            "forty_update_gradient_seconds": result["total_grad_seconds"],
            "forty_update_checkpoint_seconds": result["total_checkpoint_save_seconds"],
            "combined_stage_and_40_update_seconds": round(
                prep_seconds + compute_seconds, 3),
            "preparation_at_most_60_seconds": prep_seconds <= 60.0,
            "actor_role_quota": dict(roles),
            "checkpoint_published": False,
            "not_source_correct_continuation": True,
            "model_strength_claimed": False,
            "champion_promoted": False,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", required=True)
    ap.add_argument("--actors", required=True)
    ap.add_argument("--workers", type=int, default=4, choices=(1, 2, 4))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = run(a.bootstrap, a.actors, workers=a.workers)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
