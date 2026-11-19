"""Research-only complete-Match Learner staging and 40-update wall benchmark.

Pinned historical G0 Inputs are accepted, immutable four-win Matches, not
synthetic examples. This is an OFFLINE REPLAY using newer research code, not a
valid source-pinned continuation or a Champion/strength evaluation. The model
checkpoints written for comparison are deleted with their temporary folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

from scripts.rnad.shared_league_contract import MATCHES_PER_GENERATION
from scripts.rnad.shared_league_full_gen_benchmark import (
    HISTORIC_SOURCE_SHA, _order, _run,
)
from scripts.rnad.shared_league_staging import stage_verified_actors


def _digest_matches(root, meta):
    return {entry["match_id"]: hashlib.sha256(
        (root / (entry["match_id"] + ".json")).read_bytes()).hexdigest()
        for entry in meta}


def run(bootstrap, actors, *, repeats=2):
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("Bounded CPU benchmark repetitions must be 1..3")
    bootstrap, actors = Path(bootstrap), Path(actors)
    manifest = json.loads((bootstrap / "generation-manifest.json").read_text())
    if (manifest.get("source_sha") != HISTORIC_SOURCE_SHA or
            manifest.get("generation") != 0):
        raise ValueError("Historical provenance does not match pinned study")
    paths = [actors / f"actor-{i}" / f"actor-{i}.json.gz" for i in range(20)]
    measurements = []
    for repetition in range(repeats):
        original_hashes = None
        reference_model = None
        for label, workers, fast in (
                ("reference", 1, False),
                ("fast-one", 1, True),
                ("fast-two", 2, True),
                ("fast-four", 4, True)):
            with tempfile.TemporaryDirectory(
                    prefix=f"rnad-i5-{label}-") as temp:
                scratch = Path(temp)
                accepted = scratch / "accepted"
                start = time.perf_counter()
                ids, meta, roles, metrics = stage_verified_actors(
                    manifest, paths, accepted, workers=workers, fast_json=fast,
                    profile=True, return_metrics=True)
                prep_seconds = time.perf_counter() - start
                if len(ids) != MATCHES_PER_GENERATION or len(meta) != len(ids):
                    raise ValueError("Incomplete strict 1280 Match benchmark")
                checksums = _digest_matches(accepted, meta)
                if original_hashes is None:
                    original_hashes = checksums
                elif checksums != original_hashes:
                    raise ValueError(
                        "JSON staging byte-for-byte mismatch on real Matches")
                out = {
                    "repetition": repetition + 1,
                    "label": label,
                    "workers": workers,
                    "fast_json": fast,
                    "stage_wall_seconds": round(prep_seconds, 3),
                    "phase_cpu_seconds": metrics["sum_actor_phase_seconds"],
                    "max_worker_rss_kib": metrics["max_worker_rss_kib"],
                    "all_1280_sha256_equal": True,
                    "complete_matches": len(ids),
                    "roles": dict(roles),
                    "not_a_strength_claim": True,
                }
                # Full 40-update identical-backend dry-runs for the reference
                # and one bounded candidate; no trained artifact is published.
                if label in ("reference", "fast-two"):
                    ordered = _order(meta)
                    if len(ordered) != 1280:
                        raise ValueError("Wrong immutable 40-update ordering")
                    compute_started = time.perf_counter()
                    result = _run(
                        bootstrap, ordered, accepted, "cpu-parallel-vector",
                        scratch / "disposable-compute")
                    out["forty_update_seconds"] = round(
                        time.perf_counter() - compute_started, 3)
                    out["combined_stage_and_updates_seconds"] = round(
                        prep_seconds + out["forty_update_seconds"], 3)
                    out["checkpoint_save_seconds"] = result[
                        "total_checkpoint_save_seconds"]
                    out["gradient_seconds"] = result["total_grad_seconds"]
                    if reference_model is None:
                        reference_model = result["model"]
                    else:
                        import torch
                        largest = max(float(
                            (reference_model[k] - v).abs().max())
                            for k, v in result["model"].items())
                        if not math.isfinite(largest) or largest > 1e-5:
                            raise ValueError(
                                "Final trained model differs after staged replay")
                        out["max_model_parameter_delta"] = largest
                        out["optimizer_updates"] = result["optimizer_updates"]
                measurements.append(out)
                print(json.dumps(out, sort_keys=True), flush=True)
    def summary_for(label, field):
        values = [item[field] for item in measurements
                  if item["label"] == label and field in item]
        return {
            "median": round(statistics.median(values), 3),
            "trials": [round(x, 3) for x in values],
        } if values else None
    baseline = summary_for("reference", "stage_wall_seconds")
    fast2 = summary_for("fast-two", "stage_wall_seconds")
    total0 = summary_for("reference", "combined_stage_and_updates_seconds")
    total2 = summary_for("fast-two", "combined_stage_and_updates_seconds")
    result = {
        "schema": "rnad-i5-historical-g0-learner-stage-wall-v1",
        "original_source_sha": HISTORIC_SOURCE_SHA,
        "current_source_not_a_valid_learner_continuation": True,
        "accepted_complete_four_win_matches": MATCHES_PER_GENERATION,
        "one_optimizer_forty_updates_per_trial": True,
        "trials_per_mode": repeats,
        "stage_reference": baseline,
        "stage_fast_one": summary_for("fast-one", "stage_wall_seconds"),
        "stage_fast_two": fast2,
        "stage_fast_four": summary_for("fast-four", "stage_wall_seconds"),
        "combined_reference": total0,
        "combined_fast_two": total2,
        "stage_fast_two_below_60s_on_all_trials": all(
            x <= 60.0 for x in fast2["trials"]),
        "research_only": True,
        "checkpoint_published": False,
        "champion_promoted": False,
        "strength_established": False,
        "measurements": measurements,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", required=True)
    ap.add_argument("--actors", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = run(a.bootstrap, a.actors, repeats=a.repeats)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items()
                      if k!="measurements"}, indent=2),flush=True)


if __name__ == "__main__":
    main()
