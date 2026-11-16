"""Bounded, independently versioned zero-sum opportunity reward R-NaD study.
Old actor bytes are immutable. 64 completions for standard models; c005 128.
The 128-model starts from its own 64-Match full-state checkpoint on this SHA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

from scripts.rnad.train import train, current_git_sha, frozen_engine_sha


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, model: str, initial: str, out: str, seed: int = 1770001):
    if model not in ("base128", "base256", "c005") or seed != 1770001:
        raise ValueError("Unexpected model/seed for pinned experiment")
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to overwrite previous research")
    root.mkdir(parents=True, exist_ok=True)
    input_path = Path(initial)
    initial_sha = sha256(input_path)
    studies = []
    snapshots = {}
    totals = {"accepted": 0, "voluntary": 0, "shaped": 0,
              "cost_by_seat": [0.0, 0.0]}
    segments = 8 if model == "c005" else 4
    for block in range(segments):
        result = train(seed=seed, matches=16,
                       out=str(root / "selfplay"), resume=block > 0,
                       initial_model=str(input_path),
                       three_phase=True, features_v2=True,
                       switch_guard="anti-loop-v1",
                       switch_cost="opportunity-v1",
                       eta=.2, magnet_period=4)
        expected = (block+1)*16
        if (result["completed_matches"] != expected or
            result["updates"] != expected or
            result["source_sha"] != current_git_sha() or
            result["engine_sha"] != frozen_engine_sha() or
            result.get("reward_objective") != "opportunity-v1" or
            result.get("switch_guard") != "anti-loop-v1" or
            result.get("model_id") !=
                "RNAD-003-3phase-v2-anti-loop-v1-opportunity-v1" or
            result.get("initialization_sha256") != initial_sha or
            result.get("is_champion") is not False or
            len(result["results"]) != 16 or
            [x["seed"] for x in result["results"]] !=
                [seed + block*16+i for i in range(16)]):
            raise ValueError("Incomplete/mixed Match learning segment")
        for x in result["results"]:
            if not (x["accepted_decisions"] > 0 and
                    0 <= x["shaped_switches"] <=
                    x["accepted_voluntary_switches"] and
                    all(math.isfinite(x[k]) for k in
                        ("loss", "policy_loss", "critic_loss", "behavior_ratio_max")) and
                    len(x["switch_cost_by_seat"]) == 2 and
                    all(0 <= cost <= .2000001 for cost in
                        x["switch_cost_by_seat"])):
                raise ValueError("Bad accepted/shaped RNAD training evidence")
            totals["accepted"] += x["accepted_decisions"]
            totals["voluntary"] += x["accepted_voluntary_switches"]
            totals["shaped"] += x["shaped_switches"]
            for seat in (0, 1):
                totals["cost_by_seat"][seat] += x["switch_cost_by_seat"][seat]
        latest = json.loads(
            (root/"selfplay"/"checkpoints"/"latest.json").read_text())
        metadata = latest["metadata"]
        checkpoint = root/"selfplay"/"checkpoints"/latest["checkpoint_file"]
        if not (metadata["source_sha"] == metadata["training_code_sha"]
                == current_git_sha() and
                metadata["engine_snapshot_sha"] == frozen_engine_sha() and
                metadata["model_id"] ==
                    "RNAD-003-3phase-v2-anti-loop-v1-opportunity-v1" and
                metadata["reward_objective"] == "opportunity-v1" and
                metadata["switch_guard"] == "anti-loop-v1" and
                metadata["initialization_sha256"] == initial_sha and
                metadata["initial_seed"] == seed and
                metadata["next_match_seed"] == seed + expected and
                metadata["completed_matches"] == metadata["updates"] == expected and
                len(set(metadata["completed_match_ids"])) == expected and
                metadata["completed_match_ids"] ==
                    [f"rnad-match-{seed+i}" for i in range(expected)] and
                metadata["accepted_decisions"] == totals["accepted"] and
                latest["checkpoint_sha256"] ==
                    result["last_match_checkpoint_sha"] and
                sha256(checkpoint) == latest["checkpoint_sha256"]):
            raise ValueError("Checkpoint provenance, RNG or SHA mismatch")
        studies.append({
            "segment": block+1, "completed_matches": expected,
            "model_sha256_state": result["model_sha256"],
            "checkpoint_sha256": latest["checkpoint_sha256"],
            "accepted_actions_so_far": totals["accepted"],
            "shaped_switches_so_far": totals["shaped"]})
        if expected in (64, 128):
            snapshot = root/f"model-u{expected}.json"
            shutil.copyfile(root/"selfplay"/"final-model.json", snapshot)
            if sha256(snapshot) == initial_sha:
                raise ValueError("Weights did not change during training")
            snapshots[str(expected)] = sha256(snapshot)
        # Avoid archiving all 64/128 binary checkpoints. Only the most
        # recent complete-state resume boundary is preserved in final output.
        for stale in (root/"selfplay"/"checkpoints").glob("checkpoint-u*.pt"):
            if stale != checkpoint:
                stale.unlink()
        print(json.dumps({"model": model, **studies[-1]}), flush=True)
    if totals["shaped"] == 0:
        # Not a successful 'learned turn cost' experiment.
        raise ValueError("No voluntary switch with an attack opportunity was shaped")
    output = {
        "schema": "rnad-switch-opportunity-research-v1",
        "source_sha": current_git_sha(),
        "engine_sha": frozen_engine_sha(),
        "initial_model": model, "original_model_sha256": initial_sha,
        "training_seed": seed, "complete_matches": segments*16,
        "optimizer_updates": segments*16,
        "guard": "anti-loop-v1", "reward_objective": "opportunity-v1",
        "new_optimizer_not_original_checkpoint_resume": True,
        "accepted_actions": totals["accepted"],
        "accepted_voluntary_switches": totals["voluntary"],
        "opportunity_shaped_switches": totals["shaped"],
        "opportunity_cost_by_seat": totals["cost_by_seat"],
        "model_snapshots": snapshots,
        "latest_checkpoint_sha256": studies[-1]["checkpoint_sha256"],
        "segments": studies,
        "statistical_strength_improvement_confirmed": False,
        "is_champion": False
    }
    (root/"study-report.json").write_text(
        json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps(output), flush=True)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("base128", "base256", "c005"),
                   required=True)
    p.add_argument("--initial", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(model=a.model, initial=a.initial, out=a.out)
