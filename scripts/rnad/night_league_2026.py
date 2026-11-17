"""Early, bounded, GitHub-hosted *research* league; NEVER a Champion or league gate.

Twenty independent, single-writer series. Every accepted complete four-win
Match is NeuRD/V-trace trained by the existing train.py. Each series has a
unique seed stream and immutable frozen opponents. Two time-budgeted waves
resume the exact optimizer/magnet/target/RNG checkpoint at Match boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

from scripts.rnad.neural import (
    RNADNetwork, migrate_v2_export_to_v3, export_json,
    THREE_PHASE_FEATURE_SCHEMA_V3,
    STATE_FEATURES_V3, ACTION_FEATURES_V3,
)
from scripts.rnad.train import train, current_git_sha, frozen_engine_sha

SCHEMA = "rnad-i5-early-night-league-exploratory-v1"
SOURCE_ARCHIVE = "f08a9587d6a7033546eab00ffa6479399ca71c2f"
OLD_NAMES = ("base128", "base256", "c005")
SHARDS = 20
STRIDE = 50000
FIRST_SEED = 2600001
MAX_MATCHES_PER_SERIES = 48000
MAX_LEAGUE_SECONDS_PER_WAVE = 17280  # 4h48m training, two waves ~9h36m.
ACCEPTED_SCHEMA = "rnad-v3-frozen-league-pool-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def strategy(shard):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("Unknown early league shard")
    mode = "league" if shard % 4 != 3 else "selfplay"
    return {
        "shard": shard,
        "mode": mode,
        "origin": OLD_NAMES[shard % len(OLD_NAMES)],
        "updates_per_match": 4 if shard % 2 == 0 else 1,
        "seed": FIRST_SEED + STRIDE * shard,
        "eta": .2, "magnet_period": 4,
    }


def verify_budget_and_seeds():
    ss = [strategy(k) for k in range(SHARDS)]
    ranges = [set(range(x["seed"], x["seed"] + MAX_MATCHES_PER_SERIES))
              for x in ss]
    if any(ranges[i] & ranges[j] for i in range(len(ranges))
           for j in range(i + 1, len(ranges))):
        raise ValueError("Overlapping league seed streams")
    if not any(x["mode"] == "selfplay" for x in ss):
        raise ValueError("Missing selfplay control lanes")
    if len({x["origin"] for x in ss}) != 3:
        raise ValueError("Missing frozen origin diversity")
    if {x["updates_per_match"] for x in ss} != {1, 4}:
        raise ValueError("Missing 1/4 update ablations")
    return ss


def prepare_models(model_dir, series, shard):
    """Verify inherited real models and deterministic frozen v3 opponent pool."""
    settings = strategy(shard)
    root = Path(model_dir)
    manifest = read(root / "manifest.json")
    if (manifest.get("schema") != "rnad-i5-queued-evaluation-models-v1"
            or manifest.get("source_sha") != SOURCE_ARCHIVE
            or manifest.get("engine_sha") != frozen_engine_sha()):
        raise ValueError("Untrusted historical model source manifest")
    for name in OLD_NAMES:
        p = root / (name + ".json")
        if sha(p) != manifest["models"][name]:
            raise ValueError("Frozen source model SHA mismatch: " + name)

    series = Path(series)
    models = series / "frozen"
    models.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, name in enumerate(OLD_NAMES):
        torch.manual_seed(10012026 + k)
        model = RNADNetwork(state_features=STATE_FEATURES_V3,
                            action_features=ACTION_FEATURES_V3)
        original_sha = migrate_v2_export_to_v3(model, root / (name + ".json"))
        if original_sha != manifest["models"][name]:
            raise ValueError("Failed SHA-pinned v2-to-v3 actor migration")
        output = models / (name + "-v3.json")
        export_json(model, output, feature_schema=THREE_PHASE_FEATURE_SCHEMA_V3)
        entries.append({"name": name, "path": str(output), "sha256": sha(output)})
    if len({x["sha256"] for x in entries}) != 3:
        raise ValueError("League opponents not distinct")

    pool = series / "league-pool.json"
    encoded = json.dumps({"schema": ACCEPTED_SCHEMA, "models": entries},
                         sort_keys=True, separators=(",", ":")) + "\n"
    if pool.exists() and pool.read_text() != encoded:
        raise ValueError("Frozen league pool changed between waves")
    pool.write_text(encoded)
    return root / (settings["origin"] + ".json"), pool, sha(pool)


def _safe_cleanup(series):
    """Keep only immutable most recent full-state checkpoint after each Match."""
    root = Path(series)
    p = root / "checkpoints" / "latest.json"
    if not p.is_file():
        return
    index = read(p)
    file = p.parent / index["checkpoint_file"]
    if sha(file) != index["checkpoint_sha256"]:
        raise ValueError("Last exact checkpoint corrupted before pruning")
    for other in p.parent.glob("checkpoint-u*.pt"):
        if other != file:
            other.unlink()
    # Validated historical Match hashes/identities remain in checkpoint metadata.
    rollouts = root / "rollouts"
    if rollouts.exists():
        for old in rollouts.glob("accepted-seed*.json"):
            old.unlink()


def _validate_checkpoint(series, settings, prev_matches, origin_sha, pool_sha):
    p = Path(series) / "checkpoints" / "latest.json"
    index = read(p)
    md = index["metadata"]
    if (md["source_sha"] != current_git_sha()
        or md["engine_snapshot_sha"] != frozen_engine_sha()
        or md["training_code_sha"] != current_git_sha()
        or md["initial_seed"] != settings["seed"]
        or md["initialization_sha256"] != origin_sha
        or md["completed_matches"] != prev_matches + 1
        or md["updates"] != (prev_matches + 1) * settings["updates_per_match"]
        or md["next_match_seed"] != settings["seed"] + prev_matches + 1
        or md["model_id"].find("-anti-loop-v1") == -1
        or md["observation_schema_version"] != THREE_PHASE_FEATURE_SCHEMA_V3
        or md.get("league_pool_sha256") !=
            (pool_sha if settings["mode"] == "league" else None)
        or len(md.get("completed_match_ids", [])) != prev_matches + 1
        or len(set(md["completed_match_ids"])) != prev_matches + 1):
        raise ValueError("Invalid full-state league checkpoint/authority boundary")
    if sha(p.parent / index["checkpoint_file"]) != index["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA invalid")
    return index


def run_one_wave(*, shard, wave, models, output, seconds):
    settings = strategy(shard)
    if wave not in (1, 2) or not 1 <= seconds <= MAX_LEAGUE_SECONDS_PER_WAVE:
        raise ValueError("Invalid bounded wave or time budget")
    if not 0 < FIRST_SEED + shard * STRIDE < 0xffffffff - MAX_MATCHES_PER_SERIES:
        raise ValueError("Seed out of 32bit bound")
    torch.set_num_threads(1)
    series = Path(output)
    series.mkdir(parents=True, exist_ok=True)
    old_model, league_pool, pool_sha = prepare_models(models, series, shard)
    origin_sha = sha(old_model)
    previous_index = series / "checkpoints" / "latest.json"
    if (wave == 1 and previous_index.exists()
            or wave == 2 and not previous_index.exists()):
        raise ValueError("Wave checkpoint absent or unexpectedly present")
    prev = read(previous_index)["metadata"]["completed_matches"] if wave == 2 else 0
    _safe_cleanup(series)
    # Persist source-bound selection, no external data or fake league gate.
    identity = {
        "schema": SCHEMA, **settings, "series_source_sha": current_git_sha(),
        "engine_sha": frozen_engine_sha(), "previous_v2_source": SOURCE_ARCHIVE,
        "initial_model_file_sha256": origin_sha,
        "frozen_pool_file_sha256": pool_sha,
        "league_eligibility_gate_passed": False,
        "explicit_early_user_authorization": True,
        "unresolved_independent_review_P1": True,
        "exploratory_only": True, "champion_promoted": False,
        "main_merged": False,
    }
    path = series / "series-identity.json"
    encoded = json.dumps(identity, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("Series identity changed while resuming")
    path.write_text(encoded)
    if wave == 2:
        index = read(previous_index)
        if (index["metadata"]["initialization_sha256"] != origin_sha
                or index["metadata"]["source_sha"] != current_git_sha()):
            raise ValueError("Untrusted reloaded wave-1 checkpoint")
    journal = series / "training-journal.ndjson"
    deadline = time.monotonic() + seconds
    count = 0
    start = time.monotonic()
    while (time.monotonic() < deadline and
           prev < MAX_MATCHES_PER_SERIES and count < MAX_MATCHES_PER_SERIES):
        result = train(
            seed=settings["seed"], matches=1, out=str(series),
            resume=prev > 0, initial_model=str(old_model),
            three_phase=True, features_v3=True,
            migrate_from_v2=True, switch_guard="anti-loop-v1",
            switch_cost="none", updates_per_match=settings["updates_per_match"],
            eta=settings["eta"], magnet_period=settings["magnet_period"],
            league_pool=str(league_pool)
                if settings["mode"] == "league" else None,
        )
        index = _validate_checkpoint(series, settings, prev, origin_sha, pool_sha)
        if result["completed_matches"] != prev + 1:
            raise ValueError("Learner completed count mismatch")
        row = {
            "wave": wave, "shard": shard,
            "match_count": prev + 1,
            "match_seed": settings["seed"] + prev,
            "match_id": index["metadata"]["last_match_id"],
            "checkpoint_sha256": index["checkpoint_sha256"],
            "model_sha256": result["model_sha256"],
            "magnet_sha256": result["magnet_sha256"],
            "accepted_decisions": index["metadata"]["accepted_decisions"],
            "optimizer_updates": index["metadata"]["updates"],
            "elapsed_seconds": time.monotonic() - start,
        }
        journal.open("a").write(json.dumps(row, allow_nan=False) + "\n")
        prev += 1
        count += 1
        _safe_cleanup(series)
        if count % 8 == 0:
            print(json.dumps({"research_league_progress": row}), flush=True)
    final_index = read(series / "checkpoints" / "latest.json")
    report = {
        "schema": SCHEMA, **identity,
        "wave": wave, "wave_completed_matches": count,
        "cumulative_completed_matches": prev,
        "wall_seconds_in_wave": time.monotonic() - start,
        "last_checkpoint_sha256": final_index["checkpoint_sha256"],
        "last_model_sha256": sha(series / "final-model.json"),
        "completed_match_boundaries": len(final_index["metadata"]["completed_match_ids"]),
        "training_decisions": final_index["metadata"]["accepted_decisions"],
        "optimizer_updates": final_index["metadata"]["updates"],
        "no_unverified_champion_or_league_gate_claim": True,
    }
    if count <= 0 or prev <= 0:
        raise ValueError("Time budget elapsed without even one complete trained Match")
    (series / "wave-summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"league_wave": wave, "shard": shard,
        "completed_matches": prev, "wave_matches": count,
        "source_sha": current_git_sha(),
        "research_only": True}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate-plan", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--wave", type=int, choices=(1, 2), default=1)
    p.add_argument("--models", default="model-input")
    p.add_argument("--out", default="artifacts/series")
    p.add_argument("--seconds", type=int, default=MAX_LEAGUE_SECONDS_PER_WAVE)
    args = p.parse_args()
    if args.validate_plan:
        print(json.dumps({"schema": SCHEMA,
             "shards": verify_budget_and_seeds(),
             "planned_parallel_runners": SHARDS,
             "two_wave_training_seconds_each": MAX_LEAGUE_SECONDS_PER_WAVE,
             "user_approved_pre_gate_league": True,
             "official_league_gate_passed": False}, indent=2))
    else:
        verify_budget_and_seeds()
        run_one_wave(shard=args.shard, wave=args.wave,
                     models=args.models, output=args.out,
                     seconds=args.seconds)


if __name__ == "__main__":
    main()
