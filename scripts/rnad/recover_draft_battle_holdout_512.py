"""Strict deterministic salvage of teacher holdout after a documented invariant failure.

Only accept authority-complete four-win original-source public AQ Match traces
whose **unmodified** scorer verifies the accepted action maximizes declared
utility. A failing seed is recorded and excluded; never forge a legal result.
This is a filtered teacher-valid population, not the original unconditional
512-consecutive-seed study. All exclusions are reported before model evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

SOURCE = "022aa9cd5b4ac37514df1d845ea00c11c3c13a5c"
SCHEMA = "rnad-i5-verified-unseen-teacher-holdout-512-filter-v1"
START = 2522001
TARGET = 512
MAX_CANDIDATES = 1024
MAX_REJECTIONS = 8
ONLY_ACCEPTABLE_REJECTION = "Accepted public AQ teacher does not maximize its declared utility"
FEATURE = "rnad-three-phase-features-v5"
TEACHER = "rnad-aq-public-strategy-selection-v1"


def produce(directory):
    checkout = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              text=True, capture_output=True).stdout.strip()
    if checkout != SOURCE:
        raise ValueError("Frozen accepted teacher Git source changed")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    accepted, rejected, sha_by_seed = [], [], {}
    engine = json.loads(Path("provenance/manifest.json").read_text())["sourceCommit"]
    for seed in range(START, START + MAX_CANDIDATES):
        file = out / f"teacher-{seed}.json"
        call = ["node", "scripts/rnad/export-teacher.mjs",
                "--seed", str(seed), "--out", str(file),
                "--three-phase", "true",
                "--teacher", "aq-public-strategy-selection",
                "--features-v5", "true", "--utility-labels", "true",
                "--draft-review", "true"]
        proc = subprocess.run(call, capture_output=True, text=True, timeout=480)
        if proc.returncode != 0:
            if (ONLY_ACCEPTABLE_REJECTION not in proc.stderr or
                file.exists()):
                raise RuntimeError(
                    f"Unexpected or partial teacher export failure seed={seed}\n" +
                    proc.stderr[-3500:])
            rejected.append({"seed": seed,
                             "reason": "rejected_original_public_utility_invariant"})
            if len(rejected) > MAX_REJECTIONS:
                raise ValueError("Too many teacher-invariant failures; investigate original AQ policy")
            print(json.dumps({"excluded_invalid_teacher_seed": seed,
                              "rejections": len(rejected)}), flush=True)
            continue
        if not file.is_file():
            raise ValueError("Missing accepted teacher output")
        raw = file.read_bytes()
        d = json.loads(raw)
        if (d.get("seed") != seed or d.get("source_sha") != SOURCE or
            d.get("engine_sha") != engine or d.get("feature_schema") != FEATURE or
            d.get("teacher") != TEACHER or
            d.get("utility_labels") != "aq-public-phase-rank-v1" or
            d.get("completed") is not True or
            d.get("score", [])[d["winner"]] != 4 or
            not isinstance(d.get("examples"), list) or
            len(d["examples"]) != sum(d["action_counts"]) + 4*d["rounds"]):
            raise ValueError(f"Invalid original accepted four-win teacher Match seed={seed}")
        accepted.append(seed)
        sha_by_seed[str(seed)] = hashlib.sha256(raw).hexdigest()
        if len(accepted) % 64 == 0:
            print(json.dumps({"unseen_accepted_matches": len(accepted),
                              "excluded": len(rejected)}), flush=True)
        if len(accepted) == TARGET:
            break
    if len(accepted) != TARGET:
        raise ValueError("Not enough verified complete four-win teacher Matches")
    if len(rejected) == 0:
        print("All original proposed 512 seeds passed; conventional holdout is intact")
    index = {
        "schema": SCHEMA,
        "source_sha": SOURCE,
        "engine_sha": engine,
        "selection_mechanism": "first_512_successful_original_source_authority_match_utility_invariant",
        "initial_seed": START,
        "maximum_prespecified_candidate_seeds": MAX_CANDIDATES,
        "maximum_rejections": MAX_REJECTIONS,
        "unseen_complete_matches": TARGET,
        "accepted_seeds": accepted,
        "rejected_seeds": rejected,
        "trace_sha256_by_seed": sha_by_seed,
        "teacher_success_filter_applied_before_any_model_evaluation": True,
        "scorer_or_teacher_policy_altered": False,
        "unconditional_original_512_seed_population": len(rejected) == 0,
        "statistical_claim_limited_to_verified_teacher_valid_population": True,
        "league_entry_allowed": False,
        "champion_promoted": False,
    }
    (out / "accepted-manifest.json").write_text(json.dumps(index, indent=2)+"\n")
    print(json.dumps({"accepted_complete_matches":len(accepted),
          "excluded_invalid_teacher_matches":len(rejected),
          "last_seed":accepted[-1]}), flush=True)
    return index


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a=p.parse_args()
    produce(a.out)
