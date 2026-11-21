"""Audit frozen-weight c005 guard A/B archive; never train or impute matches."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

OLD = "f08a9587d6a7033546eab00ffa6479399ca71c2f"
ENGINE = "ea81165993110ee9fb3a09dcb97c6708c0d7397b"


def verify_pairs(report, first_seed, key):
    if (report.get("pairs") != 32 or report.get("matches") != 64 or
        type(report.get("wins")) is not int or
        not isinstance(report.get("rows"), list) or
        len(report["rows"]) != 64 or
        report["wins"] != sum(1 for row in report["rows"]
                              if row.get("won") is True)):
        raise ValueError("Invalid 32-pair full Match outcome")
    expected = {(i, seat) for i in range(first_seed, first_seed + 32)
                for seat in (0, 1)}
    seen = set()
    for row in report["rows"]:
        pair = (row.get("seed"), row.get(key))
        score = row.get("score")
        won = row.get("won")
        if (pair not in expected or pair in seen or
            type(won) is not bool or
            not isinstance(score, list) or len(score) != 2 or
            any(type(x) is not int or not 0 <= x <= 4 for x in score) or
            max(score) != 4 or
            score[pair[1] if won else 1 - pair[1]] != 4 or
            type(row.get("rounds")) is not int or
            not 1 <= row["rounds"] <= 64):
            raise ValueError("Missing/duplicate/unfinished paired Match")
        seen.add(pair)
    if seen != expected:
        raise ValueError("Wrong exact seed/seat pairs")
    if report.get("seed", first_seed) != first_seed:
        raise ValueError("Mixed top-level seed")


def audit(models: Path, reports: Path):
    manifest = json.loads((models / "manifest.json").read_text())
    sha = hashlib.sha256((models / "c005.json").read_bytes()).hexdigest()
    if (manifest.get("schema") != "rnad-i5-queued-evaluation-models-v1" or
        manifest.get("source_sha") != OLD or
        manifest.get("engine_sha") != ENGINE or
        manifest.get("models", {}).get("c005") != sha or
        manifest.get("is_champion") is not False):
        raise ValueError("Untrusted original c005 model identity")
    files = {"direct": "c005-guarded-vs-c005-legacy.json",
             "basic_legacy": "c005-legacy-basic.json",
             "basic_guarded": "c005-guarded-basic.json"}
    d = {name: json.loads((reports / filename).read_text())
         for name, filename in files.items()}
    if len({x.get("source_sha") for x in d.values()}) != 1 or any(
        x.get("source_sha") == OLD or x.get("engine_sha") != ENGINE
        for x in d.values()):
        raise ValueError("Mixed or immutable engine/source signature")
    direct = d["direct"]
    if (direct.get("schema") != "rnad-direct-switch-guard-paired-v1" or
        direct.get("deployment_policy") != {
            "teacher_switch_guard": "none", "rnad_switch_guard": "anti-loop-v1"} or
        direct.get("rnad_model_sha256") != sha or
        direct.get("teacher_model_sha256") != sha or
        direct.get("is_champion") is not False or
        direct.get("inference_selection_mode") != "greedy"):
        raise ValueError("Wrong same-weight guard direct A/B")
    verify_pairs(direct, 1841001, "rnadSeat")
    for key, schema in (
        ("basic_legacy", "rnad-three-phase-vs-basic-paired-v1"),
        ("basic_guarded", "rnad-three-phase-switch-guard-vs-basic-paired-v1")
    ):
        x = d[key]
        if (x.get("schema") != schema or x.get("model_sha256") != sha or
            x.get("opponent_id") != "basic-all-phases" or
            x.get("inference_selection_mode") != "greedy" or
            x.get("champion_promoted") is not False or
            (key == "basic_guarded" and x.get("deployment_policy") != {
                "switch_guard": "anti-loop-v1"}) or
            (key == "basic_legacy" and "deployment_policy" in x)):
            raise ValueError("Wrong Basic guard-vs-baseline label")
        verify_pairs(x, 1842001, "candidateSeat")
    return {
        "schema": "rnad-i5-switch-guard-ablation-verified-v1",
        "source_sha": direct["source_sha"],
        "engine_sha": ENGINE,
        "frozen_model_sha256": sha,
        "guarded_vs_legacy_c005": {"wins": direct["wins"], "matches": 64},
        "basic_legacy": {"wins": d["basic_legacy"]["wins"], "matches": 64},
        "basic_guarded": {"wins": d["basic_guarded"]["wins"], "matches": 64},
        "direct_seed": 1841001, "basic_seed": 1842001,
        "old_source_model_hash_preserved": True,
        "statistically_established_improvement": False,
        "is_champion": False
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--out", required=True)
    opts = p.parse_args()
    result = audit(Path(opts.models), Path(opts.reports))
    dest = Path(opts.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result), flush=True)
