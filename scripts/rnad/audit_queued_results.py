"""Audit original 36-job hosted queue without masking incomplete direct matches.

Only source-bound, two-seat, four-win full Match reports contribute to scores.
Missing/sharded failures are reported as incomplete, never imputed as wins/losses.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

SOURCE_SHA = "f08a9587d6a7033546eab00ffa6479399ca71c2f"
ENGINE_SHA = "ea81165993110ee9fb3a09dcb97c6708c0d7397b"
MODELS = ("base128", "base256", "c019", "c005")
TEACHERS = {
    "equal-0": ("equal", 0),
    "equal-025": ("equal", .025),
    "focus-0": ("draft-selection", 0),
    "focus-025": ("draft-selection", .025),
}


def read_unique(folder: Path, name: str, *, required: bool = True):
    paths = sorted(folder.rglob(name)) if folder.exists() else []
    if len(paths) != 1:
        if len(paths) == 0 and not required:
            return None
        raise ValueError(f"Missing or duplicate research file: {name} ({len(paths)})")
    return json.loads(paths[0].read_text())


def check_match_rows(report, *, seed, pairs, seat_key):
    if (report.get("pairs") != pairs or report.get("matches") != 2 * pairs
        or type(report.get("wins")) is not int
        or not isinstance(report.get("rows"), list)
        or len(report["rows"]) != 2 * pairs
        or report.get("wins") != sum(int(x.get("won") is True)
            for x in report["rows"])
        or report.get("win_rate") != report["wins"] / (2 * pairs)):
        raise ValueError("Invalid full-Match score totals")
    expected = {(s, seat) for s in range(seed, seed + pairs) for seat in (0, 1)}
    rows = set()
    for row in report["rows"]:
        seat = row.get(seat_key)
        key = (row.get("seed"), seat)
        score = row.get("score")
        won = row.get("won")
        rounds = row.get("rounds")
        if (key not in expected or key in rows or type(won) is not bool
            or type(rounds) is not int or not 4 <= rounds <= 64
            or not isinstance(score, list) or len(score) != 2
            or any(type(n) is not int or not 0 <= n <= 4 for n in score)
            or max(score) != 4
            or score[seat if won else 1 - seat] != 4
            or score[1 - seat if won else seat] >= 4):
            raise ValueError("Invalid, duplicate or unfinished seed/seat Match")
        rows.add(key)
    if rows != expected:
        raise ValueError("Not every seed has both seat assignments")


def audit(folder: Path):
    models = read_unique(folder / "models", "manifest.json")
    if (models.get("schema") != "rnad-i5-queued-evaluation-models-v1"
        or models.get("source_sha") != SOURCE_SHA
        or models.get("engine_sha") != ENGINE_SHA
        or models.get("is_champion") is not False
        or set(models.get("models", {})) != set(MODELS)
        or any(not isinstance(h, str) or len(h) != 64
            for h in models["models"].values())):
        raise ValueError("Wrong immutable model manifest")
    out = {
        "schema": "rnad-i5-queue-audited-results-v1",
        "source_sha": SOURCE_SHA, "engine_sha": ENGINE_SHA,
        "origin_queue_run": 43757054212,
        "teacher_queue_run": 43756892480,
        "model_sha256": models["models"],
        "basic": {}, "direct": {}, "teacher": {},
        "all_required_direct_evaluations_complete": False,
        "statistical_improvement_confirmed": False,
        "is_champion": False
    }
    for model in MODELS:
        wins = 0
        for block in range(4):
            rep = read_unique(folder / "basic", f"basic-{model}-{block}.json")
            if (rep.get("schema") != "rnad-three-phase-vs-basic-paired-v1"
                or rep.get("source_sha") != SOURCE_SHA
                or rep.get("engine_sha") != ENGINE_SHA
                or rep.get("model_sha256") != models["models"][model]
                or rep.get("opponent_id") != "basic-all-phases"
                or rep.get("inference_selection_mode") != "greedy"
                or rep.get("champion_promoted") is not False):
                raise ValueError("Wrong frozen Basic evaluation provenance")
            check_match_rows(rep, seed=1710001 + block * 64,
                             pairs=64, seat_key="candidateSeat")
            wins += rep["wins"]
        out["basic"][model] = {
            "wins": wins, "matches": 512, "win_rate": wins / 512,
            "completed_blocks": [0, 1, 2, 3]
        }
    complete = True
    for candidate in ("c019", "c005"):
        for baseline in ("base128", "base256"):
            wins, blocks, missing = 0, [], []
            for block in range(4):
                name = f"direct-{candidate}-{baseline}-{block}.json"
                rep = read_unique(folder / "direct", name, required=False)
                if rep is None:
                    missing.append(block)
                    complete = False
                    continue
                if (rep.get("schema") != "rnad-direct-neural-paired-match-v1"
                    or rep.get("source_sha") != SOURCE_SHA
                    or rep.get("engine_sha") != ENGINE_SHA
                    or rep.get("rnad_model_sha256") != models["models"][candidate]
                    or rep.get("teacher_model_sha256") != models["models"][baseline]
                    or rep.get("model_feature_schema") != "rnad-three-phase-features-v2"
                    or rep.get("inference_selection_mode") != "greedy"
                    or rep.get("is_champion") is not False
                    or rep.get("seed") != 1720001 + block * 64):
                    raise ValueError("Wrong frozen direct Match provenance")
                check_match_rows(rep, seed=1720001 + block * 64,
                                 pairs=64, seat_key="rnadSeat")
                wins += rep["wins"]
                blocks.append(block)
            matches = 128 * len(blocks)
            out["direct"][f"{candidate}-vs-{baseline}"] = {
                "wins_from_complete_blocks": wins,
                "complete_matches": matches,
                "win_rate_for_complete_blocks_only": (wins / matches
                                                       if matches else None),
                "complete_blocks": blocks,
                "missing_or_aborted_blocks": missing,
                "valid_full_512_match_comparison": not missing
            }
    for ident, (profile, smoothing) in TEACHERS.items():
        base = folder / "teacher"
        prefix = f"rnad-i5-teacher-{ident}-43756892480"
        archive = base / prefix
        rep = read_unique(archive, "teacher-report.json")
        held = read_unique(archive, "heldout-imitation.json")
        basic = read_unique(archive, f"{ident}.json")
        if (rep.get("schema") != "rnad-three-phase-imitation-report-v1"
            or rep.get("feature_schema") != "rnad-three-phase-features-v2"
            or rep.get("phase_profile") != profile
            or rep.get("label_smoothing") != smoothing
            or rep.get("seed") != 1730001
            or rep.get("teacher_completed_matches") != 24
            or rep.get("phase_balanced") is not True
            or rep.get("is_champion") is not False
            or held.get("seed") != 1730301
            or held.get("matches") != 8
            or held.get("source_sha") != rep.get("source_sha")
            or basic.get("source_sha") != rep.get("source_sha")
            or basic.get("engine_sha") != rep.get("engine_sha")
            or held.get("engine_sha") != rep.get("engine_sha")
            or basic.get("model_sha256") != held.get("student_model_sha256")
            or basic.get("inference_selection_mode") != "greedy"
            or basic.get("opponent_id") != "basic-all-phases"
            or basic.get("champion_promoted") is not False):
            raise ValueError("Incomparable teacher phase-loss research")
        check_match_rows(basic, seed=1731001, pairs=32,
                         seat_key="candidateSeat")
        out["teacher"][ident] = {
            "basic_wins": basic["wins"], "basic_matches": 64,
            "basic_win_rate": basic["wins"] / 64,
            "heldout_phase_top1": {
                phase: held["phases"][phase]["top1_accuracy"]
                for phase in ("draft", "selection", "battle")
            },
        }
    out["all_required_direct_evaluations_complete"] = complete
    out["reason_if_incomplete"] = (None if complete else
        "A missing full-Match seed shard cannot be counted as a win or loss; "
        "the original 3,000-decision safety limit is preserved.")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(Path(args.root))
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "basic": report["basic"],
        "direct": report["direct"],
        "teacher": report["teacher"],
        "complete": report["all_required_direct_evaluations_complete"],
        "is_champion": False
    }), flush=True)


if __name__ == "__main__":
    main()
