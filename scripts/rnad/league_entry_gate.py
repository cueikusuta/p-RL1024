"""Strict, phase-complete, source-bound imitation gate before starting a LEAGUE.

This is NOT a Champion win-rate gate.  One complete held-out teacher Match
(seed) is the independent cluster; its individual actions are dependent.
No evidence or too little evidence must result in HOLD, not an automatic start.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

SCHEMA = "rnad-i5-three-phase-accepted-teacher-heldout-v1"
CONFIG_SCHEMA = "rnad-i5-league-entry-thresholds-v1"
TARGETS = {"draft": (0.50, "strictly_greater"),
           "selection": (0.80, "at_least"),
           "battle": (0.80, "at_least")}
HEXPATTERN = re.compile(r"^[0-9a-f]+$")


def sha(value, length):
    return isinstance(value, str) and len(value) == length and bool(HEXPATTERN.fullmatch(value))


def verify_config(config):
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Wrong league entry gate contract")
    fixed = (config.get("confidence_level") == .80 and
             config.get("familywise_one_sided_alpha") == .20 and
             config.get("min_independent_seed_clusters_per_phase") == 128 and
             config.get("min_accepted_teacher_decisions_per_phase") == 256 and
             config.get("independent_unit") == "heldout_teacher_complete_match_seed" and
             config.get("require_same_integrated_policy_id") is True and
             config.get("require_distinct_training_and_holdout_seeds") is True and
             config.get("require_complete_authority_accepted_heldout") is True and
             config.get("league_training_started") is False and
             config.get("league_champion_promoted") is False and
             set(config.get("phase_top1_targets", {})) == set(TARGETS) and
             all(config["phase_top1_targets"][phase] ==
                 {"threshold": threshold, "comparison": operator}
                 for phase, (threshold, operator) in TARGETS.items()))
    if not fixed:
        raise ValueError("Thresholds or false-by-default safety contract changed")
    return config


def gate_report(config, evidence):
    verify_config(config)
    if evidence is None:
        return {"schema": "rnad-i5-league-entry-gate-result-v1",
                "league_training_allowed": False,
                "reason": "missing_heldout_evidence", "phases": {}}
    if (not isinstance(evidence, dict) or evidence.get("schema") != SCHEMA or
        not isinstance(evidence.get("integrated_policy_id"), str) or
        not evidence["integrated_policy_id"].strip() or
        not sha(evidence.get("source_sha"), 40) or
        not sha(evidence.get("engine_sha"), 64) or
        evidence.get("run_conclusion") != "success" or
        evidence.get("authority_match_legality_verified") is not True or
        evidence.get("actor_private_info_noninterference_verified") is not True or
        not isinstance(evidence.get("phases"), dict) or
        set(evidence["phases"]) != set(TARGETS)):
        raise ValueError("Unverified or incomplete integrated phase evaluation")
    out = {"schema": "rnad-i5-league-entry-gate-result-v1",
           "integrated_policy_id": evidence["integrated_policy_id"],
           "league_training_allowed": False, "phases": {},
           "ci_method": "one-sided independent seed-cluster Hoeffding",
           "confidence": "familywise at least 80%, Bonferroni across three phases",
           "strength_established": False, "champion_promoted": False}
    alpha = config["familywise_one_sided_alpha"]
    complete = True
    for phase, (threshold, comparison) in TARGETS.items():
        report = evidence["phases"][phase]
        if (not isinstance(report, dict) or
            not sha(report.get("model_sha256"), 64) or
            not sha(report.get("heldout_trace_sha256"), 64) or
            not isinstance(report.get("teacher_id"), str) or
            not report["teacher_id"].strip() or
            not isinstance(report.get("training_seeds"), list) or
            not isinstance(report.get("seed_clusters"), list)):
            raise ValueError("Missing trustworthy phase/model evidence: " + phase)
        training = report["training_seeds"]
        seeds = report["seed_clusters"]
        if (any(type(s) is not int or s < 1 or s > 0xffffffff for s in training)
            or len(training) != len(set(training))):
            raise ValueError("Invalid training seeds: "+phase)
        if len(seeds) < 128:
            out["phases"][phase] = {"passed": False, "reason": "too_few_independent_holdout_seeds",
                                    "independent_seed_clusters": len(seeds)}
            complete = False
            continue
        found = set()
        teacher_decisions = 0
        teacher_correct = 0
        means = []
        for row in seeds:
            if (not isinstance(row, dict) or
                type(row.get("seed")) is not int or
                not 1 <= row["seed"] <= 0xffffffff or
                type(row.get("accepted_decisions")) is not int or
                type(row.get("top1_correct")) is not int or
                not 1 <= row["accepted_decisions"] or
                not 0 <= row["top1_correct"] <= row["accepted_decisions"] or
                row.get("completed_four_win_match") is not True or
                row.get("teacher_action_accepted") is not True):
                raise ValueError("Invalid or incomplete accepted teacher Match: "+phase)
            if row["seed"] in found:
                raise ValueError("Duplicate heldout seed: "+phase)
            found.add(row["seed"])
            teacher_decisions += row["accepted_decisions"]
            teacher_correct += row["top1_correct"]
            means.append(row["top1_correct"]/row["accepted_decisions"])
        if found.intersection(training):
            raise ValueError("Training/heldout seed reuse: "+phase)
        if teacher_decisions < 256:
            out["phases"][phase] = {"passed": False, "reason": "too_few_accepted_teacher_decisions",
                                    "independent_seed_clusters": len(seeds),
                                    "accepted_decisions": teacher_decisions}
            complete = False
            continue
        point = teacher_correct/teacher_decisions
        # Each complete teacher Match is one independent bounded observation.
        # One-sided Hoeffding union bound controls false positive over 3 phases.
        cluster_mean = sum(means)/len(means)
        radius = math.sqrt(math.log(3/alpha)/(2*len(means)))
        lower = max(0., cluster_mean - radius)
        def reaches(x):
            return x > threshold if comparison == "strictly_greater" else x >= threshold
        passed = reaches(point) and reaches(lower)
        out["phases"][phase] = {
            "passed": passed, "independent_seed_clusters": len(seeds),
            "accepted_decisions": teacher_decisions,
            "decision_weighted_top1": point, "seed_cluster_mean_top1": cluster_mean,
            "familywise_80pct_one_sided_lower_bound": lower,
            "target": threshold, "comparison": comparison,
        }
        complete &= passed
    out["league_training_allowed"] = bool(complete)
    out["reason"] = ("all_three_imitation_thresholds_statistically_reached"
                     if complete else "one_or_more_phase_gates_unmet")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="docs/research/rnad/LEAGUE_ENTRY_GATE_2026-11-17.json")
    parser.add_argument("--evidence")
    parser.add_argument("--out")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    e = json.loads(Path(args.evidence).read_text()) if args.evidence else None
    result = gate_report(c, e)
    content = json.dumps(result, indent=2, allow_nan=False, sort_keys=True)+"\n"
    if args.out:
        Path(args.out).write_text(content)
    print(content, end="")
    if args.require_pass and not result["league_training_allowed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
