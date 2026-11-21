"""I5 frozen-teacher scaling test: 256->512 draft, 256+256 battle warm start.

Run only on source commit 022aa9... and on accepted unmodified teacher
traces. All holdout Match seeds are independent of all training or dev seeds.
This is two separate research models, never an integrated league policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from scripts.rnad.train import current_git_sha, frozen_engine_sha, state_hash
from scripts.rnad.train_draft_deepsets import (
    DeepSetsDraft, FEATURE_WIDTH, load_events)
from scripts.rnad.train_draft_imitation_128 import (
    train_variant, independent_seed_top1)
from scripts.rnad.neural import (
    RNADNetwork, load_exported_json, THREE_PHASE_FEATURE_SCHEMA_V5,
    STATE_FEATURES_V5, ACTION_FEATURES_V5)
from scripts.rnad.train_teacher import collect

SOURCE = "022aa9cd5b4ac37514df1d845ea00c11c3c13a5c"
OLD_START, OLD_COUNT = 2471001, 256
OLD_USED_DEV_START, OLD_USED_DEV_COUNT = 2472001, 128
NEW_START, NEW_COUNT = 2520001, 256
HOLDOUT_START, HOLDOUT_COUNT = 2522001, 512
DRAFT_EPOCHS = 6
BATTLE_EPOCHS = 6
SCHEMA = "rnad-i5-draft-battle-learning-scale512-v1"
TEACHER = "rnad-aq-public-strategy-selection-v1"
PHASES = ("draft", "battle")


def verified_plan():
    ranges = ((OLD_START, OLD_COUNT), (OLD_USED_DEV_START, OLD_USED_DEV_COUNT),
              (NEW_START, NEW_COUNT), (HOLDOUT_START, HOLDOUT_COUNT))
    sets = [set(range(a, a + n)) for a, n in ranges]
    if any(sets[i] & sets[j] for i in range(len(sets))
           for j in range(i + 1, len(sets))):
        raise ValueError("Overlapping previous train, dev, new train or holdout seeds")
    if current_git_sha() != SOURCE:
        raise ValueError("Must use exact frozen 2026-11-17 source")
    return {
        "source_sha": SOURCE, "engine_sha": frozen_engine_sha(),
        "old_training": [OLD_START, OLD_COUNT],
        "old_used_development": [OLD_USED_DEV_START, OLD_USED_DEV_COUNT],
        "new_training": [NEW_START, NEW_COUNT],
        "unseen_heldout": [HOLDOUT_START, HOLDOUT_COUNT],
        "new_draft_training_complete_matches": OLD_COUNT + NEW_COUNT,
        "battle_training_sequence": "old256_model_weights+new256_accepted_Matches_new_optimizer",
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model_draft(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError("Draft actual model bytes are not verified by independent report")
    data = json.loads(Path(path).read_text())
    if (data.get("schema") != "rnad-draft-deepsets-weights-v1"
            or data.get("architecture") != "deepsets-soft"
            or data.get("feature_width") != FEATURE_WIDTH
            or data.get("source_sha") != SOURCE):
        raise ValueError("Draft model identity/source/schema mismatch")
    net = DeepSetsDraft()
    expected = net.state_dict()
    if set(data.get("weights", {})) != set(expected):
        raise ValueError("Missing registered draft model tensors")
    tensors = {}
    for key, value in expected.items():
        row = data["weights"][key]
        if (row.get("shape") != list(value.shape)
                or len(row.get("values", [])) != value.numel()
                or any(type(x) not in (int, float) or not math.isfinite(x)
                       for x in row["values"])):
            raise ValueError("Malformed draft tensor: " + key)
        tensors[key] = torch.tensor(
            row["values"], dtype=value.dtype).reshape(value.shape)
    net.load_state_dict(tensors, strict=True)
    return net.eval()


def train_draft(old_dir, fresh_dir, out):
    plan = verified_plan()
    torch.set_num_threads(1)
    old = load_events(old_dir, OLD_START, OLD_COUNT)
    additional = load_events(fresh_dir, NEW_START, NEW_COUNT)
    if ({e["seed"] for e in old} != set(range(OLD_START, OLD_START + OLD_COUNT))
        or {e["seed"] for e in additional}
           != set(range(NEW_START, NEW_START + NEW_COUNT))):
        raise ValueError("Wrong or missing accepted teacher training Match clusters")
    net, losses = train_variant(old + additional, "soft", epochs=DRAFT_EPOCHS)
    result_dir = Path(out)
    result_dir.mkdir(parents=True, exist_ok=False)
    model_path = result_dir / "draft512-e6.json"
    state = net.state_dict()
    payload = {
        "schema": "rnad-draft-deepsets-weights-v1",
        "architecture": "deepsets-soft",
        "feature_width": FEATURE_WIDTH,
        "source_sha": SOURCE,
        "training_details": {
            "experiment": SCHEMA, "objective": "soft",
            "teacher_completed_matches": OLD_COUNT + NEW_COUNT,
            "epochs": DRAFT_EPOCHS,
            "old_train_seed": OLD_START, "new_train_seed": NEW_START,
            "used_heldout_labels_for_model_selection": False,
            "league_training_started": False,
        },
        "weights": {
            k: {"shape": list(t.shape), "values": t.detach().flatten().tolist()}
            for k, t in state.items()
        },
    }
    model_path.write_bytes(json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
    report = {
        "schema": SCHEMA, "phase": "draft", **plan,
        "teacher": TEACHER,
        "model_file": model_path.name, "model_file_sha256": digest(model_path),
        "epochs": DRAFT_EPOCHS,
        "accepted_draft_decisions": len(old) + len(additional),
        "old_accepted_draft_decisions": len(old),
        "new_accepted_draft_decisions": len(additional),
        "train_epoch_losses": losses,
        "no_holdout_model_selection": True,
        "league_started": False, "champion_promoted": False,
        "strength_established": False,
    }
    (result_dir / "study.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": "draft", "train_matches": OLD_COUNT + NEW_COUNT,
        "sha256": report["model_file_sha256"]}), flush=True)


def restore_battle(model_file, report_file, source_label):
    report = json.loads(Path(report_file).read_text())
    if (report.get("source_sha") != SOURCE
        or report.get("engine_sha") != frozen_engine_sha()
        or report.get("teacher_id") != TEACHER
        or report.get("feature_schema") != THREE_PHASE_FEATURE_SCHEMA_V5
        or report.get("phase_balanced") is not True
        or report.get("phase_profile") != "battle-focus"
        or report.get("utility_distillation") is not True
        or report.get("utility_hard_label_mix") != 0.0
        or report.get("is_champion") is not False):
        raise ValueError("Nonfrozen or incompatible battle teacher/model: " + source_label)
    net = RNADNetwork(state_features=STATE_FEATURES_V5,
                      action_features=ACTION_FEATURES_V5)
    load_exported_json(net, model_file, expected_feature_schema=THREE_PHASE_FEATURE_SCHEMA_V5)
    if state_hash(net) != report["trained_model_sha256"]:
        raise ValueError("Battle model state SHA does not match frozen training report")
    return net.eval(), report


def paired_bootstrap(differences, count=10000):
    if len(differences) != HOLDOUT_COUNT or not all(
        isinstance(v, (int, float)) and math.isfinite(v) and -1 <= v <= 1
        for v in differences):
        raise ValueError("Missing independent bounded paired seed-cluster differences")
    rng = random.Random(5122026)
    sample_count = len(differences)
    boot = sorted(
        sum(differences[rng.randrange(sample_count)] for _ in range(sample_count))
            / sample_count for _ in range(count))
    return [boot[int(.025 * count)], boot[int(.975 * count) - 1]]


def phase_summary(rows, target, comparison):
    if len(rows) != HOLDOUT_COUNT or len({r["seed"] for r in rows}) != HOLDOUT_COUNT:
        raise ValueError("Incomplete heldout seed cluster count")
    if any(r["count"] < 1 or not 0 <= r["old"] <= r["count"]
           or not 0 <= r["new"] <= r["count"] for r in rows):
        raise ValueError("Invalid accepted teacher predictions per seed")
    old_weighted = sum(r["old"] for r in rows) / sum(r["count"] for r in rows)
    new_weighted = sum(r["new"] for r in rows) / sum(r["count"] for r in rows)
    values_old = [r["old"] / r["count"] for r in rows]
    values_new = [r["new"] / r["count"] for r in rows]
    cluster_mean = sum(values_new) / HOLDOUT_COUNT
    differences = [v - b for v, b in zip(values_new, values_old)]
    delta = sum(differences) / HOLDOUT_COUNT
    low, high = paired_bootstrap(differences)
    lower = max(0., cluster_mean -
                math.sqrt(math.log(3 / .20) / (2 * HOLDOUT_COUNT)))
    reached = new_weighted > target if comparison == "strictly_greater" else new_weighted >= target
    bound_reached = lower > target if comparison == "strictly_greater" else lower >= target
    return {
        "old_decision_weighted_top1": old_weighted,
        "new_decision_weighted_top1": new_weighted,
        "new_seed_cluster_mean_top1": cluster_mean,
        "new_familywise_80pct_one_sided_Hoeffding_lower": lower,
        "teacher_top1_target": target, "comparison": comparison,
        "individual_phase_imitations_target_reached": bool(reached and bound_reached),
        "independent_seed_clusters": HOLDOUT_COUNT,
        "accepted_teacher_decisions": sum(r["count"] for r in rows),
        "paired_delta_seed_cluster_mean": delta,
        "paired_delta_approximate_two_sided_95pct_cluster_bootstrap_CI": [low, high],
        "learning_improvement_supported_at_95pct": low > 0,
        "learning_regression_supported_at_95pct": high < 0,
        "seed_rows": rows,
    }


def evaluate(draft_old, draft_old_report, draft_new, draft_new_report,
             battle_old, battle_old_report, battle_new, battle_new_report,
             holdout_dir, out, qualified_manifest):
    plan = verified_plan()
    qualified = json.loads(Path(qualified_manifest).read_text())
    if qualified.get('schema')!='rnad-i5-utility-qualified-teacher-512-v1' or qualified.get('source_sha')!=SOURCE or qualified.get('rule')!='strict_utility_audit_pass_before_any_model_comparison':
        raise ValueError('Untrusted qualified teacher holdout manifest')
    seeds = qualified.get('accepted_seeds')
    excluded = qualified.get('excluded_seeds')
    if (not isinstance(seeds,list) or len(seeds)!=HOLDOUT_COUNT or
        len(set(seeds))!=HOLDOUT_COUNT or
        not isinstance(excluded,list) or
        any(type(seed)!=int or not HOLDOUT_START<=seed<HOLDOUT_START+2000 for seed in seeds) or
        not isinstance(qualified.get('qualification_sha256'),str)):
        raise ValueError('Incomplete or duplicate qualified heldout seeds')
    if (seeds!=sorted(seeds) or any(type(row)!=dict or
            row.get('reason')!='strict_public_teacher_utility_invariant_failed'
            for row in excluded)):
        raise ValueError('Untrusted utility exclusion record')
    if set(seeds) & {row.get('seed') for row in excluded}:
        raise ValueError('Excluded teacher utility mismatch reused in holdout')
    canon=json.dumps({'accepted_seeds':seeds,'excluded_seeds':excluded},
        sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(canon).hexdigest()!=qualified['qualification_sha256']:
        raise ValueError('Qualified public teacher seed manifest digest mismatch')
    if (qualified.get('independent_complete_match_clusters')!=HOLDOUT_COUNT or
        qualified.get('filter_not_based_on_model_outputs') is not True or
        qualified.get('qualifier_is_strict_no_invariant_weakening') is not True):
        raise ValueError('Qualified teacher population not declared correctly')
    # Selection/filtering is entirely determined by the public teacher utility
    # invariant, before reading frozen old/new students. Never use student scores.

    torch.set_num_threads(1)
    dr_old_report = json.loads(Path(draft_old_report).read_text())
    dr_new_report = json.loads(Path(draft_new_report).read_text())
    if (dr_old_report.get("source_sha") != SOURCE
            or dr_old_report.get("engine_sha") != frozen_engine_sha()
            or dr_old_report.get("teacher_id") != TEACHER
            or dr_old_report.get("train_matches") != OLD_COUNT
            or dr_old_report.get("heldout_used_for_hyperparameter_selection") is not True
            or dr_new_report.get("schema") != SCHEMA
            or dr_new_report.get("new_draft_training_complete_matches") != OLD_COUNT + NEW_COUNT
            or dr_new_report.get("new_training") != [NEW_START, NEW_COUNT]):
        raise ValueError("Frozen draft and new draft provenance do not agree")
    old_draft_sha = dr_old_report["models"]["soft256-e6"]["sha256"]
    old_draft = load_model_draft(draft_old, old_draft_sha)
    new_draft = load_model_draft(draft_new, dr_new_report["model_file_sha256"])
    old_battle, br_old_report = restore_battle(battle_old, battle_old_report, "old")
    new_battle, br_new_report = restore_battle(battle_new, battle_new_report, "new")
    if (br_old_report.get("teacher_completed_matches") != OLD_COUNT
        or br_old_report.get("teacher_updates") != OLD_COUNT * BATTLE_EPOCHS
        or br_old_report.get("warmstart_source_model_file_sha256") is not None
        or br_new_report.get("teacher_completed_matches") != NEW_COUNT
        or br_new_report.get("teacher_updates") != NEW_COUNT * BATTLE_EPOCHS
        or br_new_report.get("warmstart_source_model_file_sha256") != digest(battle_old)
        or br_new_report.get("warmstart_optimizer_reused") is not False):
        raise ValueError("Battle weight-only continuation provenance mismatch")
    rows_draft, rows_battle = [], []
    with torch.no_grad():
        for group in range(0, HOLDOUT_COUNT, 64):
            group_seeds = seeds[group:group+64]
            batch_draft = [event for seed in group_seeds
                           for event in load_events(holdout_dir,seed,1)]
            draft_counts = {}
            for e in batch_draft:
                seed = e["seed"]
                row = draft_counts.setdefault(seed, {"seed": seed, "count": 0, "old": 0, "new": 0})
                predicted_old = int(old_draft(e["pool"], e["codes"], e["global"]).argmax())
                predicted_new = int(new_draft(e["pool"], e["codes"], e["global"]).argmax())
                row["count"] += 1
                row["old"] += int(predicted_old == e["label"])
                row["new"] += int(predicted_new == e["label"])
            records = [trace for seed in group_seeds
                       for trace in collect(seed,1,Path(out),three_phase=True,
                           teacher="aq-public-strategy-selection",
                           features_v5=True,utility_labels=True,
                           reuse_from=holdout_dir)]
            battle_counts = {}
            for record in records:
                seed = record["seed"]
                memory_old = [old_battle.initial_memory() for _ in range(2)]
                memory_new = [new_battle.initial_memory() for _ in range(2)]
                row = {"seed": seed, "count": 0, "old": 0, "new": 0}
                for event in record["examples"]:
                    seat = event["actor"]
                    if seat not in (0, 1):
                        raise ValueError("Invalid teacher actor seat")
                    features = torch.tensor([event["state"]], dtype=torch.float32)
                    actions = torch.tensor([event["actions"]], dtype=torch.float32)
                    a, _, h_old = old_battle(features, actions, memory_old[seat])
                    b, _, h_new = new_battle(features, actions, memory_new[seat])
                    memory_old[seat] = h_old.detach()
                    memory_new[seat] = h_new.detach()
                    if event["phase"] == "battle":
                        row["count"] += 1
                        row["old"] += int(int(a[0].argmax()) == event["choice"])
                        row["new"] += int(int(b[0].argmax()) == event["choice"])
                battle_counts[seed] = row
            expected = set(group_seeds)
            if set(draft_counts) != expected or set(battle_counts) != expected:
                raise ValueError("Missing fresh four-win Match seeds in holdout batch")
            rows_draft.extend(draft_counts[k] for k in sorted(expected))
            rows_battle.extend(battle_counts[k] for k in sorted(expected))
            print(json.dumps({"fresh_heldout_matches_evaluated": len(rows_draft)}), flush=True)
    expected = set(seeds)
    if ({r["seed"] for r in rows_draft} != expected or
        {r["seed"] for r in rows_battle} != expected):
        raise ValueError("Incomplete or misaligned independent draft/battle holdout")
    study = {
        "schema": SCHEMA, "kind": "independent_teacher_imitations_only",
        **plan,
        "holdout_teacher": TEACHER, "evaluation_models_fixed_before_holdout": True,
        "holdout_population": "strict_teacher_utility_consistent_complete_Matches_only",
        "qualified_seed_manifest": qualified,
        "qualified_seed_manifest_sha256": digest(qualified_manifest),
        "teacher_invariant_exclusion_policy": "only exact public AQ teacher utility mismatch; every exclusion listed before model scoring",
        "incompatible_with_unfiltered_teacher_population": True,
        "holdout_seeds": seeds,
        "excluded_seed_count": len(excluded),
        "draft": phase_summary(rows_draft, .50, "strictly_greater"),
        "battle": phase_summary(rows_battle, .80, "at_least"),
        "model_file_sha256": {
            "draft_old": old_draft_sha, "draft_new": digest(draft_new),
            "battle_old": digest(battle_old), "battle_new": digest(battle_new),
        },
        "battle_optimizer_restored": False,
        "draft_and_battle_are_separate_models": True,
        "same_three_phase_integrated_policy_evaluated": False,
        "selection_evaluated_in_this_study": False,
        "league_started": False, "league_entry_allowed": False,
        "champion_promoted": False, "strength_established": False,
    }
    result_dir = Path(out)
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "study.json").write_text(json.dumps(
        study, indent=2, allow_nan=False) + "\n")
    lines = ["# I5 Draft / Battle scaling on 512 new *utility-qualified* complete Matches",
             "", "Teacher-visitation Top1, not real Match win rate.",
             "Each seed is one independent complete four-win teacher Match.",
             "Same teacher/source and heldout seeds; different frozen students.",
             f"Utility-inconsistent seeds excluded **before model scoring**: {len(excluded)}.",
             "This is a conditional teacher-utility-qualified population; not all random complete Matches.",
             "",
             "| Phase | 256 model Top1 | Extra-training Top1 | Paired seed-cluster delta | Approx 95% CI | Teacher target lower |",
             "|---|---:|---:|---:|---:|---:|"]
    for phase in PHASES:
        p = study[phase]
        low, high = p["paired_delta_approximate_two_sided_95pct_cluster_bootstrap_CI"]
        lines.append(
            f"| {phase} | {p['old_decision_weighted_top1']:.2%} | "
            f"{p['new_decision_weighted_top1']:.2%} | "
            f"{p['paired_delta_seed_cluster_mean']:+.2%} | "
            f"[{low:+.2%}, {high:+.2%}] | "
            f"{p['new_familywise_80pct_one_sided_Hoeffding_lower']:.2%} |")
    lines += ["", "Draft=256 vs 512 from scratch, same 6-epoch soft objective.",
        "Battle=256 baseline vs initial weights+new256 6-epoch battle-focus (new optimizer).",
        "No league, no AQ strength, no Champion. No same integrated three-phase policy."]
    (result_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return study


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("train-draft")
    for name in ("old-dir", "new-dir", "out"):
        x.add_argument("--" + name, required=True)
    x = sub.add_parser("evaluate")
    for name in ("draft-old", "draft-old-report", "draft-new", "draft-new-report",
                 "battle-old", "battle-old-report", "battle-new",
                 "battle-new-report", "holdout-dir", "out", "qualified-manifest"):
        x.add_argument("--" + name, required=True)
    a = p.parse_args()
    if a.command == "train-draft":
        train_draft(a.old_dir, a.new_dir, a.out)
    else:
        evaluate(a.draft_old, a.draft_old_report, a.draft_new, a.draft_new_report,
                 a.battle_old, a.battle_old_report, a.battle_new,
                 a.battle_new_report, a.holdout_dir, a.out,
                 a.qualified_manifest)


if __name__ == "__main__":
    main()
