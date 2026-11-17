"""I5 follow-up research: 1) full-input aliases, 2) teacher gap/lead/trio,
3) controlled training-Match count x hidden width study.

Reuses sealed accepted AQ traces from run 44032488678. The 128-Match heldout
has been used for model selection already: diagnostic ONLY, never league gate.
No teacher utility is supplied to the student at inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

from scripts.rnad.train_selection_specialist import (
    SOURCE_SHA, TEACHER, MATCHES_HOLDOUT, SEED_HOLDOUT,
    load_traces, selection_events, train, evaluate, assess,
    TRIO_IDS, LEAD_IDS, feature_sufficiency_audit,
)

TRAIN_MATCHES = (128, 256)
WIDTHS = (64, 96, 128)
GAP_BOUNDS = (0.01, 0.05, 0.10)
GAP_LABELS = ("gap<=1%", "1%<gap<=5%", "5%<gap<=10%", "gap>10%")


def gap_bucket(gap):
    if not math.isfinite(gap) or gap < 0 or gap > 1.000001:
        raise ValueError("Invalid relative teacher top2 gap")
    for i, edge in enumerate(GAP_BOUNDS):
        if gap <= edge:
            return GAP_LABELS[i]
    return GAP_LABELS[-1]


def gap_diagnostics(net, traces):
    """Decision metrics by teacher normalized top2 gap, trio, conditional lead.

    Report independent seed counts for each bin. Small buckets are descriptive,
    not independent Bernoulli samples, and never infer statistical superiority.
    """
    per_bin = {label: {"decisions": 0, "top1": 0, "trio": 0,
                       "lead_given_trio": 0, "regret_sum": 0.,
                       "seeds": set()}
               for label in GAP_LABELS}
    net.eval()
    with torch.no_grad():
        for seed, event in selection_events(traces):
            utility = event["candidate_utilities"]
            ranked = sorted(utility, reverse=True)
            span = ranked[0] - ranked[-1]
            gap = (ranked[0] - ranked[1]) / span if span > 1e-12 else 0.
            label = gap_bucket(gap)
            state = torch.tensor([event["state"]], dtype=torch.float32)
            actions = torch.tensor([event["actions"]], dtype=torch.float32)
            logits = net(state, actions)[0]
            picked = int(logits.argmax())
            teacher = event["choice"]
            trio = int(TRIO_IDS[picked] == TRIO_IDS[teacher])
            lead_given = int(trio and LEAD_IDS[picked] == LEAD_IDS[teacher])
            score = assess(logits, event)
            row = per_bin[label]
            row["decisions"] += 1
            row["top1"] += score["top1"]
            row["trio"] += trio
            row["lead_given_trio"] += lead_given
            row["regret_sum"] += score["regret"]
            row["seeds"].add(seed)
    result = {}
    for label, row in per_bin.items():
        n = row["decisions"]
        result[label] = {
            "decisions": n,
            "independent_match_seed_clusters": len(row["seeds"]),
            "top1": row["top1"] / n if n else None,
            "unordered_trio_accuracy": row["trio"] / n if n else None,
            "lead_accuracy_conditional_on_correct_trio":
                row["lead_given_trio"] / row["trio"] if row["trio"] else None,
            "mean_normalized_utility_regret": row["regret_sum"] / n if n else None,
            "exploratory_only": True,
        }
    return result


def compare_earlier_96_model(net, path):
    """The 256x96 arm must reproduce archived structured-hard weights.

    A later code change may intentionally alter the architecture; if so,
    fail and require a new explicitly named experiment rather than pretending
    to compare the same fixed training process.
    """
    if not path:
        raise ValueError("Archived original structured-hard model is required")
    original = json.loads(Path(path).read_bytes())
    if original.get("arm") != "structured-hard" or original.get("structured") is not True:
        raise ValueError("Wrong archived reference arm")
    observed = net.state_dict()
    if set(original.get("weights", {})) != set(observed):
        raise ValueError("Original selection architecture differs")
    for k, tensor in observed.items():
        ref = original["weights"][k]
        if ref["shape"] != list(tensor.shape) or ref["values"] != tensor.detach().flatten().tolist():
            raise ValueError("Reproduced 256x96 structured-hard model differs: " + k)


def run(train_dir, heldout_dir, previous_model, out):
    torch.set_num_threads(1)
    train_traces, engine = load_traces(train_dir, 2471001, 256)
    held, held_engine = load_traces(heldout_dir, SEED_HOLDOUT, MATCHES_HOLDOUT)
    if engine != held_engine:
        raise ValueError("Unmatched teacher engine")
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=False)
    reference = {
        "schema": "rnad-i5-selection-followup-exploratory-v1",
        "teacher": TEACHER, "teacher_trace_source_sha": SOURCE_SHA,
        "engine_sha": engine, "teacher_match_source_run": 44032488678,
        "training_match_counts": list(TRAIN_MATCHES),
        "widths": list(WIDTHS), "epochs": 6,
        "seed": 57720, "loss": "teacher Top1 hard CE",
        "heldout_seed": SEED_HOLDOUT,
        "heldout_matches": MATCHES_HOLDOUT,
        "heldout_reused_for_model_selection": True,
        "confirmed_independent_generalization": False,
        "independent_unit": "accepted teacher complete Match seed",
        "teacher_visited_only": True,
        "phase_only_model": True,
        "league_training_allowed": False,
        "champion_promoted": False,
        "full_input_alias_audit": feature_sufficiency_audit(held),
        "arms": {},
    }
    for n in TRAIN_MATCHES:
        # First 128 independent accepted Matches are a strict subset of 256.
        events = selection_events(train_traces[:n])
        for width in WIDTHS:
            arm = f"structured-hard-m{n}-w{width}"
            net, losses = train(events, "structured-hard",
                                width=width, epochs=6, seed=57720)
            if n == 256 and width == 96:
                compare_earlier_96_model(net, previous_model)
            metrics, _ = evaluate(net, held)
            filename = arm + ".pt"
            torch.save(net.state_dict(), directory / filename)
            digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            reference["arms"][arm] = {
                "train_complete_matches": n, "train_decisions": len(events),
                "network_width": width, "weight_file": filename,
                "weight_sha256": digest, "epoch_train_losses": losses,
                "development_heldout": metrics,
                "teacher_gap_breakdown": gap_diagnostics(net, held),
            }
            print(json.dumps({"arm": arm, "heldout_top1": metrics["top1"],
                              "heldout_regret": metrics["regret"]}), flush=True)
    (directory / "selection-followup-report.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = [
        "# I5 選出：学習データ量×モデル容量／教師僅差分析",
        "", "**探索段階**：既使用128教師Match、選出教師訪問局面のみ。",
        "結果は独立未使用検証でも対AQ実戦強度でもない。",
        "", "| 教師Match | width | Top1 | Top10 | regret ↓ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in reference["arms"].values():
        m = row["development_heldout"]
        lines.append(
            f"| {row['train_complete_matches']} | {row['network_width']} | "
            f"{m['top1']:.2%} | {m['top10']:.2%} | {m['regret']:.4f} |")
    lines.extend(["", "Gap別・3体一致率・3体正解時の先発一致率はJSONに記録。",
                  "独立seedクラスタ数が小さいbucketで勝敗や優位性を判定しない。",
                  "フル入力重複が0件でも情報の十分性は証明されない。",
                  "Draft>50%, Selection>=80%, Battle>=80% の統合方策ゲートは未通過。",
                  ""])
    (directory / "FOLLOWUP_REPORT.md").write_text("\n".join(lines))
    return reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True)
    p.add_argument("--heldout-dir", required=True)
    p.add_argument("--previous-model", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args.train_dir, args.heldout_dir, args.previous_model, args.out)


if __name__ == "__main__":
    main()
