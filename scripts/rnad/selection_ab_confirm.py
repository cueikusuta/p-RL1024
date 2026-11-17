"""Single-arm fresh 256 complete Match confirmation for A/B selection research.
The best of A/B is fixed on reused development data BEFORE reading the
new independent accepted public teacher traces. No league or Champion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.rnad.selection_ab import (
    DecomposedTeacher, DetailedAttention, read_events, evaluated_logits,
    summarize, baseline_scores, clustered_delta, SOURCE_TEACHER,
)
from scripts.rnad.selection_specialist_fresh_holdout import (
    from_archive, REFERENCE_SHA,
)

FRESH_START, FRESH_MATCHES = 2477001, 256


def load_preselected(model_dir, model_report):
    report=json.loads(Path(model_report).read_bytes())
    if (report.get("schema") != "rnad-i5-selection-AB-exploratory-v1" or
        report.get("train_complete_matches") != 256 or
        report.get("development_matches") != 128 or
        report.get("development_block_previously_used") is not True or
        report.get("confirmation_result") is not None or
        report.get("league_training_allowed") is not False or
        report.get("champion_promoted") is not False or
        report.get("teacher") != SOURCE_TEACHER):
        raise ValueError("Untrusted preregistered development experiment")
    selected=report.get("preselected_for_fresh_confirmation")
    if selected not in ("A","B") or set(report.get("models",{})) != {
        "A","B","structured-hard-frozen"}:
        raise ValueError("Unknown or nonunique A/B selected model")
    dev=report["models"]
    expected=min(("A","B"),key=lambda k:(
        -dev[k]["dev"]["top1"],dev[k]["dev"]["regret"],k))
    if selected!=expected or dev[selected].get("selected") is not True:
        raise ValueError("Selected candidate differs from prior frozen development plan")
    row=dev[selected]
    name=f"selection-{selected}.pt"
    if row.get("weights")!=name:
        raise ValueError("Unknown model artifact filename")
    path=Path(model_dir)/name
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=row.get("sha256"):
        raise ValueError("Tampered frozen A/B candidate weights")
    cls={"A":DecomposedTeacher,"B":DetailedAttention}[selected]
    model=cls(report["vocab"])
    model.load_state_dict(torch.load(path,map_location="cpu",
                                     weights_only=True),strict=True)
    return model.eval(),selected,sha,report


def evaluate(model_dir,model_report,reference,fresh_dir,out):
    torch.set_num_threads(1)
    chosen,selected,sha,study=load_preselected(model_dir,model_report)
    # Fresh data is loaded ONLY after candidate and SHA are locked.
    events,vocab,engine=read_events(fresh_dir,FRESH_START,FRESH_MATCHES)
    if vocab!=study["vocab"] or engine!=study["engine_sha"]:
        raise ValueError("Fresh accepted teacher engine/encoder differs")
    frozen=from_archive(reference,REFERENCE_SHA)
    old,old_by_seed=summarize(baseline_scores(frozen,events),events)
    new,new_by_seed=summarize(evaluated_logits(chosen,events),events)
    paired=clustered_delta(old_by_seed,new_by_seed,seed=51921)
    report={
        "schema":"rnad-i5-selection-AB-fresh-256-confirmation-v1",
        "selected_from_reused_development":selected,
        "fresh_seed_start":FRESH_START,
        "fresh_independent_complete_matches":FRESH_MATCHES,
        "train_seed_start":2471001,"dev_seed_start":2472001,
        "independent_unit":"accepted_complete_teacher_Match_seed",
        "teacher":SOURCE_TEACHER,
        "same_engine_sha":engine,"rich_actor_vocab":vocab,
        "candidate_model_sha256":sha,
        "frozen_structured_reference_sha256":REFERENCE_SHA,
        "baseline":old,"candidate":new,"paired":paired,
        "improvement_supported_at_approximate_95pct":
            paired["approximate_95pct_two_sided_ci"][0]>0,
        "teacher_visited_only":True,
        "teacher_top1_not_game_win_rate":True,
        "three_phase_integrated_policy_gate_passed":False,
        "league_training_allowed":False,"champion_promoted":False,
        "strength_established":False,
    }
    path=Path(out)
    path.mkdir(parents=True,exist_ok=False)
    (path/"selection-AB-fresh-confirmation.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    delta=paired["paired_cluster_mean_top1_delta"]
    low,high=paired["approximate_95pct_two_sided_ci"]
    lines=["# A/B 未使用256完全Match・選出Top1独立確認","",
        f"既使用devから事前固定した1モデル: {selected}",
        f"凍結structured-hard: {old['top1']:.2%}",
        f"{selected}: {new['top1']:.2%}",
        f"対応あり独立Match平均差: {delta:+.2%}",
        f"近似95%両側bootstrap CI: [{low:+.2%}, {high:+.2%}]",
        f"正の下限で改善を支持: {report['improvement_supported_at_approximate_95pct']}",
        "",
        "教師Top1模倣の比較のみ。勝率・リーグ開始・3フェーズ閾値の証明ではない。"]
    (path/"SELECTION_AB_FRESH.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",required=True)
    p.add_argument("--model-report",required=True)
    p.add_argument("--reference",required=True)
    p.add_argument("--fresh-dir",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    evaluate(args.model_dir,args.model_report,args.reference,
             args.fresh_dir,args.out)
