"""A/B 512-Match scaling independently confirmed on 512 NEW accepted Matches.

Both arm+checkpoint+training-seed choices are fixed using reused development
data before touching fresh labels. Bonferroni: two planned paired comparisons,
97.5% individual two-sided seed-cluster bootstrap intervals (FWER~95%).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from scripts.rnad.selection_ab import (
    DecomposedTeacher, DetailedAttention, read_events, evaluated_logits, summarize,
)
from scripts.rnad.selection_ab_scaling import (
    SCHEMA as STUDY_SCHEMA, load_old, SEEDS, CHECKPOINT_EPOCHS, TRAIN_BLOCKS, DEV,
)

FRESH_START, FRESH_MATCHES = 2479001, 512


def paired_bonferroni_two(base, candidate):
    if set(base)!=set(candidate) or len(base)!=FRESH_MATCHES:
        raise ValueError("Not 512 complete independent paired Match seed clusters")
    differences=[candidate[k]-base[k] for k in sorted(base)]
    n=len(differences)
    rng=random.Random(281718)
    samples=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n
                   for _ in range(10000))
    # Two planned arm-specific comparisons: 97.5% two-sided each.
    return {
        "clusters":n,
        "mean_match_cluster_difference":sum(differences)/n,
        "approximate_97_5pct_two_sided_cluster_bootstrap_ci":
            [samples[125],samples[9874]],
        "familywise_95pct_two_comparison_bonferroni":True,
        "approximate_interval_only":True,
    }


def load_registered_scaling(models_dir,study_json):
    study=json.loads(Path(study_json).read_bytes())
    if (study.get("schema")!=STUDY_SCHEMA or study.get("train_matches")!=512 or
        study.get("train_blocks")!=[list(x) for x in TRAIN_BLOCKS] or
        study.get("development_block")!=list(DEV) or
        study.get("dev_previously_reused") is not True or
        study.get("used_previous_fresh_block_as_training") is not True or
        study.get("fresh_confirmation_run") is not None or
        study.get("league_training_started") is not False or
        study.get("champion_promoted") is not False or
        set(study.get("dev_selected_models",{}))!={"A","B"}):
        raise ValueError("Not a fixed preregistered A/B scaling study")
    models={}
    for kind,cls in (("A",DecomposedTeacher),("B",DetailedAttention)):
        trials=[r for r in study["trials"].values() if r["arm"]==kind]
        if len(trials)!=len(SEEDS)*len(CHECKPOINT_EPOCHS) or not all(
            r["seed"] in SEEDS and r["epochs"] in CHECKPOINT_EPOCHS
            for r in trials):
            raise ValueError("Incomplete fixed planned model selection")
        best=min(trials,key=lambda x:(
            -x["reused_development_metrics"]["top1"],
            x["reused_development_metrics"]["regret"],
            x["epochs"],x["seed"]))
        filename=best["model_filename"]
        if study["dev_selected_models"][kind]!=filename:
            raise ValueError("Selected model was changed after development")
        path=Path(models_dir)/filename
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=best["sha256"]:
            raise ValueError("Untrusted selected weight bytes")
        torch.manual_seed(best["seed"])
        model=cls(study["vocab"])
        model.load_state_dict(
            torch.load(path,map_location="cpu",weights_only=True),strict=True)
        models[kind]=(model.eval(),best)
    return models,study


def evaluate(models_dir,study_json,old_dir,old_report,fresh_dir,out):
    torch.set_num_threads(1)
    selected,study=load_registered_scaling(models_dir,study_json)
    # Confirm models/checksums locked BEFORE fresh labels are read.
    held,vocab,engine=read_events(
        fresh_dir,FRESH_START,FRESH_MATCHES,expected_sha=study["source_sha"])
    if study["vocab"]!=vocab or study["engine_sha"]!=engine:
        raise ValueError("Inconsistent source / engine / vocabulary on fresh data")
    legacy=json.loads(Path(old_report).read_bytes())
    if (legacy["teacher_source_sha"]!=study["source_sha"] or
        legacy["engine_sha"]!=engine):
        raise ValueError("Mismatched historical frozen baseline provenance")
    report={"schema":"rnad-i5-selection-AB-scaling-fresh512-v1",
            "fresh_seed_start":FRESH_START,"fresh_independent_matches":FRESH_MATCHES,
            "data_source_sha":study["source_sha"],"engine_sha":engine,
            "dev_block_previously_used":True,"training_uses_old_confirmation_block":True,
            "chosen_without_fresh_labels":True,"arms":{},
            "no_league":True,"no_champion":True,
            "teacher_top1_not_match_win_rate":True,
            "integrated_three_phase_gate_passed":False}
    for kind,cls in (("A",DecomposedTeacher),("B",DetailedAttention)):
        baseline=load_old(old_dir,legacy,vocab,kind)
        old_metrics,old_seed=summarize(evaluated_logits(baseline,held),held)
        model,meta=selected[kind]
        new_metrics,new_seed=summarize(evaluated_logits(model,held),held)
        paired=paired_bonferroni_two(old_seed,new_seed)
        report["arms"][kind]={
            "baseline_256_match_10epoch":old_metrics,
            "scaled_512_match":new_metrics,
            "scaled_model_filename":meta["model_filename"],
            "scaled_model_sha256":meta["sha256"],
            "paired":paired,
            "familywise_positive_improvement_supported":
                paired["approximate_97_5pct_two_sided_cluster_bootstrap_ci"][0]>0,
        }
    directory=Path(out)
    directory.mkdir(parents=True,exist_ok=False)
    (directory/"selection-AB-scaling-fresh512.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# A/B 512学習規模：未使用512完全Match独立確認","",
        "旧256Match×10epochとの同一seed完全Match比較。2比較の多重性をBonferroni補正。",
        "使用済みデータを学習に転用した部分を新規未使用検証と混同しない。","",
        "| 方式 | 旧Top1 | 新Top1 | 対応ありMatch平均差 | 近似97.5%CI |",
        "|---|---:|---:|---:|---|"]
    for kind,row in report["arms"].items():
        low,high=row["paired"]["approximate_97_5pct_two_sided_cluster_bootstrap_ci"]
        lines.append(
            f"| {kind} | {row['baseline_256_match_10epoch']['top1']:.2%} | "
            f"{row['scaled_512_match']['top1']:.2%} | "
            f"{row['paired']['mean_match_cluster_difference']:+.2%} | "
            f"[{low:+.2%}, {high:+.2%}] |")
    lines.extend(["",
        "これは受理済み教師訪問局面での模倣Top1であり実戦勝率ではない。",
        "3フェーズ統合模倣閾値未達につきリーグ学習／Champion昇格なし。"])
    (directory/"SELECTION_AB_SCALING_FRESH.md").write_text(
        "\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for item in ("models-dir","study-json","old-dir","old-report","fresh-dir","out"):
        parser.add_argument("--"+item,required=True)
    args=parser.parse_args()
    evaluate(args.models_dir,args.study_json,args.old_dir,args.old_report,
             args.fresh_dir,args.out)
