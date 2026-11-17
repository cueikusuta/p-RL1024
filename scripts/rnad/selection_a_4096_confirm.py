"""Independent 2048 complete Match A-4096 vs frozen A-2048 comparison.

Selection target is now >= 50% Top1, unchanged draft >50 and battle >=80.
Selection standalone cannot unlock the three-phase integrated-policy gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from scripts.rnad.selection_ab import (
    DecomposedTeacher,read_events,evaluated_logits,summarize
)
from scripts.rnad.selection_a_2048_confirm import lock_study as lock_previous
from scripts.rnad.selection_a_4096 import (
    SCHEMA,TRAIN_BLOCKS,DEV_BLOCK,SEEDS,CHECKPOINT_EPOCHS,TRAIN_MATCHES
)

HOLDOUT_START,HOLDOUT_MATCHES=2496001,2048
SELECTION_THRESHOLD=.50


def paired_seed_cluster_bootstrap(old_seeds,new_seeds):
    if set(old_seeds)!=set(new_seeds) or len(old_seeds)!=HOLDOUT_MATCHES:
        raise ValueError("Not 2048 distinct paired heldout complete Match seeds")
    differences=[new_seeds[k]-old_seeds[k] for k in sorted(old_seeds)]
    n=len(differences)
    rng=random.Random(40961117)
    bootstrap=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n
                     for _ in range(10000))
    return {"complete_match_seed_clusters":n,
            "paired_cluster_mean_top1_delta":sum(differences)/n,
            "bootstrap_repetitions":10000,
            "approximate_two_sided_95pct_ci":[bootstrap[250],bootstrap[9749]],
            "improvement_supported_at_approximate_95pct":
                bootstrap[250]>0}


def lock_study(study_json,new_models,prior_study,prior_models,
               old_study,old_models,orig_study,orig_models):
    study=json.loads(Path(study_json).read_bytes())
    if (study.get("schema")!=SCHEMA or
        study.get("train_blocks")!=[list(x) for x in TRAIN_BLOCKS] or
        study.get("train_matches")!=TRAIN_MATCHES or
        study.get("dev_block")!=list(DEV_BLOCK) or
        study.get("seeds")!=list(SEEDS) or
        study.get("epochs")!=list(CHECKPOINT_EPOCHS) or
        study.get("dev_reused") is not True or
        study.get("new_fresh_block_loaded_during_model_selection") is not False or
        study.get("league_training_started") is not False or
        study.get("champion_promoted") is not False or
        study.get("pr_merged") is not False):
        raise ValueError("Invalid immutable A4096 registered study")
    trials=list(study.get("models",{}).values())
    if len(trials)!=len(SEEDS)*len(CHECKPOINT_EPOCHS) or set(
        (r["seed"],r["epochs"]) for r in trials)!=set(
        (s,e) for s in SEEDS for e in CHECKPOINT_EPOCHS):
        raise ValueError("A4096 frozen grid incomplete")
    best=min(trials,key=lambda r:(
        -r["previously_used_dev"]["top1"],
        r["previously_used_dev"]["regret"],
        r["epochs"],r["seed"]))
    if best["filename"]!=study.get("preselected_model") or best["filename"]!=(
        f"selection-A-4096-e{best['epochs']}-s{best['seed']}.pt"):
        raise ValueError("Candidate postselected after holdout")
    path=Path(new_models)/best["filename"]
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=best["sha256"]:
        raise ValueError("Selected A4096 model bytes were modified")
    torch.manual_seed(best["seed"])
    candidate=DecomposedTeacher(study["vocab"])
    candidate.load_state_dict(
        torch.load(path,map_location="cpu",weights_only=True),strict=True)
    frozen,_,frozen_meta,registry=lock_previous(
        prior_study,prior_models,old_study,old_models,orig_study,orig_models)
    if (registry["teacher_source_sha"]!=study["teacher_source_sha"] or
        registry["engine_sha"]!=study["engine_sha"] or
        registry["vocab"]!=study["vocab"] or
        frozen_meta["sha256"]!=study["frozen_A_2048"]["sha256"] or
        frozen_meta["filename"]!=study["frozen_A_2048"]["filename"]):
        raise ValueError("Frozen A2048 comparator provenance changed")
    return candidate.eval(),frozen.eval(),best,study


def assess(study_json,new_models,prior_study,prior_models,
           old_study,old_models,orig_study,orig_models,heldout_dir,out):
    torch.set_num_threads(1)
    # Freeze selected and reference model SHA before fresh Match labels are read.
    selected,frozen,best,study=lock_study(
        study_json,new_models,prior_study,prior_models,
        old_study,old_models,orig_study,orig_models)
    events,vocab,engine=read_events(
        heldout_dir,HOLDOUT_START,HOLDOUT_MATCHES,
        expected_sha=study["teacher_source_sha"])
    if vocab!=study["vocab"] or engine!=study["engine_sha"]:
        raise ValueError("Holdout engine, source or public vocab changed")
    old_metrics,old_by_seed=summarize(evaluated_logits(frozen,events),events)
    new_metrics,new_by_seed=summarize(evaluated_logits(selected,events),events)
    paired=paired_seed_cluster_bootstrap(old_by_seed,new_by_seed)
    cluster_mean=sum(new_by_seed.values())/HOLDOUT_MATCHES
    # Formally the selection-only component of the league 3-phase Bonferroni
    # gate. Not evidence for an integrated three-phase policy. The gate itself
    # demands same integrated policy, draft and battle phase, and further proofs.
    radius=math.sqrt(math.log(3/.2)/(2*HOLDOUT_MATCHES))
    lower=max(0.,cluster_mean-radius)
    status={
        "target":SELECTION_THRESHOLD,
        "decision_weighted_top1_meets_target":
            new_metrics["top1"]>=SELECTION_THRESHOLD,
        "seed_cluster_mean_top1":cluster_mean,
        "independent_seed_clusters":HOLDOUT_MATCHES,
        "familywise_80pct_one_sided_3phase_hoeffding_lower_bound":lower,
        "selection_only_80pct_lower_meets_target":lower>=SELECTION_THRESHOLD,
        "counts_toward_integrated_league_gate":False
    }
    report={
        "schema":"rnad-i5-selection-A-4096-fresh1024-v1",
        "training_matches":TRAIN_MATCHES,
        "holdout_seed_start":HOLDOUT_START,
        "holdout_complete_teacher_matches":HOLDOUT_MATCHES,
        "selected_model_filename":best["filename"],
        "selected_model_sha256":best["sha256"],
        "frozen_comparator_sha256":study["frozen_A_2048"]["sha256"],
        "training_seed_nonoverlap":True,"chosen_before_new_holdout":True,
        "teacher_source_sha":study["teacher_source_sha"],
        "engine_sha":engine,
        "old_A_2048":old_metrics,"new_A_4096":new_metrics,
        "paired_difference":paired,"selection_50pct_gate_diagnostic":status,
        "teacher_top1_not_actual_winrate":True,
        "draft_battle_unchanged":True,
        "integrated_three_phase_gate_passed":False,
        "league_started":False,"champion_promoted":False,"pr_merged":False
    }
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    (root/"SELECTION_A_4096_FRESH.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    low,high=paired["approximate_two_sided_95pct_ci"]
    lines=["# A-4096 未使用2048完全Match独立検証","",
        f"固定候補：{best['filename']}",
        f"旧A2048 Top1: **{old_metrics['top1']:.2%}**",
        f"新A4096 Top1: **{new_metrics['top1']:.2%}**",
        f"対応ありMatch-cluster平均Top1差：**{paired['paired_cluster_mean_top1_delta']:+.2%}**",
        f"近似95%両側cluster bootstrap CI：**[{low:+.2%},{high:+.2%}]**",
        f"差の正値支持：{paired['improvement_supported_at_approximate_95pct']}",
        f"選出目標Top1 >=50%の点推定合格：{status['decision_weighted_top1_meets_target']}",
        f"3相Bonferroni相当の選出-only one-sided 80%下限：**{lower:.2%}**",
        f"選出-onlyの同下限 >=50%：{status['selection_only_80pct_lower_meets_target']}",
        "",
        "モデルは選出専用。これだけで同一統合AIの3相リーグゲートは通らない。",
        "これは教師Top1の模倣率であり、完全Match実戦勝率ではない。",
        "リーグ開始・Champion昇格・PRマージなし。"]
    (root/"SELECTION_A_4096_FRESH.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("study","new-models","prior-study","prior-models",
                "old-study","old-models","orig-study","orig-models",
                "heldout","out"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    assess(a.study,a.new_models,a.prior_study,a.prior_models,
           a.old_study,a.old_models,a.orig_study,a.orig_models,a.heldout,a.out)
