"""Independent new 512 complete Match paired A-1024 vs frozen A-512 Top1.

Commit to one training-only checkpoint on previously reused 128-Match dev
before loading any genuinely unseen holdout accepted teacher Match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from scripts.rnad.selection_ab import (
    DecomposedTeacher, read_events, evaluated_logits, summarize,
)
from scripts.rnad.selection_ab_scaling_confirm import load_registered_scaling
from scripts.rnad.selection_a_1024 import (
    SCHEMA as STUDY_SCHEMA, TRAIN_BLOCKS, DEV_BLOCK, SEEDS, EPOCHS,
)

HOLDOUT_START=2481001
HOLDOUT_MATCHES=512


def cluster_delta(a,b):
    if set(a)!=set(b) or len(a)!=HOLDOUT_MATCHES:
        raise ValueError("Paired independent 512 accepted Match clusters required")
    diffs=[b[s]-a[s] for s in sorted(a)]
    n=len(diffs)
    rng=random.Random(362010)
    bootstrap=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n
                     for _ in range(10000))
    return {"independent_complete_Match_seed_clusters":n,
            "paired_Match_mean_delta":sum(diffs)/n,
            "bootstrap_repetitions":10000,
            "approximate_two_sided_95pct_ci":[bootstrap[250],bootstrap[9749]]}


def lock_models(study_json,new_models,old_report,old_models):
    study=json.loads(Path(study_json).read_bytes())
    if (study.get("schema")!=STUDY_SCHEMA or study.get("train_matches")!=1024
        or study.get("train_blocks")!=[list(x) for x in TRAIN_BLOCKS]
        or study.get("development_block")!=list(DEV_BLOCK)
        or study.get("development_reused") is not True
        or study.get("seed_grid")!=list(SEEDS)
        or study.get("epoch_grid")!=list(EPOCHS)
        or study.get("holdout_read_during_selection") is not False
        or study.get("new_holdout_result") is not None
        or study.get("league_started") is not False
        or study.get("champion_promoted") is not False
        or study.get("pr_merged") is not False
        or len(study.get("trials",{}))!=len(SEEDS)*len(EPOCHS)):
        raise ValueError("Invalid immutable A-1024 training selection registry")
    rows=list(study["trials"].values())
    if set((r["seed"],r["epochs"]) for r in rows)!=set(
        (s,e) for s in SEEDS for e in EPOCHS):
        raise ValueError("Missing or duplicated A-1024 train grid")
    best=min(rows,key=lambda r:(
        -r["development_metrics"]["top1"],
        r["development_metrics"]["regret"],
        r["epochs"],r["seed"]))
    if study.get("selected_checkpoint_before_new_holdout")!=best["filename"]:
        raise ValueError("Changed A-1024 checkpoint after model selection")
    expected=f"selection-A-1024-e{best['epochs']}-s{best['seed']}.pt"
    if best["filename"]!=expected:
        raise ValueError("A-1024 unexpected checkpoint filename")
    model_path=Path(new_models)/expected
    sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if sha!=best["sha256"]:
        raise ValueError("Changed frozen A-1024 checkpoint bytes")
    torch.manual_seed(best["seed"])
    selected=DecomposedTeacher(study["vocab"])
    selected.load_state_dict(
        torch.load(model_path,map_location="cpu",weights_only=True),strict=True)
    old,history=load_registered_scaling(old_models,old_report)
    if (history["source_sha"]!=study["teacher_source_sha"]
        or history["engine_sha"]!=study["engine_sha"]
        or history["vocab"]!=study["vocab"]):
        raise ValueError("Training baseline / source / vocabulary mismatch")
    legacy,legacy_meta=old["A"]
    if (legacy_meta["sha256"]!=study["historical_512_A_model"]["sha256"]
        or legacy_meta["model_filename"]!=
           study["historical_512_A_model"]["filename"]):
        raise ValueError("Unpinned old A-512 comparator")
    return selected.eval(),legacy.eval(),best,study


def run(study_json,new_models,old_report,old_models,fresh_dir,out):
    torch.set_num_threads(1)
    # Lock both SHA-256 checked checkpoints first. Never inspect holdout
    # for selection, adaptation, hyperparameter adjustment or refitting.
    trained,baseline,chosen,study=lock_models(
        study_json,new_models,old_report,old_models)
    events,vocab,engine=read_events(
        fresh_dir,HOLDOUT_START,HOLDOUT_MATCHES,
        expected_sha=study["teacher_source_sha"])
    if vocab!=study["vocab"] or engine!=study["engine_sha"]:
        raise ValueError("Independent complete Match incompatible engine / encoding")
    old,old_seed=summarize(evaluated_logits(baseline,events),events)
    new,new_seed=summarize(evaluated_logits(trained,events),events)
    paired=cluster_delta(old_seed,new_seed)
    result={
        "schema":"rnad-i5-selection-A-1024-fresh512-confirm-v1",
        "original_train_matches":1024,
        "independent_holdout_seed_start":HOLDOUT_START,
        "independent_holdout_Matches":HOLDOUT_MATCHES,
        "old_training_matches":512,
        "old_model_sha256":study["historical_512_A_model"]["sha256"],
        "selected_model_sha256":chosen["sha256"],
        "selected_model_name":chosen["filename"],
        "selection_exclusively_on_reused_development":True,
        "teacher_source_sha":study["teacher_source_sha"],
        "engine_sha":engine,"vocab":vocab,
        "old_A_512_metrics":old,"new_A_1024_metrics":new,
        "paired":paired,
        "imitation_improvement_supported_at_approximate_95pct":
            paired["approximate_two_sided_95pct_ci"][0]>0,
        "this_is_teacher_top1_not_real_game_winrate":True,
        "three_phase_integrated_policy_gate_passed":False,
        "league_started":False,"champion_promoted":False,"pr_merged":False,
    }
    folder=Path(out)
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"SELECTION_A_1024_FRESH.json").write_text(
        json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    low,high=paired["approximate_two_sided_95pct_ci"]
    lines=[
        "# I5 A-1024 新規未使用512完全Matchの独立確認","",
        f"開発データで事前固定：{chosen['filename']}",
        f"旧A-512 Top1: **{old['top1']:.2%}**",
        f"新A-1024 Top1: **{new['top1']:.2%}**",
        f"同じ512 Matchの対応ありクラスタ平均差: **{paired['paired_Match_mean_delta']:+.2%}**",
        f"近似95%両側クラスタbootstrap CI: **[{low:+.2%},{high:+.2%}]**",
        f"独立模倣改善を統計的支持: {result['imitation_improvement_supported_at_approximate_95pct']}",
        "",
        "判断単位は512独立の受理済み完全教師Match seed。",
        "教師Top1模倣は、AとBの実戦勝率差でも、AQ対戦勝率でもない。",
        "同一統合3フェーズ政策のゲート未通過、リーグ学習・Champion昇格・PRマージなし。",
    ]
    (folder/"SELECTION_A_1024_FRESH.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("study-json","new-models","old-report","old-models",
              "fresh-dir","out"):
        p.add_argument("--"+k,required=True)
    a=p.parse_args()
    run(a.study_json,a.new_models,a.old_report,a.old_models,a.fresh_dir,a.out)
