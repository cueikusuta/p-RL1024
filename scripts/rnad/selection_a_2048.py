"""I5 A-only 2048 accepted teacher Match supervised scale-up.

Research only. New train data and heldout Match identities are fixed in the
workflow launch manifest. Frozen A-1024 comparator is loaded by SHA before
dev choice; this script never reads the new 1024-Match holdout labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from scripts.rnad.selection_ab import (
    DecomposedTeacher, BATCH, read_events, objective, evaluated_logits, summarize
)
from scripts.rnad.selection_a_1024_confirm import lock_models

SCHEMA="rnad-i5-selection-A-2048-supervised-study-v1"
TRAIN_BLOCKS=((2471001,256),(2477001,256),(2479001,512),
              (2481001,512),(2483001,512))
DEV_BLOCK=(2472001,128)
TRAIN_MATCHES=2048
SEEDS=(90251,90252,90253)
CHECKPOINT_EPOCHS=(40,60)
FROZEN_TEACHER_SOURCE_COMMIT="a7224b05eeef49650794b24546576119dffed951"


def collect(dirs,devdir):
    if len(dirs)!=len(TRAIN_BLOCKS):
        raise ValueError("Exactly five non-overlapping accepted training blocks")
    batches=[]; all_clusters=[]
    vocab=engine=source=None
    for path,(start,n) in [*zip(dirs,TRAIN_BLOCKS),(devdir,DEV_BLOCK)]:
        rows,cat,run_engine=read_events(path,start,n,expected_sha=source)
        if len({seed for seed,_ in rows})!=n:
            raise ValueError("Partial accepted complete teacher Matches")
        sha=rows[0][1]["trace_source_sha"]
        if source is None: source=sha
        if vocab is None: vocab=cat
        if engine is None: engine=run_engine
        if cat!=vocab or run_engine!=engine or sha!=source:
            raise ValueError("Teacher, engine, vocab mismatch")
        batches.append(rows)
        all_clusters.append(set(seed for seed,_ in rows))
    if any(all_clusters[i]&all_clusters[j]
           for i in range(len(all_clusters))
           for j in range(i+1,len(all_clusters))):
        raise ValueError("Training/development seed reuse")
    if sum(map(len,all_clusters[:-1]))!=TRAIN_MATCHES:
        raise ValueError("Missing exactly 2048 independent training Match clusters")
    return sum(batches[:-1],[]),batches[-1],vocab,engine,source


def fit_one(seed,train,dev,vocab,root):
    torch.manual_seed(seed)
    net=DecomposedTeacher(vocab)
    optim=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001)
    rng=random.Random(seed)
    curve=[]
    checkpoints={}
    for epoch in range(1,max(CHECKPOINT_EPOCHS)+1):
        net.train()
        indexes=list(range(len(train)))
        rng.shuffle(indexes)
        total=0.
        for offset in range(0,len(indexes),BATCH):
            rows=[train[i] for i in indexes[offset:offset+BATCH]]
            optim.zero_grad(set_to_none=True)
            loss=objective(net,rows,"A")
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite A-only loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            optim.step()
            total+=float(loss.detach())*len(rows)
        curve.append(total/len(train))
        if epoch%10==0:
            print(json.dumps({"seed":seed,"epoch":epoch,"loss":curve[-1]}),flush=True)
        if epoch in CHECKPOINT_EPOCHS:
            net.eval()
            dev_metrics,_=summarize(evaluated_logits(net,dev),dev)
            name=f"selection-A-2048-e{epoch}-s{seed}.pt"
            path=root/name
            torch.save(net.state_dict(),path)
            checkpoints[name]={
                "filename":name,"arm":"A","seed":seed,"epochs":epoch,
                "training_complete_matches":TRAIN_MATCHES,
                "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "previously_used_dev":dev_metrics,
                "loss_curve":list(curve),
            }
            print(json.dumps({"checkpoint":name,"dev":dev_metrics}),flush=True)
    return checkpoints


def train(dirs,devdir,old_study,old_dir,original_study,original_dir,out):
    torch.set_num_threads(1)
    rows,dev,vocab,engine,source=collect(dirs,devdir)
    # Frozen baseline from exact A-1024 selected checkpoint, verified via
    # SHA and trusted model registry rather than arbitrary caller weights.
    old_model,_,old_meta,old_registry=lock_models(
        old_study,old_dir,original_study,original_dir)
    if (old_registry["teacher_source_sha"]!=source or
        old_registry["engine_sha"]!=engine or old_registry["vocab"]!=vocab):
        raise ValueError("A-1024 frozen comparison provenance changed")
    prior_metrics,_=summarize(evaluated_logits(old_model,dev),dev)
    if abs(prior_metrics["top1"]-
           old_meta["development_metrics"]["top1"])>1e-10:
        raise ValueError("Old A-1024 development Top1 not reproducible")
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    trials={}
    for seed in SEEDS:
        trials.update(fit_one(seed,rows,dev,vocab,root))
    if len(trials)!=len(SEEDS)*len(CHECKPOINT_EPOCHS):
        raise ValueError("Incomplete planned three-seed 40/60 grid")
    best=min(trials.values(),key=lambda row:(
        -row["previously_used_dev"]["top1"],
        row["previously_used_dev"]["regret"],
        row["epochs"],row["seed"]))
    report={
        "schema":SCHEMA,
        "train_blocks":[list(x) for x in TRAIN_BLOCKS],
        "train_matches":TRAIN_MATCHES,
        "dev_block":list(DEV_BLOCK),
        "dev_reused":True,"new_fresh_block_loaded_during_model_selection":False,
        "seeds":list(SEEDS),"epochs":list(CHECKPOINT_EPOCHS),
        "teacher_source_sha":source,"engine_sha":engine,"vocab":vocab,
        "frozen_A_1024":{ "filename":old_meta["filename"],
                         "sha256":old_meta["sha256"],
                         "dev":prior_metrics},
        "models":trials,
        "preselected_model":best["filename"],
        "teacher_labels_are_training_targets_only":True,
        "opponent_hidden_info_used":False,
        "league_training_started":False,
        "champion_promoted":False,
        "pr_merged":False,
    }
    (root/"selection-A-2048-development.json").write_text(
        json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    lines=[
        "# A専用2048完全Match 学習・再使用dev 128Matchの探索結果",
        "",
        "開発データはモデル選択に再使用済み。新規未使用1024 Matchで別評価。",
        "",
        "| checkpoint | Top1 | Top10 | trio | normalized regret |",
        "|---|---:|---:|---:|---:|",
        f"| frozen A-1024 | {prior_metrics['top1']:.2%} |"
        f" {prior_metrics['top10']:.2%} | {prior_metrics['trio']:.2%}"
        f" | {prior_metrics['regret']:.4f} |"]
    for name,v in trials.items():
        m=v["previously_used_dev"]
        lines.append(f"| {name} | {m['top1']:.2%} | {m['top10']:.2%} |"
                     f" {m['trio']:.2%} | {m['regret']:.4f} |")
    lines.extend(["",f"Preselected before new evaluation: {best['filename']}",
                  "Model Top1 != real four-win Match win rate.",
                  "No league training, promotion, or PR merge."])
    (root/"SELECTION_A_2048_DEV.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for i in range(1,6):p.add_argument(f"--train-{i}",required=True)
    for key in ("dev","old-study","old-dir","original-study","original-dir","out"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    train([getattr(a,f"train_{i}") for i in range(1,6)],
          a.dev,a.old_study,a.old_dir,a.original_study,a.original_dir,a.out)
