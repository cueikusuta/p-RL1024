"""I5 A-only selection imitation scale-up, no league or practical win-rate claims.

A-512 selection is frozen. Train 3 independent A-1024 seeds and save
checkpoints at 20/40 epochs. Rank only on the already reused 128-Match
development set; the new 512-Match block must stay unread until fixed.
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
    BATCH, DecomposedTeacher, read_events, objective,
    evaluated_logits, summarize,
)
from scripts.rnad.selection_ab_scaling_confirm import load_registered_scaling

TRAIN_BLOCKS=((2471001,256),(2477001,256),(2479001,512))
DEV_BLOCK=(2472001,128)
EPOCHS=(20,40)
SEEDS=(90251,90252,90253)
N_MATCHES=1024
SCHEMA="rnad-i5-selection-A-1024-supervised-study-v1"


def collect(train_a,train_b,train_c,dev_path):
    sources=[(train_a,*TRAIN_BLOCKS[0]),
             (train_b,*TRAIN_BLOCKS[1]),
             (train_c,*TRAIN_BLOCKS[2]),
             (dev_path,*DEV_BLOCK)]
    rows=[]
    vocab=engine=source=None
    sets=[]
    for root,start,n in sources:
        batch, v, e=read_events(root,start,n,expected_sha=source)
        s=batch[0][1]["trace_source_sha"]
        if source is None: source=s
        if vocab is None: vocab=v
        if engine is None: engine=e
        if s!=source or v!=vocab or e!=engine:
            raise ValueError("Mixed engine or rich actor vocabulary / source")
        sets.append(set(seed for seed,_ in batch))
        if len(sets[-1])!=n:
            raise ValueError("Incomplete accepted teacher Match count")
        rows.append(batch)
    if any(sets[i]&sets[j] for i in range(len(sets))
           for j in range(i+1,len(sets))):
        raise ValueError("Training and dev accepted seed overlap")
    if sum(len(s) for s in sets[:3])!=N_MATCHES:
        raise ValueError("Training must contain 1024 distinct complete Matches")
    return rows[:3],rows[3],vocab,engine,source


def train_one(seed,train_rows,dev,vocab,directory):
    torch.manual_seed(seed)
    model=DecomposedTeacher(vocab)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.0001)
    rng=random.Random(seed)
    curve=[]
    result={}
    for epoch in range(1,max(EPOCHS)+1):
        model.train()
        order=list(range(len(train_rows)))
        rng.shuffle(order)
        total=0.
        for begin in range(0,len(order),BATCH):
            batch=[train_rows[i] for i in order[begin:begin+BATCH]]
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model,batch,"A")
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite A-only scaling loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            optimizer.step()
            total+=float(loss.detach())*len(batch)
        curve.append(total/len(train_rows))
        if epoch in EPOCHS:
            model.eval()
            metrics,_=summarize(evaluated_logits(model,dev),dev)
            path=directory/f"selection-A-1024-e{epoch}-s{seed}.pt"
            torch.save(model.state_dict(),path)
            result[path.name]={
                "kind":"A","training_complete_matches":N_MATCHES,
                "epochs":epoch,"seed":seed,"filename":path.name,
                "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "development_metrics":metrics,
                "training_losses_through_epoch":list(curve),
            }
            print(json.dumps({"model":path.name,"dev":metrics}),flush=True)
        if epoch%5==0:
            print(json.dumps({"arm":"A","seed":seed,"epoch":epoch,
                              "loss":curve[-1]}),flush=True)
    return result


def run(train_a,train_b,train_c,dev_dir,old_models,old_report,out):
    torch.set_num_threads(1)
    blocks,dev,vocab,engine,source=collect(
        train_a,train_b,train_c,dev_dir)
    old,registry=load_registered_scaling(old_models,old_report)
    if (registry["source_sha"]!=source or
        registry["engine_sha"]!=engine or registry["vocab"]!=vocab):
        raise ValueError("Old 512 baseline source, engine or feature mismatch")
    prev,prev_meta=old["A"]
    baseline,baseline_by_seed=summarize(evaluated_logits(prev,dev),dev)
    expected=prev_meta["reused_development_metrics"]["top1"]
    if abs(baseline["top1"]-expected)>1e-10:
        raise ValueError("Frozen A-512 baseline dev mismatch")
    directory=Path(out)
    directory.mkdir(parents=True,exist_ok=False)
    # Avoid applying the same loss backwards through stale order list; the
    # sample order is freshly shuffled per epoch with fixed seed.
    merged=sum(blocks,[])
    trials={}
    for seed in SEEDS:
        trials.update(train_one(seed,merged,dev,vocab,directory))
    if len(trials)!=len(SEEDS)*len(EPOCHS):
        raise ValueError("Missing A-only checkpoint")
    chosen=min(trials.values(),key=lambda row:(
        -row["development_metrics"]["top1"],
        row["development_metrics"]["regret"],
        row["epochs"],row["seed"]))
    result={
        "schema":SCHEMA,
        "trainer":"teacher-component-core20-lead6-A",
        "teacher_source_sha":source,"engine_sha":engine,"vocab":vocab,
        "train_blocks":[list(x) for x in TRAIN_BLOCKS],
        "development_block":list(DEV_BLOCK),
        "train_matches":N_MATCHES,"development_reused":True,
        "seed_grid":list(SEEDS),"epoch_grid":list(EPOCHS),
        "historical_512_A_model":{
            "filename":prev_meta["model_filename"],
            "sha256":prev_meta["sha256"],"development_metrics":baseline},
        "trials":trials,
        "selected_checkpoint_before_new_holdout":chosen["filename"],
        "holdout_read_during_selection":False,
        "new_holdout_result":None,
        "teacher_labels_only_in_loss":True,
        "opponent_hidden_sets_used":False,
        "league_started":False,"champion_promoted":False,
        "pr_merged":False,
    }
    (directory/"selection-A-1024-development.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=[
        "# I5 Aのみ：1024完全Match教師あり追加学習・既使用dev評価","",
        "A=教師core20＋先発lead6分解。旧256+256+512の受理済みMatchを学習に再利用。",
        "開発128Matchは既使用。以下の値だけで改善の確証とはしない。","",
        "| モデル | epoch | seed | Top1 | Top10 | 3体一致 | regret↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 旧A-512固定 | {prev_meta['epochs']} | {prev_meta['seed']} | "
        f"{baseline['top1']:.2%} | {baseline['top10']:.2%} | "
        f"{baseline['trio']:.2%} | {baseline['regret']:.4f} |",
    ]
    for row in trials.values():
        metrics=row["development_metrics"]
        lines.append(
            f"| A-1024 | {row['epochs']} | {row['seed']} | "
            f"{metrics['top1']:.2%} | {metrics['top10']:.2%} | "
            f"{metrics['trio']:.2%} | {metrics['regret']:.4f} |")
    lines.extend(["",f"未使用Matchを開く前の固定候補：{chosen['filename']}",
                  "新規未使用512完全Matchの対応あり95%CIで判定する。",
                  "教師Top1模倣は実戦勝率ではない。リーグ／Champion／PRマージなし。"])
    (directory/"SELECTION_A_1024_DEV.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("train-a","train-b","train-c","dev-dir","old-models",
                "old-report","out"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    run(a.train_a,a.train_b,a.train_c,a.dev_dir,a.old_models,a.old_report,a.out)
