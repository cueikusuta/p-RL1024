"""I5 A/B fixed-cost supervised scaling: 512 accepted Matches, 10/20 epochs, 3 seeds.

The 512 training Matches are old 256 training + the 256 block previously
used to confirm the first A model; neither is a fresh holdout any longer.
Reused 128-Match dev ranks hyperparameters. This does NOT train a league AI.
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
    DecomposedTeacher, DetailedAttention, read_events, objective, evaluated_logits,
    summarize, baseline_scores, BATCH, SOURCE_TEACHER,
)
from scripts.rnad.selection_specialist_fresh_holdout import from_archive, REFERENCE_SHA

SCHEMA="rnad-i5-selection-AB-scaling-512-v1"
TRAIN_BLOCKS=((2471001,256),(2477001,256))
DEV=(2472001,128)
SEEDS=(90251,90252,90253)
CHECKPOINT_EPOCHS=(10,20)
NEW_MATCHES=512

def verify_data(original,new,dev):
    rows_a,vocab,engine=read_events(original,*TRAIN_BLOCKS[0])
    source=rows_a[0][1]["trace_source_sha"]
    rows_b,vocab_b,engine_b=read_events(
        new,*TRAIN_BLOCKS[1],expected_sha=source)
    dev_rows,vocab_d,engine_d=read_events(dev,*DEV,expected_sha=source)
    if not (vocab==vocab_b==vocab_d and engine==engine_b==engine_d):
        raise ValueError("Inconsistent source, engine, rich vocabulary or phase data")
    sets=[set(seed for seed,_ in rows) for rows in (rows_a,rows_b,dev_rows)]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Training/development Match seeds overlap")
    return rows_a+rows_b,dev_rows,vocab,engine,source


def load_old(directory,report,vocab,kind):
    if (report.get("schema")!="rnad-i5-selection-AB-exploratory-v1" or
        report.get("train_complete_matches")!=256 or
        report.get("development_matches")!=128 or
        report.get("development_block_previously_used") is not True or
        report.get("vocab")!=vocab):
        raise ValueError("Not the pinned historical A/B 256-Match research")
    row=report["models"][kind]
    name=f"selection-{kind}.pt"
    if row["weights"]!=name:
        raise ValueError("Unexpected historical model path")
    raw=(Path(directory)/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row["sha256"]:
        raise ValueError("Untrusted historical A/B model bytes")
    torch.manual_seed(90251)
    cls={"A":DecomposedTeacher,"B":DetailedAttention}[kind]
    net=cls(vocab)
    net.load_state_dict(torch.load(Path(directory)/name,
                                   map_location="cpu",weights_only=True),strict=True)
    return net.eval()


def fit_snapshots(kind,seed,rows,vocab,dev,folder):
    torch.manual_seed(seed)
    cls={"A":DecomposedTeacher,"B":DetailedAttention}[kind]
    model=cls(vocab)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0007,weight_decay=0.0001)
    rng=random.Random(seed)
    losses=[]
    outputs={}
    for epoch in range(1,max(CHECKPOINT_EPOCHS)+1):
        model.train()
        idx=list(range(len(rows)))
        rng.shuffle(idx)
        accum=0.
        for k in range(0,len(idx),BATCH):
            batch=[rows[i] for i in idx[k:k+BATCH]]
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model,batch,kind)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite A/B scaling loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            optimizer.step()
            accum+=float(loss.detach())*len(batch)
        losses.append(accum/len(rows))
        print(json.dumps({"arm":kind,"seed":seed,"epoch":epoch,
                          "train_loss":losses[-1]}),flush=True)
        if epoch in CHECKPOINT_EPOCHS:
            model.eval()
            metrics,seed_scores=summarize(evaluated_logits(model,dev),dev)
            name=f"selection-{kind}-512-e{epoch}-s{seed}.pt"
            path=folder/name
            torch.save(model.state_dict(),path)
            outputs[name]={
                "arm":kind,"train_matches":NEW_MATCHES,"epochs":epoch,"seed":seed,
                "model_filename":name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "train_losses_to_checkpoint":list(losses),
                "reused_development_metrics":metrics,
            }
            print(json.dumps({"checkpoint":name,"dev":metrics}),flush=True)
    return outputs


def run(train_a,train_b,dev_dir,old_dir,old_report,reference,out):
    torch.set_num_threads(1)
    train_rows,dev,vocab,engine,source=verify_data(train_a,train_b,dev_dir)
    if len(set(seed for seed,_ in train_rows))!=NEW_MATCHES:
        raise ValueError("Incomplete 512 unique accepted complete Match clusters")
    history=json.loads(Path(old_report).read_bytes())
    if history["teacher_source_sha"]!=source or history["engine_sha"]!=engine:
        raise ValueError("Historical A/B teacher source / engine mismatch")
    directory=Path(out)
    directory.mkdir(parents=True,exist_ok=False)
    old_models={}
    for kind in ("A","B"):
        baseline=load_old(old_dir,history,vocab,kind)
        metrics,_=summarize(evaluated_logits(baseline,dev),dev)
        if abs(metrics["top1"]-history["models"][kind]["dev"]["top1"])>1e-8:
            raise ValueError("Historical A/B dev baseline is not reproducible")
        old_models[kind]=metrics
    structured=from_archive(reference,REFERENCE_SHA)
    baseline_structured,_=summarize(baseline_scores(structured,dev),dev)
    if abs(baseline_structured["top1"]-
           history["models"]["structured-hard-frozen"]["dev"]["top1"])>1e-8:
        raise ValueError("Historical structured-hard baseline changed")
    trials={}
    for kind in ("A","B"):
        for seed in SEEDS:
            trials.update(fit_snapshots(kind,seed,train_rows,vocab,dev,directory))
    chosen={}
    for kind in ("A","B"):
        candidates=[v for v in trials.values() if v["arm"]==kind]
        best=min(candidates,key=lambda x:(
            -x["reused_development_metrics"]["top1"],
            x["reused_development_metrics"]["regret"],
            x["epochs"],x["seed"]))
        chosen[kind]=best["model_filename"]
    report={
        "schema":SCHEMA,
        "source_sha":source,"engine_sha":engine,"vocab":vocab,
        "teacher":SOURCE_TEACHER,"train_blocks":[list(x) for x in TRAIN_BLOCKS],
        "train_matches":NEW_MATCHES,"development_block":list(DEV),
        "dev_previously_reused":True,
        "used_previous_fresh_block_as_training":True,
        "study_hypotheses":"joint data-size/optimization-duration effect",
        "seeds":list(SEEDS),"checkpoint_epochs":list(CHECKPOINT_EPOCHS),
        "historical_models":old_models,
        "historical_structured":baseline_structured,
        "trials":trials,"dev_selected_models":chosen,
        "teacher_targets_training_only":True,"opponent_hidden_data_used":False,
        "fresh_confirmation_run":None,"league_training_started":False,
        "champion_promoted":False,
    }
    (directory/"selection-AB-scaling-development.json").write_text(
        json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    lines=["# A/B 512 Match×10/20 epoch×3 seeds 学習結果","",
        "**探索用**：旧256学習Match＋以前の独立確認256Matchを新学習データに転用。",
        "以前の128Match開発データ上での結果であり、検証済みの一般化改善とは扱わない。",
        "","| 方式 | epochs | seed | Top1 | Top10 | 3体一致 | regret↓ |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for kind,old in old_models.items():
        lines.append(f"| {kind}旧256Match | 10 | 90251 | {old['top1']:.2%} | "
                     f"{old['top10']:.2%} | {old['trio']:.2%} | {old['regret']:.4f} |")
    for name,row in trials.items():
        m=row["reused_development_metrics"]
        lines.append(f"| {row['arm']} | {row['epochs']} | {row['seed']} | "
                     f"{m['top1']:.2%} | {m['top10']:.2%} | {m['trio']:.2%} | {m['regret']:.4f} |")
    for kind in ("A","B"):
        x=trials[chosen[kind]]
        lines.append(f"開発データで事前固定した{kind}候補：{x['model_filename']} "
                     f"({x['reused_development_metrics']['top1']:.2%})")
    lines.extend(["","未使用Matchで2方式それぞれ独立seed対応あり97.5%両側CIを使用するまで判断保留。"
                  " リーグ/Champion/実戦勝率とは別。"])
    (directory/"SELECTION_AB_SCALING_DEV.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    for arg in ("train-a","train-b","dev-dir","old-dir","old-report","reference","out"):
        parser.add_argument("--"+arg,required=True)
    a=parser.parse_args()
    run(a.train_a,a.train_b,a.dev_dir,a.old_dir,a.old_report,a.reference,a.out)


if __name__=="__main__":
    main()
