"""Research-only 128-match Deep Sets draft imitation with a frozen 32-match control.

Never modifies frozen v3/v4, old Deep Sets model bytes, Champion, or the
authority Match. One shared 64-match held-out public-AQ teacher dataset is
evaluated ONCE after all training. Its outcomes are not game win rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

import torch
from torch.nn import functional as F
from scripts.rnad.train import current_git_sha, frozen_engine_sha
from scripts.rnad.train_draft_deepsets import (
    DeepSetsDraft, FEATURE_WIDTH, eval_model, load_events, teacher_soft_loss,
)

SCHEMA="rnad-i5-draft-imitation-128-v1"
BASELINE_SHA="da186af7581485382b4f558f356ac285bbf071817239cba7271cb05615e3b037"
BASELINE_SOURCE_RE=re.compile(r"^[0-9a-f]{40}$")
TRAINS={"soft128": "soft", "sharp128": "sharp"}
SEED=48612
TRAIN_MATCHES=128
HOLDOUT_MATCHES=64

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def restore_old_32match_model(path: str):
    blob=Path(path).read_bytes()
    if sha256(blob)!=BASELINE_SHA:
        raise ValueError("Frozen 32-teacher-match baseline model SHA mismatch")
    data=json.loads(blob)
    if (data.get("schema")!="rnad-draft-deepsets-weights-v1"
        or data.get("architecture")!="deepsets-soft"
        or data.get("feature_width")!=FEATURE_WIDTH
        or not BASELINE_SOURCE_RE.fullmatch(data.get("source_sha",""))):
        raise ValueError("Unrecognized frozen Deep Sets baseline schema")
    model=DeepSetsDraft()
    weights=data.get("weights")
    if not isinstance(weights,dict) or set(weights)!=set(model.state_dict()):
        raise ValueError("Frozen baseline tensor keys disagree")
    restored={}
    for name,old in model.state_dict().items():
        t=weights[name]
        if (t.get("shape")!=list(old.shape)
            or not isinstance(t.get("values"),list)
            or len(t["values"])!=old.numel()
            or any(type(v) not in (float,int) or not math.isfinite(v)
                   for v in t["values"])):
            raise ValueError("Bad frozen tensor "+name)
        restored[name]=torch.tensor(t["values"],dtype=old.dtype).reshape(old.shape)
    model.load_state_dict(restored,strict=True)
    model.eval()
    return model

def sharp_teacher_loss(logits, util, label):
    """Blend exact top1 (teacher tie-break) and sharply ranked teacher top16.

    Previous top12 distribution could put too little probability on the
    accepted best choice when many near-tied candidates exist.
    """
    if not (0<=label<util.numel() and logits.shape==util.shape):
        raise ValueError("Invalid teacher label/action support")
    top=torch.argsort(util,descending=True,stable=True)[:min(16,util.numel())]
    temp=torch.clamp(util.std(unbiased=False)*.25,min=.25)
    selected=util[top]
    target=F.softmax((selected-selected.max())/temp,dim=0)
    logp=F.log_softmax(logits,dim=0)
    return .5*(-logp[label])+.5*(-torch.sum(target*logp[top]))

def train_variant(events,kind,epochs=4):
    if kind not in ("soft","sharp") or epochs not in (4,6):
        raise ValueError("Unsupported predeclared experimental training arm")
    torch.manual_seed(SEED)
    model=DeepSetsDraft()
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    rng=random.Random(SEED)
    losses=[]
    for epoch in range(epochs):
        indices=list(range(len(events)))
        rng.shuffle(indices)
        accumulated=0.0
        for ix in indices:
            event=events[ix]
            logits=model(event["pool"],event["codes"],event["global"])
            if kind=="soft":
                loss=teacher_soft_loss(logits,event["utilities"])
            else:
                loss=sharp_teacher_loss(logits,event["utilities"],event["label"])
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite teacher objective")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,
                error_if_nonfinite=True)
            opt.step()
            accumulated+=float(loss.detach())
        losses.append(accumulated/len(events))
    model.eval()
    return model,losses

def save_model(model,outpath,objective,train_seed):
    data={
      "schema":"rnad-draft-deepsets-weights-v1",
      "architecture":"deepsets-soft",
      "feature_width":FEATURE_WIDTH,
      "source_sha":current_git_sha(),
      "training_details":{
          "experiment":SCHEMA,"objective":objective,
          "teacher_matches":TRAIN_MATCHES,"epochs":4,
          "train_seed":train_seed,"training_only":True,
      },
      "weights":{
          k:{"shape":list(t.shape),"values":t.detach().flatten().tolist()}
          for k,t in model.state_dict().items()
      },
    }
    raw=json.dumps(data,separators=(",",":"),allow_nan=False).encode()
    outpath.write_bytes(raw)
    return sha256(raw)

def independent_seed_top1(model, events):
    counts={}
    with torch.no_grad():
        for event in events:
            logits=model(event["pool"],event["codes"],event["global"])
            pick=int(torch.argmax(logits).item())
            top=torch.topk(logits,min(10,len(logits))).indices
            seed=event["seed"]
            row=counts.setdefault(seed,{"n":0,"top1":0,"top10":0,"regret":0.})
            util=event["utilities"]
            gap=float((util.max()-util.min()).clamp(min=1e-9))
            row["n"]+=1
            row["top1"]+=int(pick==event["label"])
            row["top10"]+=int(bool((top==event["label"]).any()))
            row["regret"]+=float(util.max()-util[pick])/gap
    for x in counts.values():
        for key in ("top1","top10","regret"):
            x[key]/=x["n"]
    return counts

def cluster_bootstrap80(baseline,candidate,seed_ids,replicates=8000):
    """Exploratory percentile bootstrap with 2-family Bonferroni bounds.

    Two candidate comparisons are each shown with a nominal 90% two-sided
    interval; simultaneous coverage is approximately >=80%, NOT an exact
    finite-sample strength guarantee. Original match/seed is the cluster.
    """
    if len(seed_ids)!=HOLDOUT_MATCHES or replicates<1000:
        raise ValueError("Insufficient independent held-out teacher Matches")
    rand=random.Random(830871)
    d={key:[candidate[s][key]-baseline[s][key] for s in seed_ids]
       for key in ("top1","top10","regret")}
    report={}
    for key,values in d.items():
        n=len(values)
        samples=[]
        for _ in range(replicates):
            samples.append(sum(values[rand.randrange(n)] for _ in range(n))/n)
        samples.sort()
        observed=sum(values)/n
        report[key]={
          "difference":observed,
          "familywise_80pct_approximate_interval":
              [samples[int(replicates*.05)],samples[int(replicates*.95)]],
        }
    return report

def main(argv=None):
    p=argparse.ArgumentParser()
    for name in ("train-dir","holdout-dir","frozen-baseline","out"):
        p.add_argument("--"+name,required=True)
    for name in ("train-seed","holdout-seed"):
        p.add_argument("--"+name,type=int,required=True)
    args=p.parse_args(argv)
    if (args.train_seed!=2461001 or args.holdout_seed!=2462001
        or args.train_seed+TRAIN_MATCHES>=args.holdout_seed
        or args.holdout_seed+HOLDOUT_MATCHES>0xffffffff):
        raise ValueError("Unregistered or overlapping teacher/holdout seeds")
    torch.set_num_threads(1)
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=False)
    train=load_events(args.train_dir,args.train_seed,TRAIN_MATCHES)
    held=load_events(args.holdout_dir,args.holdout_seed,HOLDOUT_MATCHES)
    if ({e["seed"] for e in train} & {e["seed"] for e in held}
        or len({e["seed"] for e in held})!=HOLDOUT_MATCHES):
        raise ValueError("Held-out teacher seed reuse")
    old=restore_old_32match_model(args.frozen_baseline)
    old_metrics=eval_model(old,held)
    old_clusters=independent_seed_top1(old,held)
    expected_seeds=list(range(args.holdout_seed,
                              args.holdout_seed+HOLDOUT_MATCHES))
    if sorted(old_clusters)!=expected_seeds:
        raise ValueError("Incomplete heldout teacher Match clusters")
    results={
      "baseline_32_frozen":{
          "matches_trained":32,
          "weight_sha256":BASELINE_SHA,
          "holdout":old_metrics,
      }
    }
    for name,kind in TRAINS.items():
        model,losses=train_variant(train,kind)
        digest=save_model(model,output/(name+".json"),kind,args.train_seed)
        metric=eval_model(model,held)
        by_seed=independent_seed_top1(model,held)
        if sorted(by_seed)!=expected_seeds:
            raise ValueError("Missing independent teacher-heldout Match")
        results[name]={
          "matches_trained":TRAIN_MATCHES,
          "weight_sha256":digest,
          "train_losses":losses,
          "holdout":metric,
          "paired_seed_diagnostic":cluster_bootstrap80(
              old_clusters,by_seed,expected_seeds),
        }
        print(json.dumps({
            "arm":name,"train_matches":TRAIN_MATCHES,
            "heldout_teacher_matches":HOLDOUT_MATCHES,
            "heldout_decisions":metric["heldout_decisions"],
            "top1":metric["top1"],"top10":metric["top10"],
            "normalized_regret":metric["normalized_regret"],
        },ensure_ascii=False),flush=True)
    report={
      "schema":SCHEMA,
      "source_sha":current_git_sha(),
      "engine_sha":frozen_engine_sha(),
      "train_seed":args.train_seed,"train_matches":TRAIN_MATCHES,
      "holdout_seed":args.holdout_seed,"holdout_matches":HOLDOUT_MATCHES,
      "heldout_teacher_decisions":len(held),
      "simultaneous_confidence_level":.80,
      "ci_method":"2-candidate Bonferroni 90pct cluster-percentile bootstrap, approximate",
      "independent_unit":"teacher Match seed, not individual draft decision",
      "models":results,
      "strength_established":False,
      "champion_promoted":False,
      "evaluation_type":"teacher-visited counterfactual imitation only",
      "chosen_for_champion":None,
      "frozen_32match_baseline_sha256":BASELINE_SHA,
    }
    (output/"imitation-report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=[
        "# R-NaD ドラフト模倣：凍結32教師Match版 vs 128教師Matchの2方式",
        "",
        "実対戦の勝率ではなく、未使用の公開AQ教師の64完全Matchで",
        "同じドラフト局面を入力した際の予測一致率。",
        "64独立教師Matchに含まれる複数ドラフト判断を独立試合として数えない。",
        "",
        "| 固定条件 | 学習完全Match | 教師Top1 | Top10 | 正規化utility regret↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name,rec in results.items():
        m=rec["holdout"]
        lines.append(
            f"| {name} | {rec['matches_trained']} | {m['top1']:.2%} | "
            f"{m['top10']:.2%} | {m['normalized_regret']:.4f} |"
        )
    lines.extend([
      "",
      "80%は将来新規評価のユーザー指定。",
      "ここでの2比較補正済みseed-cluster bootstrap区間は近似的かつ探索的で、",
      "モデルの実戦優位、Champion採用、模倣の独立確証を意味しない。",
      "比較対象は学習済み旧32Match版の凍結重みそのもの。",
      "教師と同一の未使用64Matchに対して3モデルを一度だけ評価。",
      "",
    ])
    for name in TRAINS:
        t=results[name]["paired_seed_diagnostic"]["top1"]
        l,u=t["familywise_80pct_approximate_interval"]
        lines.append(
          f"- {name} − frozen32 Top1: {t['difference']:+.2%} "
          f"(approx simultaneous 80% CI: [{l:+.2%}, {u:+.2%}])"
        )
    (output/"IMITATION_REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({
      "schema":SCHEMA,
      "holdout_matches":HOLDOUT_MATCHES,
      "heldout_decisions":len(held),
      "strength_established":False,
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
