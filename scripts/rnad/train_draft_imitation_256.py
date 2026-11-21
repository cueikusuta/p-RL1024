"""Next 256 AQ teacher Match DeepSets study, with frozen128 control.

Single explicitly reserved 128-Match teacher holdout; decisions within a
Match do not count as independent statistical clusters. Research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import torch

from scripts.rnad.train_draft_deepsets import (
    DeepSetsDraft, FEATURE_WIDTH, eval_model, load_events)
from scripts.rnad.train_draft_imitation_128 import (
    train_variant, independent_seed_top1)
from scripts.rnad.train import current_git_sha, frozen_engine_sha

BASELINE_SHA = "4c50df5de8001988fa077283f0a2fa0ca964992d5e1782c4b279120e01b39825"
TRAIN_SEED = 2471001
HELDOUT_SEED = 2472001
TRAIN_MATCHES = 256
HELDOUT_MATCHES = 128
SCHEMA = "rnad-i5-draft-256-teacher-heldout-v1"
ARMS = {"soft256-e4": 4, "soft256-e6": 6}

def restore_frozen128(path):
    b=Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=BASELINE_SHA:
        raise ValueError("Frozen128 draft weight SHA mismatch")
    o=json.loads(b)
    if (o.get("schema")!="rnad-draft-deepsets-weights-v1" or
        o.get("architecture")!="deepsets-soft" or
        o.get("feature_width")!=FEATURE_WIDTH):
        raise ValueError("Invalid frozen DeepSets schema")
    model=DeepSetsDraft()
    expected=model.state_dict()
    if set(o.get("weights",{}))!=set(expected):
        raise ValueError("Missing frozen draft tensors")
    restored={}
    for k,target in expected.items():
        t=o["weights"][k]
        if (t.get("shape")!=list(target.shape) or
            len(t.get("values",[]))!=target.numel() or
            any(type(x) not in (int,float) or not math.isfinite(x)
                for x in t["values"])):
            raise ValueError("Bad frozen draft tensor: "+k)
        restored[k]=torch.tensor(t["values"],dtype=target.dtype).reshape(target.shape)
    model.load_state_dict(restored,strict=True)
    return model.eval()

def save(model,path,epochs):
    payload={
       "schema":"rnad-draft-deepsets-weights-v1",
       "architecture":"deepsets-soft","feature_width":FEATURE_WIDTH,
       "source_sha":current_git_sha(),
       "training_details":{
           "experiment":SCHEMA,"teacher_matches":TRAIN_MATCHES,
           "epochs":epochs,"teacher_seed":TRAIN_SEED,
           "training_only":True,
       },
       "weights":{k:{"shape":list(v.shape),
                     "values":v.detach().flatten().tolist()}
                  for k,v in model.state_dict().items()},
    }
    b=json.dumps(payload,separators=(",",":"),allow_nan=False).encode()
    path.write_bytes(b)
    return hashlib.sha256(b).hexdigest()

def cluster_difference(base,arm,seed_ids,*,replicates=6000):
    data=[arm[s]["top1"]-base[s]["top1"] for s in seed_ids]
    mean=sum(data)/len(data)
    rng=random.Random(247010)
    draws=sorted(sum(data[rng.randrange(len(data))] for _ in data)/len(data)
                 for _ in range(replicates))
    # 2 pre-declared comparisons, Bonferroni (90% each => >=80% approx).
    return {"mean_diff_top1":mean,
       "approximate_familywise_80pct_seed_cluster_CI":
           [draws[int(.05*replicates)],draws[int(.95*replicates)]],
       "bootstrap_draws":replicates,
       "exact_finite_sample_CI":False}

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--train-dir",required=True)
    p.add_argument("--heldout-dir",required=True)
    p.add_argument("--baseline",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args(argv)
    torch.set_num_threads(1)
    train=load_events(args.train_dir,TRAIN_SEED,TRAIN_MATCHES)
    held=load_events(args.heldout_dir,HELDOUT_SEED,HELDOUT_MATCHES)
    if (len(set(e["seed"] for e in train))!=TRAIN_MATCHES or
        len(set(e["seed"] for e in held))!=HELDOUT_MATCHES or
        set(e["seed"] for e in train)&set(e["seed"] for e in held)):
        raise ValueError("Incomplete or overlapping complete teacher Match seeds")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    base=restore_frozen128(args.baseline)
    base_metric=eval_model(base,held,examples=0)
    base_per_seed=independent_seed_top1(base,held)
    all_seeds=list(range(HELDOUT_SEED,HELDOUT_SEED+HELDOUT_MATCHES))
    if sorted(base_per_seed)!=all_seeds:
        raise ValueError("Incomplete frozen baseline seed evidence")
    models={"frozen-soft128":{"top1":base_metric["top1"],
              "top10":base_metric["top10"],
              "normalized_ce":base_metric["normalized_ce"],
              "normalized_regret":base_metric["normalized_regret"],
              "sha256":BASELINE_SHA}}
    for name,epochs in ARMS.items():
        net,losses=train_variant(train,"soft",epochs=epochs)
        digest=save(net,out/(name+".json"),epochs)
        values=eval_model(net,held,examples=0)
        per_seed=independent_seed_top1(net,held)
        if sorted(per_seed)!=all_seeds:
            raise ValueError("Incomplete independent seed clusters")
        models[name]={
            "top1":values["top1"],"top10":values["top10"],
            "normalized_ce":values["normalized_ce"],
            "normalized_regret":values["normalized_regret"],
            "sha256":digest,"losses":losses,
            "paired_vs_frozen128":cluster_difference(
                base_per_seed,per_seed,all_seeds)
        }
        print(json.dumps({"arm":name,
          "top1":values["top1"],"top10":values["top10"]}),flush=True)
    report={
        "schema":SCHEMA,"source_sha":current_git_sha(),
        "engine_sha":frozen_engine_sha(),"teacher_id":
            "rnad-aq-public-strategy-selection-v1",
        "train_seed":TRAIN_SEED,"train_matches":TRAIN_MATCHES,
        "heldout_seed":HELDOUT_SEED,"heldout_matches":HELDOUT_MATCHES,
        "heldout_decisions":len(held),
        "independent_unit":"complete_teacher_Match_seed",
        "familywise_confidence_level":.80,
        "tested_arms":list(ARMS),
        "heldout_used_for_hyperparameter_selection":True,
        "unseen_future_league_gate_required":True,
        "teacher_visited_counterfactual_only":True,
        "strength_established":False,"champion_promoted":False,
        "models":models,
    }
    (out/"draft-imitation-256-report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n")
    lines=["# ドラフト模倣学習256Match：未使用128教師Match評価",
           "","Top1/Top10は教師が訪れた局面の一致率。実戦勝率ではありません。",
           "","| モデル | 教師Top1 | Top10 | normalized CE ↓ | utility regret ↓ |",
           "|---|---:|---:|---:|---:|"]
    for name,m in models.items():
        lines.append(f"| {name} | {m['top1']:.2%} | {m['top10']:.2%} "
                     f"| {m['normalized_ce']:.4f} | {m['normalized_regret']:.4f} |")
    (out/"DRAFT_256.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"heldout_Matches":len(all_seeds),
         "heldout_decisions":len(held),"strength_established":False}),flush=True)

if __name__=="__main__":
    main()
