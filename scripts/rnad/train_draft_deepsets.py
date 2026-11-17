"""Bounded, research-only draft teacher distillation with a team-set encoder.

Four controlled arms vary additive-vs-DeepSets architecture, hard-vs-soft
utility ranking, and 4-vs-16 accepted teacher Match volume. All observations
come from seat-local --draft-review authority traces, never hidden opponent
sets. This model is NOT a deployed R-NaD actor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from scripts.rnad.train import current_git_sha, frozen_engine_sha

TYPES = ("normal","fire","water","electric","grass","ice","fighting",
         "poison","ground","flying","psychic","bug","rock","ghost",
         "dragon","dark","steel","fairy")
ROLES = ("attack","setup","recovery","teamRecovery","hazard","hazardRemoval",
         "phazing","yawn","pivot","screen","batonPass","trickRoom","tailwind",
         "statusControl","weatherStart","terrainStart","priorityAttack")
STATS = ("hp","atk","def","spa","spd","spe")
FEATURE_WIDTH = len(TYPES)+len(ROLES)+len(STATS)+3

def mon_features(p):
    if not isinstance(p,dict) or not isinstance(p.get("base_stats"),dict):
        raise ValueError("Missing explicit known own-set stats; old trace version forbidden")
    stats=p["base_stats"]
    if any(not isinstance(stats.get(k),(int,float)) or
           not math.isfinite(stats[k]) or stats[k]<0 for k in STATS):
        raise ValueError("Bad public/own stat")
    types=[str(t).lower() for t in p.get("types",[])]
    roles=p.get("roles",[])
    if (any(t not in TYPES for t in types) or
        any(t not in ROLES for t in roles) or
        not isinstance(p.get("credit_cost"),int) or p["credit_cost"]<0):
        raise ValueError("Invalid known own-set capabilities or price")
    return ([min(1.,stats[k]/600.) for k in STATS] +
            [float(t in types) for t in TYPES] +
            [float(t in roles) for t in ROLES] +
            [float(bool(p.get("mega"))),min(1.,p["credit_cost"]/100.),
             float(bool(p.get("is_carry")))])

def load_events(directory,seed,matches):
    output=[];source=current_git_sha();engine=frozen_engine_sha()
    for i in range(matches):
        trace=json.loads((Path(directory)/f"teacher-{seed+i}.json").read_text())
        if (trace.get("source_sha")!=source or trace.get("engine_sha")!=engine
            or trace.get("feature_schema")!="rnad-three-phase-features-v4"
            or trace.get("teacher")!="rnad-aq-public-strategy-selection-v1"
            or trace.get("completed") is not True or
            trace["score"][trace["winner"]]!=4):
            raise ValueError("Untrusted or incomplete accepted teacher data")
        for event in trace["examples"]:
            if event["phase"]!="draft":
                continue
            v=event.get("draft_review")
            if not v or v.get("schema")!="rnad-accepted-draft-pick-review-v1":
                raise ValueError("Missing seat-local named draft review")
            codes=v["candidate_codes"]
            util=event.get("candidate_utilities")
            choice=event["choice"]
            if (not codes or len(codes)!=len(util) or
                choice!=v["accepted_choice_index"] or
                choice<0 or choice>=len(codes) or
                any(len(c)!=6 or len(set(c))!=6 or
                    any(type(k)!=int or not 0<=k<len(v["roster"]) for k in c)
                    for c in codes) or
                any(not math.isfinite(u) for u in util) or
                max(util)-util[choice]>1e-5):
                raise ValueError("Teacher action/utility candidate mismatch")
            x=torch.tensor([mon_features(p) for p in v["roster"]],
                           dtype=torch.float32)
            codes=torch.tensor(codes,dtype=torch.long)
            utility=torch.tensor(util,dtype=torch.float32)
            prices=torch.tensor([p["credit_cost"] for p in v["roster"]])
            carry=torch.tensor([float(p["is_carry"]) for p in v["roster"]])
            mega=torch.tensor([float(p["mega"]) for p in v["roster"]])
            spent=prices[codes].sum(1).float()
            if not bool((spent<=v["credits"]).all()):
                raise ValueError("Non-legal budget candidate in accepted trace")
            global_f=torch.stack([
                spent/100.,
                (v["credits"]-spent)/100.,
                carry[codes].sum(1)/6.,
                mega[codes].sum(1)/6.,
            ],dim=1)
            output.append({
                "pool":x,"codes":codes,"global":global_f,
                "label":choice,"utilities":utility,"view":v,
                "seed":trace["seed"],"seat":event["actor"],
            })
    if not output:
        raise ValueError("Empty accepted draft decisions")
    return output

class AdditiveDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Linear(FEATURE_WIDTH+4,1)
    def forward(self,pool,codes,extra):
        team=pool[codes].mean(1)
        return self.net(torch.cat((team,extra),dim=1)).squeeze(-1)

class DeepSetsDraft(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        self.phi=nn.Sequential(nn.Linear(FEATURE_WIDTH,width),
                               nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        # Mean, maximum and second moment are all permutation-invariant.
        self.rho=nn.Sequential(nn.Linear(width*3+4,64),nn.ReLU(),
                               nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,pool,codes,extra):
        # A roster mon can occur in thousands of legal six-mon teams.
        # Compute its embedding ONCE, then gather.  The same embedding
        # receives summed gradients from all teams containing the mon.
        # This is algebraically identical to phi(pool[codes]) with shared
        # parameters, and scales far better for large accepted teacher data.
        m=self.phi(pool)[codes]
        avg=m.mean(1)
        second=(m*m).mean(1)
        highest=m.amax(1)
        return self.rho(torch.cat((avg,second,highest,extra),dim=1)).squeeze(-1)

def teacher_soft_loss(logits,util):
    top=torch.argsort(util,descending=True,stable=True)[:min(12,util.numel())]
    scores=util[top]
    temperature=torch.clamp(util.std(unbiased=False),min=1.0)
    targets=torch.softmax((scores-scores.max())/temperature,dim=0)
    logp=F.log_softmax(logits,dim=0)
    return -(targets*logp[top]).sum()

def train_model(arm,events,epochs,seed):
    torch.manual_seed(seed)
    model=AdditiveDraft() if arm.startswith("additive") else DeepSetsDraft()
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    randomizer=random.Random(seed)
    losses=[]
    for epoch in range(epochs):
        indexes=list(range(len(events)))
        randomizer.shuffle(indexes)
        total=0.
        for index in indexes:
            e=events[index]
            logits=model(e["pool"],e["codes"],e["global"])
            if arm.endswith("soft"):
                # Utility ranking, not a single arbitrary top-1 label.
                loss=teacher_soft_loss(logits,e["utilities"])
            else:
                loss=F.cross_entropy(logits.unsqueeze(0),
                                     torch.tensor([e["label"]]))
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite draft teacher loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,
                                           error_if_nonfinite=True)
            opt.step()
            total+=float(loss.detach())
        losses.append(total/len(events))
    return model,losses

def eval_model(model,events,examples=10):
    model.eval()
    totals={"top1":0,"top10":0,"normalized_ce":0.,"normalized_regret":0.}
    rows=[]
    with torch.no_grad():
        for e in events:
            logits=model(e["pool"],e["codes"],e["global"])
            ordered=torch.argsort(logits,descending=True,stable=True)
            choice=int(ordered[0]);teacher=e["label"]
            total=len(ordered)
            ce=-float(F.log_softmax(logits,dim=0)[teacher])
            util=e["utilities"]
            gap=float((util.max()-util.min()).clamp(min=1e-9))
            regret=float((util.max()-util[choice]))/gap
            totals["top1"]+=int(choice==teacher)
            totals["top10"]+=int(teacher in ordered[:10].tolist())
            totals["normalized_ce"]+=ce/math.log(total) if total>1 else 0.
            totals["normalized_regret"]+=regret
            if len(rows)<examples:
                roster=e["view"]["roster"]
                def name(k):
                    return [roster[int(i)]["species"] for i in
                            e["codes"][k].tolist()]
                rows.append({
                    "seed":e["seed"],"seat":e["seat"],
                    "legal_candidates":total,"credits":e["view"]["credits"],
                    "teacher_pick":name(teacher),"student_pick":name(choice),
                    "teacher_utility":float(util[teacher]),
                    "student_utility":float(util[choice]),
                    "student_teacher_utility_regret":regret,
                    "student_matches_teacher":choice==teacher,
                    "top5_student": [
                        {"pokemon":name(int(k)),"teacher_utility":float(util[k])}
                        for k in ordered[:5]]
                })
    return {**{k:v/len(events) for k,v in totals.items()},
            "heldout_decisions":len(events),"examples":rows}

def save_report(report,path):
    if not path.is_dir():
        raise ValueError("Missing isolated study output directory")
    (path/"draft-study.json").write_text(json.dumps(
        report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    lines=["# R-NaD ドラフト専用Deep Sets・順位蒸留 実験",
           "",
           "教師が訪れた公開・自席局面での模倣指標です。"
           "実戦Match勝率・Champion強度を証明しません。",
           "",
           f"- Source: `{report['source_sha']}`",
           f"- Train / holdout: {report['train_matches']} / {report['heldout_matches']} 完全Match",
           f"- Dataset seed: {report['train_seed']} / {report['heldout_seed']}",
           "",
           "| 条件 | 学習決定数 | Top1 | Top10 | 正規化CE ↓ | utility regret ↓ |",
           "|---|---:|---:|---:|---:|---:|"]
    for arm,x in report["arms"].items():
        row=x["holdout"]
        lines.append(f"| {arm} | {x['train_decisions']} | {row['top1']:.2%} | "
                     f"{row['top10']:.2%} | {row['normalized_ce']:.3f} | "
                     f"{row['normalized_regret']:.3f} |")
    lines.append("")
    for arm,x in report["arms"].items():
        lines.extend([f"## {arm}：未使用局面の具体的ピック",""])
        for e in x["holdout"]["examples"][:5]:
            lines.extend([
                f"### seed {e['seed']} 席{e['seat']} / {e['legal_candidates']}候補",
                "- 教師が受理: "+", ".join(e["teacher_pick"]),
                "- 学生greedy予測（未受理）: "+", ".join(e["student_pick"]),
                f"- 一致 {e['student_matches_teacher']} / utility regret {e['student_teacher_utility_regret']:.3f}",
                ""])
    (path/"DRAFT_STUDY.md").write_text("\n".join(lines)+"\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-dir",required=True)
    p.add_argument("--train-seed",type=int,required=True)
    p.add_argument("--train-matches",type=int,required=True)
    p.add_argument("--holdout-dir",required=True)
    p.add_argument("--holdout-seed",type=int,required=True)
    p.add_argument("--holdout-matches",type=int,required=True)
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    if (not 4<=args.train_matches<=64 or
        not 1<=args.holdout_matches<=16 or
        not 1<=args.epochs<=4 or
        args.train_seed<=0 or args.holdout_seed<=0 or
        args.train_seed<=args.holdout_seed+args.holdout_matches-1 and
        args.holdout_seed<=args.train_seed+args.train_matches-1):
        raise ValueError("Invalid bounded or overlapping source seed blocks")
    torch.set_num_threads(1)
    train=load_events(args.train_dir,args.train_seed,args.train_matches)
    holdout=load_events(args.holdout_dir,args.holdout_seed,args.holdout_matches)
    arms={
        "additive-hard-full":("additive-hard",len(train)),
        "deepsets-hard-full":("deepsets-hard",len(train)),
        "deepsets-soft-small":("deepsets-soft",len([
            e for e in train if e["seed"]<args.train_seed+4])),
        "deepsets-soft-full":("deepsets-soft",len(train)),
    }
    if args.train_matches>16:
        arms["deepsets-soft-medium"]=("deepsets-soft",len([
            e for e in train if e["seed"]<args.train_seed+16]))
    results={}
    outdir=Path(args.out)
    outdir.mkdir(parents=True,exist_ok=False)
    for name,(kind,count) in arms.items():
        model,losses=train_model(kind,train[:count],args.epochs,seed=23873)
        export={"schema":"rnad-draft-deepsets-weights-v1",
                "architecture":kind,"feature_width":FEATURE_WIDTH,
                "source_sha":current_git_sha(),
                "weights":{k:{"shape":list(v.shape),
                              "values":v.detach().flatten().tolist()}
                           for k,v in model.state_dict().items()}}
        raw=json.dumps(export,separators=(",",":"),allow_nan=False).encode()
        (outdir/(name+".json")).write_bytes(raw)
        results[name]={"train_decisions":count,"losses":losses,
                       "weights_sha256":hashlib.sha256(raw).hexdigest(),
                       "holdout":eval_model(model,holdout)}
        print(json.dumps({"arm":name,"training":count,
             "holdout":{k:v for k,v in results[name]["holdout"].items() if k!="examples"}},
             ensure_ascii=False),flush=True)
    report={"schema":"rnad-draft-deepsets-utility-study-v1",
            "source_sha":current_git_sha(),"engine_sha":frozen_engine_sha(),
            "train_seed":args.train_seed,"train_matches":args.train_matches,
            "heldout_seed":args.holdout_seed,
            "heldout_matches":args.holdout_matches,
            "epochs":args.epochs,"is_champion":False,
            "strength_established":False,"student_picks_are_counterfactual":True,
            "arms":results}
    save_report(report,outdir)
if __name__=="__main__":
    main()
