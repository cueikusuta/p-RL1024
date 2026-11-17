"""I5 isolated selection imitation: 6 own -> unordered trio + ordered lead.

Uses immutable accepted AQ 256 train / 128 *development* heldout traces from
run 44032488678. NO game victories, league promotion or final confirmation.
Candidate ordering matches legalPureRlSelections from pure-rl-phases.js and
is guarded by a Node contract test; three leads per unordered trio.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from scripts.rnad.neural import (
    RNADNetwork, load_exported_json,
    THREE_PHASE_FEATURE_SCHEMA_V5,
    STATE_FEATURES_V5, ACTION_FEATURES_V5)
from scripts.rnad.evaluate_aq_imitation import per_decision_metrics
from scripts.rnad.train_teacher import AQ_WEIGHTS_SHA256
from scripts.rnad.train import current_git_sha, frozen_engine_sha

SOURCE_SHA = "022aa9cd5b4ac37514df1d845ea00c11c3c13a5c"
TEACHER = "rnad-aq-public-strategy-selection-v1"
TEACHER_SCHEMA = "rnad-aq-public-strategy-selection-three-phase-teacher-v1"
SEED_TRAIN, MATCHES_TRAIN = 2471001,256
SEED_HOLDOUT, MATCHES_HOLDOUT = 2472001,128
ARMS = ("flat-hard", "structured-hard", "flat-mixed", "structured-mixed")
EPOCHS, BATCH = 6, 24
TRIPLES = [(i,j,k) for i in range(6) for j in range(6)
           for k in range(j+1,6) if i!=j and i!=k]
assert len(TRIPLES)==60
KEYS = sorted({tuple(sorted(t)) for t in TRIPLES})
assert len(KEYS)==20
TRIO_MAP = {key:i for i,key in enumerate(KEYS)}
TRIO_IDS = [TRIO_MAP[tuple(sorted(t))] for t in TRIPLES]
LEAD_IDS = [t[0] for t in TRIPLES]
TRIO_ROWS = [[idx for idx,t in enumerate(TRIPLES)
              if tuple(sorted(t))==key] for key in KEYS]
LEAD_ROWS = [[i for i,t in enumerate(TRIPLES) if t[0]==lead]
             for lead in range(6)]
assert all(len(x)==3 for x in TRIO_ROWS)
assert all(len(x)==10 for x in LEAD_ROWS)


def load_traces(folder, start, count):
    root=Path(folder)
    traces=[]
    engine=None
    for seed in range(start,start+count):
        file=root/f"teacher-{seed}.json"
        t=json.loads(file.read_bytes())
        if (t.get("seed")!=seed or
            t.get("schema")!=TEACHER_SCHEMA or
            t.get("source_sha")!=SOURCE_SHA or
            t.get("feature_schema")!=THREE_PHASE_FEATURE_SCHEMA_V5 or
            t.get("teacher")!=TEACHER or
            t.get("teacher_model_sha256")!=AQ_WEIGHTS_SHA256 or
            t.get("utility_labels")!="aq-public-phase-rank-v1" or
            t.get("completed") is not True or
            not isinstance(t.get("winner"),int) or
            t.get("score",[])[t["winner"]]!=4 or
            not isinstance(t.get("examples"),list) or
            len(t["examples"])!=sum(t["action_counts"])+4*t["rounds"]):
            raise ValueError("Invalid or nonaccepted sealed teacher trace "+str(seed))
        if not isinstance(t.get("engine_sha"),str) or len(t["engine_sha"])!=64:
            raise ValueError("Missing sealed engine SHA")
        if engine is not None and t["engine_sha"]!=engine:
            raise ValueError("Mixed engine snapshots")
        engine=t["engine_sha"]
        for e in t["examples"]:
            if e.get("phase")!="selection":continue
            if (len(e.get("state",[]))!=STATE_FEATURES_V5 or
                len(e.get("actions",[]))!=60 or
                any(len(x)!=ACTION_FEATURES_V5 for x in e["actions"]) or
                any(not math.isfinite(v) for v in e["state"]) or
                any(not math.isfinite(v) for a in e["actions"] for v in a) or
                len(e.get("candidate_utilities",[]))!=60 or
                any(not math.isfinite(v) for v in e["candidate_utilities"]) or
                type(e.get("choice")) is not int or
                not 0<=e["choice"]<60 or
                max(e["candidate_utilities"])-e["candidate_utilities"][e["choice"]]>1e-5):
                raise ValueError("Invalid legal/finite 60 candidate selection teacher signal")
        traces.append(t)
    return traces,engine


class SelectionSpecialist(nn.Module):
    """Independent selection-only network, no draft/battle shared gradients.

    Scores are a sum of:
    - unordered trio value (pooled over its three possible leads);
    - lead value (pooled across ten possible reserve pairs);
    - lead x trio x candidate interaction.
    No arbitrary slot ID embedding, hidden opponent set, or privileged feature.
    """
    def __init__(self, structured=True):
        super().__init__()
        self.structured=structured
        dim=96
        self.state=nn.Linear(STATE_FEATURES_V5,dim)
        self.action=nn.Linear(ACTION_FEATURES_V5,dim)
        if structured:
            self.trio=nn.Linear(32,dim)
            self.lead=nn.Linear(32,dim)
            self.trio_head=nn.Linear(dim,1)
            self.lead_head=nn.Linear(dim,1)
        self.interaction=nn.Sequential(nn.Linear(dim,dim),
                                        nn.ReLU(),nn.Linear(dim,1))
        self.register_buffer("trio_ids",torch.tensor(TRIO_IDS,dtype=torch.long))
        self.register_buffer("lead_ids",torch.tensor(LEAD_IDS,dtype=torch.long))
        self.register_buffer("trio_rows",torch.tensor(TRIO_ROWS,dtype=torch.long))
        self.register_buffer("lead_rows",torch.tensor(LEAD_ROWS,dtype=torch.long))

    def forward(self, state, actions):
        if (state.ndim!=2 or state.size(-1)!=STATE_FEATURES_V5 or
            actions.ndim!=3 or actions.size(0)!=state.size(0) or
            actions.shape[1:]!=(60,ACTION_FEATURES_V5)):
            raise ValueError("Selection specialist expects public96D state + 60x48D legal candidates")
        base=F.relu(self.state(state))[:,None,:]
        act=F.relu(self.action(actions))
        if not self.structured:
            return self.interaction(F.relu(base+act)).squeeze(-1)
        # Last 16 features contain public lead-vs-foe matchup, first 32
        # contain team-capability and lead semantics. Trio/lead pooling
        # is tied to actual 60 candidate enumeration, not candidate order guesses.
        trio_vec=actions[:,:,0:32][:,self.trio_rows,:].mean(dim=2)
        lead_vec=actions[:,:,16:48][:,self.lead_rows,:].mean(dim=2)
        trio=F.relu(self.trio(trio_vec))
        lead=F.relu(self.lead(lead_vec))
        tripick=trio[:,self.trio_ids,:]
        leadpick=lead[:,self.lead_ids,:]
        team_value=self.trio_head(F.relu(base+tripick)).squeeze(-1)
        head_value=self.lead_head(F.relu(base+leadpick)).squeeze(-1)
        pair=self.interaction(F.relu(base+tripick+leadpick+act)).squeeze(-1)
        return team_value+head_value+pair


def selection_events(traces):
    events=[]
    for t in traces:
        for e in t["examples"]:
            if e["phase"]=="selection":
                events.append((t["seed"],e))
    return events


def teacher_loss(logits, choice, utility, *, mixed):
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite selection logits")
    hard=F.cross_entropy(logits,choice)
    if not mixed:return hard
    # Teacher utility never becomes an actor input. Full 60-choice soft rank
    # target, no arbitrary top-K boundary. Hard label remains dominant (70%).
    spread=utility.std(dim=1,unbiased=False).clamp(min=1e-5)
    temp=(spread*.25).clamp(min=.25)
    target=F.softmax((utility-utility.max(dim=1,keepdim=True).values)/temp[:,None],dim=1)
    soft=-(target*F.log_softmax(logits,dim=1)).sum(dim=1).mean()
    return .7*hard+.3*soft


def pack_batch(events):
    states=torch.tensor([e["state"] for _,e in events],dtype=torch.float32)
    actions=torch.tensor([e["actions"] for _,e in events],dtype=torch.float32)
    choices=torch.tensor([e["choice"] for _,e in events],dtype=torch.long)
    utilities=torch.tensor([e["candidate_utilities"] for _,e in events],dtype=torch.float32)
    return states,actions,choices,utilities


def train(events, name):
    if name not in ARMS:raise ValueError("Unplanned training arm")
    torch.manual_seed(57720)
    net=SelectionSpecialist(structured=name.startswith("structured"))
    optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    rng=random.Random(57720)
    losses=[]
    for epoch in range(EPOCHS):
        ix=list(range(len(events)))
        rng.shuffle(ix)
        running=0.
        net.train()
        for offset in range(0,len(ix),BATCH):
            batch=[events[j] for j in ix[offset:offset+BATCH]]
            states,actions,choices,utilities=pack_batch(batch)
            logits=net(states,actions)
            loss=teacher_loss(logits,choices,utilities,
                              mixed=name.endswith("mixed"))
            if not bool(torch.isfinite(loss)):
                raise ValueError("Diverged selection distillation")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            optim.step()
            running+=float(loss.detach())*len(batch)
        losses.append(running/len(events))
        print(json.dumps({"model":name,"epoch":epoch+1,"loss":losses[-1]}),flush=True)
    return net.eval(),losses


def assess(logits, event):
    with torch.no_grad():
        m=per_decision_metrics(logits,event["choice"])
        utility=event["candidate_utilities"]
        selected=int(logits.argmax())
        spread=max(utility)-min(utility)
        regret=(max(utility)-utility[selected])/spread if spread>1e-12 else 0.
        return {"top1":m["top1"],"top10":m["top10"],
                "normalized_ce":m["normalized_cross_entropy"],
                "regret":regret}


def evaluate(net,traces,*,is_rnad=False):
    results=defaultdict(list)
    with torch.no_grad():
        for record in traces:
            if is_rnad:memory=[net.initial_memory() for _ in range(2)]
            for e in record["examples"]:
                if is_rnad:
                    state=torch.tensor([e["state"]],dtype=torch.float32)
                    actions=torch.tensor([e["actions"]],dtype=torch.float32)
                    scores,_,memory_next=net(state,actions,memory[e["actor"]])
                    memory[e["actor"]]=memory_next.detach()
                    if e["phase"]!="selection":continue
                else:
                    if e["phase"]!="selection":continue
                    state=torch.tensor([e["state"]],dtype=torch.float32)
                    actions=torch.tensor([e["actions"]],dtype=torch.float32)
                    scores=net(state,actions)
                results[record["seed"]].append(assess(scores[0],e))
    if len(results)!=MATCHES_HOLDOUT:
        raise ValueError("Incomplete heldout seed clusters")
    keys=("top1","top10","normalized_ce","regret")
    overall={k:sum(row[k] for rows in results.values() for row in rows)
              /sum(len(v) for v in results.values()) for k in keys}
    overall["heldout_decisions"]=sum(len(v) for v in results.values())
    by_seed={seed:{k:sum(row[k] for row in rows)/len(rows)
                   for k in keys} for seed,rows in results.items()}
    return overall,by_seed


def paired_bootstrap80(base,can,seed_ids):
    # Four predeclared arms: percentile Bonferroni familywise >=80%
    # approximately: each two-sided 95% (alpha=.05 per comparison).
    diffs=[can[s]["top1"]-base[s]["top1"] for s in seed_ids]
    mean=sum(diffs)/len(diffs)
    rng=random.Random(991)
    n=len(diffs)
    vals=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n
                for _ in range(5000))
    return {"per_match_cluster_top1_delta":mean,
            "approximate_familywise80_CI":[vals[125],vals[4875]],
            "approximate_not_exact":True}


def feature_sufficiency_audit(traces):
    """V5 representational alias diagnostics; NO guarantee of sufficiency.

    Same action feature vectors with different utility imply irreducible
    action-level label ambiguities, even with a nonlinear model.
    """
    conflicts=0
    alias_decisions=0
    unseparated_trio_leads=0
    assessed=0
    rank_gaps=[]
    for _,event in selection_events(traces):
        assessed+=1
        utilities=event["candidate_utilities"]
        by_features=defaultdict(list)
        for i,feature in enumerate(event["actions"]):
            by_features[tuple(feature)].append(i)
        local_conflict=False
        for idxs in by_features.values():
            if len(idxs)>1 and max(utilities[i] for i in idxs)-min(utilities[i] for i in idxs)>1e-5:
                conflicts+=1
                local_conflict=True
        if local_conflict:alias_decisions+=1
        for rows in TRIO_ROWS:
            if len({tuple(event["actions"][i]) for i in rows})<3:
                unseparated_trio_leads+=1
        ranked=sorted(utilities,reverse=True)
        rank_gaps.append((ranked[0]-ranked[1])/
                         (max(utilities)-min(utilities)+1e-8))
    return {
        "selection_teacher_decisions":assessed,
        "same_action_vector_different_utility_groups":conflicts,
        "decisions_with_action_feature_utility_collision":alias_decisions,
        "trio_groups_with_indistinguishable_leads":unseparated_trio_leads,
        "mean_relative_best_vs_runner_up_utility_gap":sum(rank_gaps)/len(rank_gaps),
        "input_sufficiency_proven":False,
        "scope":"action-level exact collisions only; global context ambiguity and absent features remain possible",
    }


def save_model(net,path,name):
    data={
        "schema":"rnad-selection-specialist-96x48-v1",
        "arm":name,"structured":net.structured,
        "source_sha":current_git_sha(),
        "weights":{k:{"shape":list(t.shape),
                      "values":t.detach().flatten().tolist()}
                   for k,t in net.state_dict().items()},
        "training_accepted_teacher_matches":MATCHES_TRAIN,
        "training_epochs":EPOCHS,
        "not_champion":True,
    }
    raw=json.dumps(data,separators=(",",":"),allow_nan=False).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-dir",required=True)
    parser.add_argument("--heldout-dir",required=True)
    parser.add_argument("--frozen-selection-model",required=True)
    parser.add_argument("--baseline-report",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    train_traces,train_engine=load_traces(args.train_dir,SEED_TRAIN,MATCHES_TRAIN)
    held,held_engine=load_traces(args.heldout_dir,SEED_HOLDOUT,MATCHES_HOLDOUT)
    if train_engine!=held_engine:
        raise ValueError("Training/heldout different game engines")
    baseline_report=json.loads(Path(args.baseline_report).read_text())
    baseline_ref=baseline_report["phases"]["selection"]["top1_accuracy"]
    old=RNADNetwork(state_features=STATE_FEATURES_V5,action_features=ACTION_FEATURES_V5)
    sha=load_exported_json(old,args.frozen_selection_model,
                           expected_feature_schema=THREE_PHASE_FEATURE_SCHEMA_V5)
    if sha!=baseline_report["student_model_sha256"]:
        raise ValueError("Frozen selection model not the original heldout evaluator")
    baseline,base_clusters=evaluate(old.eval(),held,is_rnad=True)
    if abs(baseline["top1"]-baseline_ref)>1e-10:
        raise ValueError("Failed to reproduce baseline on original sealed teacher trace")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    events=selection_events(train_traces)
    results={"frozen-v5-selection-focus":{
        "model_sha256":sha,"heldout":baseline,"training_from_this_run":False}}
    seeds=list(range(SEED_HOLDOUT,SEED_HOLDOUT+MATCHES_HOLDOUT))
    for arm in ARMS:
        net,losses=train(events,arm)
        held_metrics,clusters=evaluate(net,held)
        results[arm]={
            "model_sha256":save_model(net,out/(arm+".json"),arm),
            "heldout":held_metrics,"losses":losses,
            "paired_vs_frozen":paired_bootstrap80(base_clusters,clusters,seeds),
        }
        print(json.dumps({"arm":arm,"top1":held_metrics["top1"],
            "top10":held_metrics["top10"],"regret":held_metrics["regret"]}),flush=True)
    result={
        "schema":"rnad-i5-selection-four-arm-exploratory-v1",
        "source_sha":current_git_sha(),"teacher_trace_source_sha":SOURCE_SHA,
        "engine_sha":train_engine,
        "teacher_id":TEACHER,
        "train_seed":SEED_TRAIN,"train_complete_matches":MATCHES_TRAIN,
        "holdout_seed":SEED_HOLDOUT,"development_holdout_matches":MATCHES_HOLDOUT,
        "development_holdout_reused_from_run":44032488678,
        "heldout_is_final_league_gate":False,
        "independent_unit":"teacher_complete_Match_seed",
        "familywise_CI":"approximate seed-cluster bootstrap 80% with 4-arm Bonferroni",
        "phase_only_network_not_integrated_policy":True,
        "teacher_visited_only":True,
        "league_training_allowed":False,"strength_established":False,
        "champion_promoted":False,
        "representation_audit":feature_sufficiency_audit(held),
        "models":results,
    }
    (out/"selection-four-arm-report.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# 選出専用モデル・3体構造化・Top1重点・入力十分性の検証",
           "","教師256完全Match／**再利用した探索用**128完全Match",
           "（教師が訪れた局面のTop1。一つのMatch内の選出は独立でない）。",
           "このholdoutは開発に使っているためリーグ開始の未使用確認には使えない。",
           "","| Arm | Top1 | Top10 | normalized CE ↓ | teacher regret ↓ |",
           "|---|---:|---:|---:|---:|"]
    for arm,row in results.items():
        v=row["heldout"]
        lines.append(f"| {arm} | {v['top1']:.2%} | {v['top10']:.2%} "
                     f"| {v['normalized_ce']:.4f} | {v['regret']:.4f} |")
    lines+=["","**表のモデルは研究専用。選出80%は統合方策では未達で、実戦勝率でもない。**",
            "","入力衝突監査（v5）:",
            "","- 完全一致action特徴量でutilityが異なる候補群: "+
            str(result["representation_audit"]["same_action_vector_different_utility_groups"]),
            "- 先発を変更しても特徴量が同じだったtrio群: "+
            str(result["representation_audit"]["trio_groups_with_indistinguishable_leads"]),
            "","ゼロでも情報が十分だとは証明しない。"]
    (out/"SELECTION_REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    main()
