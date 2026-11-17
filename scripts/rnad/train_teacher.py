"""Optional, bounded Stage0 fixed-Policy imitation experiment.

The teacher is a legitimate, versioned authority-accepted Actor,
not Metamon; trained weights are NOT R-NaD and never promoted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F

from scripts.rnad.neural import (RNADNetwork, export_json, load_exported_json,
                                 BATTLE_FEATURE_SCHEMA, THREE_PHASE_FEATURE_SCHEMA,
                                 THREE_PHASE_FEATURE_SCHEMA_V2,
                                 THREE_PHASE_FEATURE_SCHEMA_V4,
                                 THREE_PHASE_FEATURE_SCHEMA_V5,
                                 STATE_FEATURES_V3,ACTION_FEATURES_V3,
                                 STATE_FEATURES_V5,ACTION_FEATURES_V5)
from scripts.rnad.train import state_hash, current_git_sha, frozen_engine_sha


AQ_WEIGHTS_SHA256="5ccc11a3362c7f5a8874847e3afc1f19c539d4ef258bd82704f53d8a1db1946e"

def collect(seed, count, folder, *, three_phase=False,
            teacher='basic-stage0',features_v2=False,
            features_v4=False,features_v5=False,
            utility_labels=False,reuse_from=None):
    root=Path(reuse_from) if reuse_from is not None else folder/"teacher-matches"
    if reuse_from is None:
        root.mkdir(parents=True,exist_ok=True)
    elif not root.is_dir():
        raise ValueError("Cannot reuse missing independent accepted teacher traces")
    traces=[]
    for i in range(count):
        n=seed+i
        file=root/f"teacher-{n}.json"
        if reuse_from is None:
            if file.exists():
                raise ValueError("Refuse to overwrite teacher experiment")
            cmd=["node","scripts/rnad/export-teacher.mjs",
                 "--seed",str(n),"--out",str(file)]
            if three_phase:cmd.extend(["--three-phase","true"])
            if teacher!='basic-stage0':cmd.extend(["--teacher",teacher])
            if features_v2:cmd.extend(["--features-v2","true"])
            if features_v4:cmd.extend(["--features-v4","true"])
            if features_v5:cmd.extend(["--features-v5","true"])
            if utility_labels:cmd.extend(["--utility-labels","true"])
            subprocess.run(cmd,check=True,timeout=120)
        elif not file.is_file():
            raise ValueError("Cannot reuse missing authority-accepted teacher Match")
        trace=json.loads(file.read_text())
        expected_schema=("rnad-aq-public-strategy-selection-three-phase-teacher-v1"
                         if teacher=="aq-public-strategy-selection" else
                         "rnad-aq-public-belief-three-phase-teacher-v1"
                         if teacher=="aq-public-belief" else
                         "rnad-basic-stage0-three-phase-teacher-v1" if three_phase
                         else "rnad-fixed-stage0-teacher-match-v1")
        expected_teacher=("rnad-aq-public-strategy-selection-v1"
                          if teacher=="aq-public-strategy-selection" else
                          "rnad-aq-public-belief-v2" if teacher=="aq-public-belief"
                          else "basic-stage0-3phase" if three_phase
                          else "stage0-fixed-untrained")
        if (trace.get("schema")!=expected_schema
            or trace.get("feature_schema")!=(
                THREE_PHASE_FEATURE_SCHEMA_V5 if features_v5 else
                THREE_PHASE_FEATURE_SCHEMA_V4 if features_v4 else
                THREE_PHASE_FEATURE_SCHEMA_V2 if features_v2 else
                THREE_PHASE_FEATURE_SCHEMA if three_phase
                else BATTLE_FEATURE_SCHEMA)
            or trace.get('utility_labels',None)!=
              ('aq-public-phase-rank-v1' if utility_labels else None)
            or trace.get("seed")!=n
            or trace.get("source_sha")!=current_git_sha()
            or trace.get("engine_sha")!=frozen_engine_sha()
            or trace.get("completed") is not True
            or trace.get("teacher")!=expected_teacher
            or (teacher.startswith("aq-public-") and
                trace.get("teacher_model_sha256")!=AQ_WEIGHTS_SHA256)
            or trace.get("score",[])[trace["winner"]]!=4
            or not trace.get("examples") or
            len(trace["examples"])!=sum(trace.get("action_counts",[]))+
                (4*trace.get("rounds",0) if three_phase else 0)):
            raise ValueError("Incomplete or untrusted teacher Match")
        traces.append(trace)
    return traces


def phase_balanced_mean_loss(events, *, enabled=False, weights=None):
    """Each completed Match gives each phase an equal mean-gradient weight.

    Standard mode is byte-for-byte the earlier event-mean objective. Each
    event is (phase, differentiable scalar loss), never a privileged state.
    """
    if not events or any(phase not in ('draft','selection','battle')
                         for phase,_ in events):
        raise ValueError("Missing or malformed accepted event losses")
    if weights is not None:
        if (not enabled or not isinstance(weights,dict) or
            set(weights)!=set(('draft','selection','battle')) or
            any(type(x) not in (int,float) or not math.isfinite(x) or x<=0
                for x in weights.values())):
            raise ValueError("Invalid positive finite phase weights")
    if not enabled:
        return torch.stack([loss for _,loss in events]).mean()
    grouped={p:[] for p in ('draft','selection','battle')}
    for phase,loss in events:
        grouped[phase].append(loss)
    if any(not losses for losses in grouped.values()):
        raise ValueError("Phase-balanced training requires all three phases")
    if weights is None:
        return torch.stack([torch.stack(grouped[p]).mean()
            for p in ('draft','selection','battle')]).mean()
    return sum(torch.stack(grouped[p]).mean()*weights[p]
        for p in ('draft','selection','battle'))/sum(weights.values())



def utility_teacher_loss(logits, utilities, phase):
    """Top utility candidates get graded soft labels; off-shortlist logits
    remain in the normalizer. This is a surrogate, never a win objective.
    """
    if phase not in ('draft','selection') or not isinstance(logits,torch.Tensor) or (
        logits.ndim!=2 or logits.size(0)!=1 or not isinstance(utilities,list)
        or len(utilities)!=logits.size(1) or not utilities or
        any(type(v) not in (int,float) or not math.isfinite(v) for v in utilities)):
        raise ValueError('Invalid public AQ distillation logits/utility vector')
    scores=torch.tensor(utilities,dtype=logits.dtype,device=logits.device)
    topk=min(12 if phase=='draft' else 10,len(utilities))
    indices=torch.argsort(scores,descending=True,stable=True)[:topk]
    top_scores=scores[indices]
    # A stable within-topk temperature bounds the teacher signal.
    targets=torch.softmax((top_scores-top_scores.max())/5.0,dim=0)
    loss=-(targets*torch.log_softmax(logits[0],dim=0)[indices]).sum()
    if not bool(torch.isfinite(loss)):
        raise ValueError('Invalid AQ utility distillation result')
    return loss


def normalized_utility_regret(utilities,index):
    if (not isinstance(utilities,list) or not utilities or
        not isinstance(index,int) or not 0<=index<len(utilities) or
        any(type(v) not in (int,float) or not math.isfinite(v) for v in utilities)):
        raise ValueError('Invalid public AQ utility regret')
    gap=max(utilities)-min(utilities)
    return (max(utilities)-utilities[index])/gap if gap>1e-12 else 0.0


def initialize_teacher_model(feature_schema,initial_model=None):
    """Optional version-compatible v2/v4/v5 warm start with a NEW optimizer.

    This is not an exact optimizer/magnet/RNG checkpoint resume. It copies
    weights only, never edits the saved source model or its provenance.
    """
    if feature_schema not in (THREE_PHASE_FEATURE_SCHEMA_V2,
                              THREE_PHASE_FEATURE_SCHEMA_V4,
                              THREE_PHASE_FEATURE_SCHEMA_V5):
        if initial_model is not None:
            raise ValueError('Only compatible v2/v4/v5 actor models may warm-start')
    model=(RNADNetwork(state_features=STATE_FEATURES_V5,
                       action_features=ACTION_FEATURES_V5)
           if feature_schema==THREE_PHASE_FEATURE_SCHEMA_V5 else
           RNADNetwork(state_features=STATE_FEATURES_V3,
                       action_features=ACTION_FEATURES_V3)
           if feature_schema==THREE_PHASE_FEATURE_SCHEMA_V4 else RNADNetwork())
    if initial_model is None:
        return model,None
    path=Path(initial_model)
    if not path.is_file():
        raise ValueError('Only a present compatible v2/v4 model may warm-start')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    parsed=load_exported_json(model,path,expected_feature_schema=feature_schema)
    if parsed!=digest:
        raise ValueError("Untrusted source model SHA for teacher warm-start")
    return model,digest


def train(*,seed=975601,out="artifacts/rnad-teacher",
          three_phase=False,matches=32,epochs=4,
          teacher='basic-stage0',features_v2=False,features_v4=False,
          features_v5=False,
          phase_balanced=False,
          phase_profile='equal',label_smoothing=0.0,reuse_teacher_traces=None,
          initial_model=None,utility_distillation=False,utility_mix=1.0):
    if (type(seed) is not int or type(matches) is not int or
        type(epochs) is not int or type(three_phase) is not bool or
        type(features_v2) is not bool or
        type(features_v4) is not bool or
        type(features_v5) is not bool or
        type(utility_distillation) is not bool or
        type(utility_mix) not in (int,float) or
        not math.isfinite(utility_mix) or not 0<=utility_mix<=1 or
        (not utility_distillation and utility_mix!=1.0) or
        (utility_distillation and
         (not (features_v4 or features_v5) or
          teacher!='aq-public-strategy-selection'
          or float(label_smoothing)!=0.0)) or
        ((features_v2 or features_v4 or features_v5) and not three_phase) or
        sum((features_v2,features_v4,features_v5))>1 or
        type(phase_balanced) is not bool or
        (phase_balanced and not three_phase) or
        phase_profile not in ('equal','draft-selection','selection-focus','battle-focus') or
        (phase_profile!='equal' and not phase_balanced) or
        type(label_smoothing) not in (int,float) or
        not math.isfinite(float(label_smoothing)) or
        not 0 <= float(label_smoothing) <= .15 or
        (reuse_teacher_traces is not None and
         not isinstance(reuse_teacher_traces,(str,Path))) or
        (initial_model is not None and
         (not isinstance(initial_model,(str,Path)) or
          not three_phase or not (features_v2 or features_v4 or features_v5) or
          not teacher.startswith('aq-public-'))) or
        not 1<=matches<=256 or not 1<=epochs<=6 or
        teacher not in ('basic-stage0','aq-public-belief',
                        'aq-public-strategy-selection') or
        (teacher.startswith('aq-public-') and not three_phase) or
        not 1<=seed<=0xffffffff-matches):
        raise ValueError("Invalid bounded teacher seed/matches/epochs")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    folder=Path(out)
    folder.mkdir(parents=True,exist_ok=True)
    feature_schema=(THREE_PHASE_FEATURE_SCHEMA_V5 if features_v5 else
                    THREE_PHASE_FEATURE_SCHEMA_V4 if features_v4 else
                    THREE_PHASE_FEATURE_SCHEMA_V2 if features_v2 else
                    THREE_PHASE_FEATURE_SCHEMA if three_phase
                    else BATTLE_FEATURE_SCHEMA)
    model,initial_file_sha=initialize_teacher_model(feature_schema,initial_model)
    initial_hash=state_hash(model)
    export_json(model,folder/"initial-model.json",feature_schema=feature_schema)
    traces=collect(seed,matches,folder,three_phase=three_phase,
                   teacher=teacher,features_v2=features_v2,
                   features_v4=features_v4,features_v5=features_v5,
                   utility_labels=utility_distillation,
                   reuse_from=reuse_teacher_traces)
    phase_weights=({'draft':2.,'selection':2.,'battle':1.}
                   if phase_profile=='draft-selection' else
                   {'draft':1.,'selection':4.,'battle':1.}
                   if phase_profile=='selection-focus' else
                   {'draft':1.,'selection':1.,'battle':4.}
                   if phase_profile=='battle-focus' else None)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0006)
    results=[]
    for epoch in range(epochs):
        order=list(range(len(traces)))
        random.Random(seed+epoch).shuffle(order)
        cumulative=0.
        n=0
        correct=0
        phase_correct={p:0 for p in ('draft','selection','battle')}
        phase_count={p:0 for p in ('draft','selection','battle')}
        phase_regret={p:0.0 for p in ('draft','selection')}
        for index in order:
            match=traces[index]
            memory=[model.initial_memory() for _ in range(2)]
            losses=[]
            for event in match["examples"]:
                seat=event["actor"]
                if (seat not in (0,1) or
                    len(event["state"])!=model.state_features or
                    not event["actions"] or
                    any(len(x)!=model.action_features for x in event["actions"]) or
                    not 0<=event["choice"]<len(event["actions"]) or
                    event["terminal_target"] not in (-1,1)):
                    raise ValueError("Invalid accepted teacher example")
                s=torch.tensor([event["state"]],dtype=torch.float32)
                a=torch.tensor([event["actions"]],dtype=torch.float32)
                logits,value,h=model(s,a,memory[seat])
                memory[seat]=h.detach()
                phase=event.get('phase')
                if utility_distillation and phase in phase_regret:
                    utilities=event.get('candidate_utilities')
                    soft=utility_teacher_loss(logits,utilities,phase)
                    hard=F.cross_entropy(logits,
                      torch.tensor([event['choice']]))
                    loss=utility_mix*soft+(1-utility_mix)*hard
                    phase_regret[phase]+=normalized_utility_regret(
                        utilities,int(logits[0].argmax()))
                else:
                    loss=F.cross_entropy(
                        logits,torch.tensor([event["choice"]]),
                        label_smoothing=float(label_smoothing))
                target=torch.tensor([float(event["terminal_target"])])
                phase=event.get("phase")
                if phase not in phase_count:
                    raise ValueError("Unexpected supervised event phase")
                losses.append((phase,loss+0.05*F.mse_loss(value,target)))
                matched=int(int(logits[0].argmax())==event["choice"])
                correct+=matched
                phase_correct[phase]+=matched
                phase_count[phase]+=1
                n+=1
            batch_loss=phase_balanced_mean_loss(
                losses,enabled=phase_balanced,weights=phase_weights)
            if not torch.isfinite(batch_loss):
                raise ValueError("Nonfinite teacher learning loss")
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.,
                                           error_if_nonfinite=True)
            optimizer.step()
            cumulative+=float(batch_loss.detach())*len(match["examples"])
        if not n:raise ValueError("Empty teacher learning epoch")
        row={"epoch":epoch+1,"mean_loss":cumulative/n,
             "phase_normalized_utility_regret":
               ({p:phase_regret[p]/phase_count[p] for p in phase_regret}
                if utility_distillation else None),
             "imitation_top1_accuracy":correct/n,
             "accepted_examples":n,
             "phase_example_counts":phase_count,
             "phase_top1_accuracy":{p:phase_correct[p]/phase_count[p]
                 if phase_count[p] else None for p in phase_count},
             "model_sha256":state_hash(model)}
        if not (math.isfinite(row["mean_loss"]) and
                math.isfinite(row["imitation_top1_accuracy"])):
            raise ValueError("Nonfinite imitation metrics")
        print(json.dumps(row),flush=True)
        results.append(row)
    trained_hash=state_hash(model)
    if trained_hash==initial_hash:
        raise AssertionError("Teacher learning did not change weights")
    export_json(model,folder/"trained-model.json",feature_schema=feature_schema)
    report={"schema":("rnad-three-phase-imitation-report-v1" if three_phase
                      else "rnad-optional-stage0-imitation-report-v1"),
            "feature_schema":feature_schema,
            "phase_balanced":phase_balanced,
            "phase_profile":phase_profile,
            "utility_distillation":utility_distillation,
            "utility_hard_label_mix":float(utility_mix),
            "label_smoothing":float(label_smoothing),
            "reused_authority_accepted_teacher_traces":
                reuse_teacher_traces is not None,
            "source_sha":current_git_sha(),"engine_sha":frozen_engine_sha(),
            "teacher_id":("rnad-aq-public-strategy-selection-v1"
                          if teacher=="aq-public-strategy-selection" else
                          "rnad-aq-public-belief-v2"
                          if teacher=="aq-public-belief" else
                          "basic-stage0-3phase" if three_phase
                          else "stage0-fixed-untrained"),
            "teacher_weights_sha256":(AQ_WEIGHTS_SHA256 if
                                     teacher.startswith('aq-public-') else None),
            "seed":seed,
            "teacher_completed_matches":matches,"teacher_updates":matches*epochs,
            "initial_model_sha256":initial_hash,
            "warmstart_source_model_file_sha256":initial_file_sha,
            "warmstart_optimizer_reused":False,
            "trained_model_sha256":trained_hash,
            "epochs":results,"rnad_updates":0,"is_champion":False}
    (folder/"teacher-report.json").write_text(json.dumps(
        report,indent=2,allow_nan=False)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="artifacts/rnad-teacher")
    p.add_argument("--seed",type=int,default=975601)
    p.add_argument("--matches",type=int,default=32)
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--three-phase",action="store_true")
    p.add_argument("--features-v2",action="store_true")
    p.add_argument("--features-v4",action="store_true")
    p.add_argument("--features-v5",action="store_true")
    p.add_argument("--utility-distillation",action="store_true")
    p.add_argument("--utility-mix",type=float,default=1.0)
    p.add_argument("--phase-balanced",action="store_true")
    p.add_argument("--phase-profile",choices=["equal","draft-selection","selection-focus","battle-focus"],
                   default="equal")
    p.add_argument("--label-smoothing",type=float,default=0.0)
    p.add_argument("--reuse-teacher-traces",default=None)
    p.add_argument("--initial-model",default=None)
    p.add_argument("--teacher",choices=["basic-stage0","aq-public-belief","aq-public-strategy-selection"],
                   default="basic-stage0")
    args=p.parse_args()
    print(json.dumps(train(out=args.out,seed=args.seed,matches=args.matches,
                           epochs=args.epochs,three_phase=args.three_phase,
                           teacher=args.teacher,
                           features_v2=args.features_v2,
                           features_v4=args.features_v4,
                           features_v5=args.features_v5,
                           utility_distillation=args.utility_distillation,
                           utility_mix=args.utility_mix,
                           phase_balanced=args.phase_balanced,
                           phase_profile=args.phase_profile,
                           label_smoothing=args.label_smoothing,
                           reuse_teacher_traces=args.reuse_teacher_traces,
                           initial_model=args.initial_model),
                     indent=2))
