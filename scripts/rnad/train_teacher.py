"""Optional, bounded Stage0 fixed-Policy imitation experiment.

The teacher is a legitimate, versioned authority-accepted Actor,
not Metamon; trained weights are NOT R-NaD and never promoted.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F

from scripts.rnad.neural import RNADNetwork, export_json
from scripts.rnad.train import state_hash, current_git_sha, frozen_engine_sha


def collect(seed, count, folder):
    root=folder/"teacher-matches"
    root.mkdir(parents=True,exist_ok=True)
    traces=[]
    for i in range(count):
        n=seed+i
        file=root/f"teacher-{n}.json"
        if file.exists():
            raise ValueError("Refuse to overwrite teacher experiment")
        subprocess.run(["node","scripts/rnad/export-teacher.mjs",
                        "--seed",str(n),"--out",str(file)],
                       check=True,timeout=120)
        trace=json.loads(file.read_text())
        if (trace.get("schema")!="rnad-fixed-stage0-teacher-match-v1"
            or trace.get("feature_schema")!="rnad-battle-features-v1"
            or trace.get("seed")!=n
            or trace.get("source_sha")!=current_git_sha()
            or trace.get("engine_sha")!=frozen_engine_sha()
            or trace.get("completed") is not True
            or trace.get("teacher")!="stage0-fixed-untrained"
            or trace.get("score",[])[trace["winner"]]!=4
            or not trace.get("examples")):
            raise ValueError("Incomplete or untrusted teacher Match")
        traces.append(trace)
    return traces


def train(*,seed=975601,out="artifacts/rnad-teacher"):
    if not isinstance(seed,int) or not 1<=seed<=0xffffffff-32:
        raise ValueError("Invalid bounded teacher seed")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    folder=Path(out)
    folder.mkdir(parents=True,exist_ok=True)
    model=RNADNetwork()
    initial_hash=state_hash(model)
    export_json(model,folder/"initial-model.json")
    traces=collect(seed,32,folder)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0006)
    results=[]
    for epoch in range(4):
        order=list(range(len(traces)))
        random.Random(seed+epoch).shuffle(order)
        cumulative=0.
        n=0
        correct=0
        for index in order:
            match=traces[index]
            memory=[model.initial_memory() for _ in range(2)]
            losses=[]
            for event in match["examples"]:
                seat=event["actor"]
                if (seat not in (0,1) or len(event["state"])!=32 or
                    not event["actions"] or
                    any(len(x)!=16 for x in event["actions"]) or
                    not 0<=event["choice"]<len(event["actions"]) or
                    event["terminal_target"] not in (-1,1)):
                    raise ValueError("Invalid accepted teacher example")
                s=torch.tensor([event["state"]],dtype=torch.float32)
                a=torch.tensor([event["actions"]],dtype=torch.float32)
                logits,value,h=model(s,a,memory[seat])
                memory[seat]=h.detach()
                loss=F.cross_entropy(logits,torch.tensor([event["choice"]]))
                target=torch.tensor([float(event["terminal_target"])])
                losses.append(loss+0.05*F.mse_loss(value,target))
                correct+=int(int(logits[0].argmax())==event["choice"])
                n+=1
            batch_loss=torch.stack(losses).mean()
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
             "imitation_top1_accuracy":correct/n,
             "accepted_examples":n,"model_sha256":state_hash(model)}
        if not (math.isfinite(row["mean_loss"]) and
                math.isfinite(row["imitation_top1_accuracy"])):
            raise ValueError("Nonfinite imitation metrics")
        print(json.dumps(row),flush=True)
        results.append(row)
    trained_hash=state_hash(model)
    if trained_hash==initial_hash:
        raise AssertionError("Teacher learning did not change weights")
    export_json(model,folder/"trained-model.json")
    report={"schema":"rnad-optional-stage0-imitation-report-v1",
            "source_sha":current_git_sha(),"engine_sha":frozen_engine_sha(),
            "teacher_id":"stage0-fixed-untrained","seed":seed,
            "teacher_completed_matches":32,"teacher_updates":128,
            "initial_model_sha256":initial_hash,
            "trained_model_sha256":trained_hash,
            "epochs":results,"rnad_updates":0,"is_champion":False}
    (folder/"teacher-report.json").write_text(json.dumps(
        report,indent=2,allow_nan=False)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="artifacts/rnad-teacher")
    args=p.parse_args()
    print(json.dumps(train(out=args.out),indent=2))
