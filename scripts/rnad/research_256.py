"""Bounded RNAD-001 256-completed-Match research, NOT Champion promotion.

Runs on a single GitHub-hosted runner, one writer, 16-match checkpoints.
All evaluation happens on disjoint held-out seeds in the same pinned SHA.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch

from scripts.rnad.train import train, state_hash, current_git_sha, frozen_engine_sha
from scripts.rnad.neural import RNADNetwork, export_json


def run(*, seed=975201, batches=16, out="artifacts/rnad-256"):
    batch_size=16
    if (not isinstance(seed,int) or not 1 <= seed <= 0xffffffff-256
        or batches!=16):
        raise ValueError("RNAD research scale gate is exactly 16 x 16 complete Matches")
    root=Path(out)
    root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    initial=RNADNetwork()
    initial_hash=state_hash(initial)
    export_json(initial,root/"initial-model.json")
    summaries=[]
    total_actions=0
    for block in range(batches):
        report=train(seed=seed,matches=batch_size,out=str(root),
                     resume=(block>0))
        if (report["completed_matches"]!=(block+1)*batch_size
            or report["updates"]!=(block+1)*batch_size
            or report["source_sha"]!=current_git_sha()
            or len(report["results"])!=batch_size):
            raise AssertionError("Unexpected truncated/duplicated training block")
        if [x["seed"] for x in report["results"]] != [
                seed+block*batch_size+j for j in range(batch_size)]:
            raise AssertionError("A self-play seed was reused or skipped")
        if not all(math.isfinite(x["loss"]) and
                   math.isfinite(x["critic_loss"]) and
                   math.isfinite(x["policy_loss"]) and
                   x["accepted_decisions"]>0 for x in report["results"]):
            raise AssertionError("Nonfinite or empty learning batch")
        delta=sum(x["accepted_decisions"] for x in report["results"])
        total_actions+=delta
        summaries.append({
            "batch":block+1,"completed_matches":report["completed_matches"],
            "optimizer_updates":report["updates"],
            "new_accepted_actions":delta,
            "model_sha256":report["model_sha256"],
            "magnet_sha256":report["magnet_sha256"],
            "checkpoint_sha256":report["last_match_checkpoint_sha"],
            "mean_loss":sum(x["loss"] for x in report["results"])/batch_size})
        print(json.dumps(summaries[-1]),flush=True)
    checkpoint=json.loads((root/"checkpoints"/"latest.json").read_text())
    md=checkpoint.get("metadata",{})
    if (md.get("completed_matches")!=256 or md.get("updates")!=256
        or md.get("accepted_decisions")!=total_actions
        or md.get("initial_seed")!=seed
        or len(set(md.get("completed_match_ids",[])))!=256
        or checkpoint.get("checkpoint_sha256")!=summaries[-1]["checkpoint_sha256"]
        or report["model_sha256"]==initial_hash):
        raise AssertionError("Incorrect 256-Match checkpoint, accepted counts or weights")
    output={
        "schema":"rnad-256-complete-match-research-v1",
        "source_sha":current_git_sha(),"engine_sha":frozen_engine_sha(),
        "training_seed":seed,"completed_matches":256,
        "optimizer_updates":256,"accepted_actions":total_actions,
        "initial_model_sha256":initial_hash,
        "trained_model_sha256":report["model_sha256"],
        "magnet_sha256":report["magnet_sha256"],
        "checkpoint_sha256":checkpoint["checkpoint_sha256"],
        "learning_weight_changed":True,
        "batches":summaries,"is_champion":False,
        "evidence":"complete authoritative match, not policy improvement"}
    (root/"scale-report.json").write_text(
        json.dumps(output,indent=2,allow_nan=False)+"\n")
    return output


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--seed",type=int,default=975201)
    cli.add_argument("--out",default="artifacts/rnad-256")
    args=cli.parse_args()
    print(json.dumps(run(seed=args.seed,out=args.out),indent=2))
