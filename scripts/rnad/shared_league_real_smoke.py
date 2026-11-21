"""Tiny real-authority Match NeuRD/V-trace update preflight for I5.

This is NOT the shared generation-0 Learner's checkpoint or a performance
claim. It verifies the actual three-phase accepted Match tensors can backprop
through each Head and save/restore a separate disposable optimizer state.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from scripts.rnad.shared_league_contract import _verify_events, verify_manifest
from scripts.rnad.shared_league_learner import (
    _models, _match_loss, _atomic_checkpoint, SCHEMA,
)
from scripts.rnad.train import current_git_sha, frozen_engine_sha


def run(bootstrap, smoke, out):
    root, source, target=Path(bootstrap),Path(smoke),Path(out)
    m=json.loads((root/"generation-manifest.json").read_text())
    verify_manifest(m)
    if m["engine_sha"]!=frozen_engine_sha():
        raise ValueError("Actual accepted Match uses different game engine")
    files=sorted(source.rglob("*.json.gz"))
    if len(files)!=4:
        raise ValueError("Need exactly four independently accepted smoke Matches")
    rows=[]
    for file in files:
        with gzip.open(file,"rt") as f:
            x=json.load(f)
        if (x.get("source_sha")!=m["source_sha"] or
            x.get("engine_sha")!=m["engine_sha"] or
            x.get("policy_id")!=m["policy_id"] or
            x.get("main_sha256")!=m["main_sha256"] or
            x.get("magnet_sha256")!=m["magnet_sha256"] or
            x.get("completed") is not True or
            x["score"][x["winner"]]!=4):
            raise ValueError("Fake or mismatched accepted real smoke")
        _verify_events(x, battle_schema=m.get("battle_feature_schema", "rnad-three-phase-features-v5"))
        role="fixed" if "frozen" in file.parts else "self"
        rows.append((x,role))
    if (len({x["match_id"] for x,_ in rows})!=4 or
        sorted(role for _,role in rows)!=
        ["fixed","fixed","self","self"]):
        raise ValueError("Wrong real-smoke seat/role coverage")
    index=json.loads((root/"checkpoint-index.json").read_text())
    checkpoint=root/"checkpoint-g0.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=index["sha256"]:
        raise ValueError("Corrupted shared full-state starting checkpoint")
    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model,target_model,magnet,optimizer,scheduler=_models(root,ckpt)
    torch.set_num_threads(1)
    optimizer.zero_grad(set_to_none=True)
    losses,ratios=[],[]
    for match,role in rows:
        loss,ratio=_match_loss(model,target_model,magnet,match,role=role)
        (loss/len(rows)).backward()
        losses.append(float(loss.detach()))
        ratios.append(ratio)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,
                                        error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite real accepted gradient")
    old={k:v.clone() for k,v in model.state_dict().items()}
    optimizer.step()
    scheduler.step()
    changed={part: any(
        not torch.equal(v,model.state_dict()[k])
        for k,v in old.items() if k.startswith(part))
        for part in ("draft.","selection.","battle.",
                     "draft_value.","selection_value.")}
    if not all(changed.values()):
        raise ValueError("Real accepted Match failed to update a shared Head")
    ckpt.update({"model":{k:v.clone() for k,v in model.state_dict().items()},
                 "optimizer":optimizer.state_dict(),
                 "scheduler":scheduler.state_dict(),
                 "step":1,
                 "training_match_ids":[x["match_id"] for x,_ in rows],
                 "replay_code_sha":current_git_sha()})
    target.mkdir(parents=True,exist_ok=False)
    _atomic_checkpoint(target/"research-smoke-only.pt",ckpt)
    summary={
        "schema":"rnad-i5-real-accepted-shared-learner-smoke-v1",
        "full_four_win_authoritative_matches":4,
        "accepted_decisions":sum(len(x["events"]) for x,_ in rows),
        "training_code_sha":current_git_sha(),
        "original_acceptance_source_sha":m["source_sha"],
        "engine_sha":m["engine_sha"],
        "optimizer_updates":1,
        "updated_heads":changed,
        "mean_neurd_vtrace_loss":sum(losses)/len(losses),
        "maximum_importance_ratio":max(ratios),
        "research_only":True,
        "generation0_1280_match_training_finished":False,
        "champion_promoted":False,
    }
    (target/"smoke-report.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("bootstrap","smoke","out"):
        p.add_argument("--"+n,required=True)
    a=p.parse_args()
    run(a.bootstrap,a.smoke,a.out)
