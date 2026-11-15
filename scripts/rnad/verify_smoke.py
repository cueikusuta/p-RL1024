"""RNAD-001 hosted 16-complete-Match learning evidence verifier.

This does not assert strategic strength or Champion status. It fails closed
if Match attribution, counts, reward schema, or model updates are invalid.
"""
import argparse
import json
import math
from pathlib import Path
import torch

from scripts.rnad.neural import (RNADNetwork,BATTLE_FEATURE_SCHEMA,
                                 THREE_PHASE_FEATURE_SCHEMA)
from scripts.rnad.train import state_hash, current_git_sha, frozen_engine_sha


def verify(*, root, seed, matches=16, three_phase=False):
    if not (1 <= seed <= 0xffffffff and 1 <= matches <= 16 and
            seed+matches-1 <= 0xffffffff):
        raise ValueError("Invalid bounded learning run")
    p=Path(root)
    schema=(THREE_PHASE_FEATURE_SCHEMA if three_phase
            else BATTLE_FEATURE_SCHEMA)
    model_id='RNAD-002-3phase' if three_phase else 'RNAD-001'
    report_schema=('rnad-three-phase-training-report-v1' if three_phase
                   else 'rnad-smoke-training-report-v1')
    trace_schema=('rnad-accepted-three-phase-training-match-v1' if three_phase
                  else 'rnad-accepted-training-match-v1')
    report=json.loads((p/"report.json").read_text())
    index=json.loads((p/"checkpoints"/"latest.json").read_text())
    if (report.get("schema")!=report_schema
        or report.get("feature_schema")!=schema
        or report.get("model_id")!=model_id
        or report.get("is_champion") is not False
        or report.get("completed_matches")!=matches
        or report.get("updates")!=matches
        or len(report.get("results",[]))!=matches
        or report.get("source_sha")!=current_git_sha()
        or report.get("engine_sha")!=frozen_engine_sha()
        or index.get("checkpoint_sha256")!=report.get("last_match_checkpoint_sha")
        or index.get("metadata",{}).get("completed_matches")!=matches
        or index.get("metadata",{}).get("updates")!=matches
        or index.get("metadata",{}).get("initial_seed")!=seed
        or index.get("metadata",{}).get("model_id")!=model_id
        or index.get("metadata",{}).get("observation_schema_version")!=schema):
        raise ValueError("Incorrect accepted Match or checkpoint evidence")
    initial_seed=seed
    torch.manual_seed(initial_seed)
    initial_hash=state_hash(RNADNetwork())
    if report["model_sha256"]==initial_hash:
        raise ValueError("Learning weights did not change")
    if report["magnet_sha256"]==initial_hash:
        raise ValueError("Periodically refreshed magnet did not change")
    action_sum=0
    completed_ids=[]
    observed_seats=set()
    for i,summary in enumerate(report["results"]):
        match_seed=seed+i
        if summary.get("seed")!=match_seed or summary.get("rounds",0)<4:
            raise ValueError("Incorrect complete Match seed/order")
        for key in ("loss","critic_loss","policy_loss","behavior_ratio_max"):
            val=summary.get(key)
            if not isinstance(val,(int,float)) or not math.isfinite(val):
                raise ValueError("Nonfinite training measure: "+key)
        trace=json.loads((p/"rollouts"/f"accepted-seed{match_seed}.json").read_text())
        events=trace.get("events",[])
        match_id=f"rnad-match-{match_seed}"
        if (trace.get("schema")!=trace_schema
            or trace.get("feature_schema")!=schema
            or trace.get("source_sha")!=current_git_sha()
            or trace.get("engine_sha")!=frozen_engine_sha()
            or trace.get("seed")!=match_seed
            or trace.get("match_id")!=match_id
            or trace.get("completed") is not True
            or trace.get("score",[])[trace["winner"]]!=4
            or not events
            or len(events)!=summary.get("accepted_decisions")
            or len(events)!=sum(trace.get("action_counts",[]))+
              (4*trace["rounds"] if three_phase else 0)):
            raise ValueError("Untrusted, partial or mismatched Match")
        if three_phase:
            for stage in ('draft','selection'):
                if sum(event.get('phase')==stage for event in events)!=2*trace['rounds']:
                    raise ValueError('Missing accepted phase decisions')
            if sum(event.get('phase')=='battle' for event in events)!=sum(trace['action_counts']):
                raise ValueError('Missing accepted battle decisions')
        elif any(event.get('phase')!='battle' for event in events):
            raise ValueError('Unexpected phase in Battle-only evaluation')
        for item in events:
            if (item.get("actor") not in (0,1)
                or item.get("reward") not in ([0,0],[1,-1],[-1,1])
                or len(item.get("state",[]))!=32
                or not item.get("actions")
                or any(len(a)!=16 for a in item["actions"])
                or not 0<=item.get("action_index",-1)<len(item["actions"])):
                raise ValueError("Invalid two-player accepted learning event")
            observed_seats.add(item["actor"])
        if any(x["reward"]!=[0,0] for x in events[:-1]) or (
            events[-1]["reward"] not in ([1,-1],[-1,1])):
            raise ValueError("Terminal-only reward violated")
        action_sum+=len(events)
        completed_ids.append(match_id)
    md=index["metadata"]
    if (md.get("accepted_decisions")!=action_sum
        or md.get("completed_match_ids")!=completed_ids
        or observed_seats!={0,1}):
        raise ValueError("Accepted action total or seat coverage mismatch")
    return {"schema":("rnad-three-phase-16-match-smoke-gate-v1" if three_phase
                       else "rnad-16-match-smoke-gate-v1"),
            "three_phase":three_phase,
            "completed_matches":matches,"optimizer_updates":matches,
            "accepted_actions":action_sum,"actor_seats":sorted(observed_seats),
            "source_sha":current_git_sha(),
            "model_sha256":report["model_sha256"],
            "magnet_sha256":report["magnet_sha256"],
            "learned_weights_changed":True,
            "frozen_magnet_refreshed":True,"champion":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--matches",type=int,default=16)
    parser.add_argument("--three-phase",action="store_true")
    args=parser.parse_args()
    print(json.dumps(verify(root=args.out,seed=args.seed,
                            matches=args.matches,
                            three_phase=args.three_phase),indent=2))
