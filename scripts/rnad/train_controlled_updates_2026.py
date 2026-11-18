"""Controlled RNAD update-frequency A/B: ONLY optimizer passes differ (1 vs 4).
Frozen v2 c005 -> v3 policy, same model weights, source code, opponent pool,
seed/Match stream, accepted-action engine and exact 2048 Match budget.
One independent writer per arm; no empirical claim until paired heldout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from scripts.rnad.neural import (
    RNADNetwork,STATE_FEATURES_V3,ACTION_FEATURES_V3,
    THREE_PHASE_FEATURE_SCHEMA_V3,migrate_v2_export_to_v3,export_json
)
from scripts.rnad.train import train,current_git_sha,frozen_engine_sha
from scripts.rnad.night_league_2026 import prepare_models,_safe_cleanup,sha

SCHEMA = "rnad-i5-controlled-c005-v3-update1-vs-update4-v1"
TRAIN_SEED = 6500001
TRAIN_MATCHES = 2048
MODEL_ORIGIN = "c005"
SOURCE_ARCHIVE_SHA = "f08a9587d6a7033546eab00ffa6479399ca71c2f"

def require(cond, msg):
    if not cond:
        raise ValueError("Controlled 1vs4 experiment: " + msg)

def initial_exact(model_dir, series):
    root=Path(model_dir)
    manifest=json.loads((root/"manifest.json").read_text())
    require(manifest["schema"]=="rnad-i5-queued-evaluation-models-v1" and
            manifest["source_sha"]==SOURCE_ARCHIVE_SHA and
            manifest["engine_sha"]==frozen_engine_sha(),
            "unexpected historical original model manifest")
    source=root/(MODEL_ORIGIN+".json")
    require(sha(source)==manifest["models"][MODEL_ORIGIN],
            "historical v2 warmstart SHA mismatch")
    torch.manual_seed(TRAIN_SEED)
    model=RNADNetwork(state_features=STATE_FEATURES_V3,
                      action_features=ACTION_FEATURES_V3)
    origin=migrate_v2_export_to_v3(model,source)
    require(origin==manifest["models"][MODEL_ORIGIN],
            "initialization didn't reuse the exact original model")
    target=series/"initial-model.json"
    export_json(model,target,feature_schema=THREE_PHASE_FEATURE_SCHEMA_V3)
    return source, sha(target), sha(source)

def train_controlled(*,updates,model_dir,out,matches=TRAIN_MATCHES):
    require(updates in (1,4) and type(updates)==int,
            "arm update count must be one or four")
    require(type(matches)==int and 1<=matches<=TRAIN_MATCHES,
            "invalid preregistered training budget")
    torch.set_num_threads(1)
    series=Path(out)
    series.mkdir(parents=True,exist_ok=True)
    model,initial_sha,origin_sha=initial_exact(model_dir,series)
    old_model,pool,pool_sha=prepare_models(model_dir,"unused-v5",series,shard=2)
    require(Path(old_model)==Path(model) and
            sha(pool)==pool_sha,"v3 three-frozen-opponent pool changed")
    current=current_git_sha()
    identity={
        "schema":SCHEMA,"training_source_sha":current,
        "engine_sha":frozen_engine_sha(),
        "initial_seed":TRAIN_SEED,"target_complete_matches":matches,
        "initial_v2_origin_sha256":origin_sha,
        "initial_v3_model_sha256":initial_sha,
        "frozen_opponent_pool_sha256":pool_sha,
        "initial_model":"c005-v2-to-v3",
        "eta":0.2,"magnet_period":4,
        "trainer_source":"scripts/rnad/train.py",
        "optimizer_updates_per_match":updates,
        "same_accepted_match_seed_stream_for_both_arms":True,
        "same_model_weights_and_frozen_opponents_for_both_arms":True,
        "use_same_unseen_holdout_seeds":True,
        "training_during_heldout_evaluation":False,
        "champion_promoted":False,"main_merged":False,
    }
    ident=series/"controlled-identity.json"
    require(not ident.exists(),"refuse to replace a previous preregistered writer")
    ident.write_text(json.dumps(identity,indent=2)+"\n")
    journal=series/"controlled-training-journal.ndjson"
    for i in range(matches):
        result=train(seed=TRAIN_SEED,matches=1,out=str(series),
                     resume=i>0,initial_model=str(old_model),
                     three_phase=True,features_v3=True,
                     migrate_from_v2=True,
                     switch_guard="anti-loop-v1",
                     switch_cost="none",
                     eta=0.2,magnet_period=4,
                     updates_per_match=updates,
                     league_pool=str(pool))
        index=json.loads((series/"checkpoints/latest.json").read_text())
        md=index["metadata"]
        require(md["completed_matches"]==i+1 and
                md["updates"]==(i+1)*updates and
                md["initial_seed"]==TRAIN_SEED and
                md["next_match_seed"]==TRAIN_SEED+i+1 and
                md["initialization_sha256"]==origin_sha and
                md["league_pool_sha256"]==pool_sha and
                md["source_sha"]==current and
                md["training_code_sha"]==current and
                md["observation_schema_version"]==THREE_PHASE_FEATURE_SCHEMA_V3 and
                md["optimizer_updates_per_match"]==updates and
                sha(series/"checkpoints"/index["checkpoint_file"])==
                    index["checkpoint_sha256"] and
                len(md["completed_match_ids"])==i+1 and
                len(set(md["completed_match_ids"]))==i+1,
                "wrong accepted complete Match, origin, optimizer updates or state")
        row={"complete_matches":i+1,"matches_seed":TRAIN_SEED+i,
             "optimizer_updates":md["updates"],
             "model_sha256":sha(series/"final-model.json"),
             "checkpoint_sha256":index["checkpoint_sha256"]}
        with journal.open("a") as f:
            f.write(json.dumps(row)+"\n")
        _safe_cleanup(series)
        if (i+1)%64==0:
            print(json.dumps({"research_controlled_updates":updates,
                              "accepted_complete_matches":i+1,
                              "target":matches,
                              "optimizer_updates":md["updates"]}),flush=True)
    index=json.loads((series/"checkpoints/latest.json").read_text())
    result={
        **identity,"completed_matches":matches,
        "optimizer_updates":index["metadata"]["updates"],
        "final_model_sha256":sha(series/"final-model.json"),
        "final_checkpoint_sha256":index["checkpoint_sha256"],
        "accepted_match_ids_unique":
           len(set(index["metadata"]["completed_match_ids"]))==matches,
        "completed":True,
    }
    require(result["accepted_match_ids_unique"] and
            result["optimizer_updates"]==matches*updates,
            "incomplete controlled study")
    (series/"controlled-train-report.json").write_text(
        json.dumps(result,indent=2)+"\n")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--updates",type=int,required=True)
    p.add_argument("--model-dir",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--matches",type=int,default=TRAIN_MATCHES)
    args=p.parse_args()
    print(json.dumps(train_controlled(updates=args.updates,
        model_dir=args.model_dir,out=args.out,matches=args.matches)),
        flush=True)
if __name__=="__main__":
    main()
