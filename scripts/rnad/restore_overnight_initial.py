"""Reconstruct EXACT pre-overnight policy from source-pinned frozen weights.
Never approximate a training initialization or use a different checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from scripts.rnad.neural import (
    RNADNetwork, STATE_FEATURES_V3, ACTION_FEATURES_V3,
    STATE_FEATURES_V5, ACTION_FEATURES_V5,
    THREE_PHASE_FEATURE_SCHEMA_V3, THREE_PHASE_FEATURE_SCHEMA_V5,
    migrate_v2_export_to_v3, load_exported_json, export_json)
from scripts.rnad.night_league_2026 import (
    strategy, prepare_models, sha, SOURCE_ARCHIVE, SCHEMA)

ROOTS={"base128":0,"base256":1,"c005":2,"battle-focus256-v5":3}
RUN=44221678145

def restore(models,battle_v5,series,out,shard):
    config=strategy(shard)
    root=Path(series)
    ident=json.loads((root/"series-identity.json").read_text())
    checkpoint=json.loads((root/"checkpoints"/"latest.json").read_text())
    metadata=checkpoint["metadata"]
    assert ident["schema"]==SCHEMA and ident["shard"]==shard
    assert ident["series_source_sha"]=="833f4039b69bbad64dc7a28c059d4b934004e5bf"
    assert metadata["initial_seed"]==config["seed"]
    assert metadata["source_sha"]==ident["series_source_sha"]
    assert ident["origin"]==config["origin"]
    assert metadata["initialization_sha256"]==ident["initial_model_file_sha256"]
    original,pool,poolsha=prepare_models(models,battle_v5,root,shard)
    assert sha(original)==ident["initial_model_file_sha256"]
    assert sha(pool)==ident["frozen_pool_file_sha256"]
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    v5=config["mode"]=="selfplay"
    net=(RNADNetwork(state_features=STATE_FEATURES_V5,
                     action_features=ACTION_FEATURES_V5) if v5
         else RNADNetwork(state_features=STATE_FEATURES_V3,
                          action_features=ACTION_FEATURES_V3))
    schema=THREE_PHASE_FEATURE_SCHEMA_V5 if v5 else THREE_PHASE_FEATURE_SCHEMA_V3
    source_sha=(load_exported_json(net,original,expected_feature_schema=schema)
                if v5 else migrate_v2_export_to_v3(net,original))
    assert source_sha==metadata["initialization_sha256"]
    output=Path(out)
    output.mkdir(parents=True,exist_ok=False)
    result=output/"initial-model.json"
    export_json(net,result,feature_schema=schema)
    digest=sha(result)
    summary={"schema":"rnad-i5-exact-overnight-initial-model-v1",
        "training_run":RUN,"shard":shard,"mode":config["mode"],
        "origin":config["origin"],"initial_seed":config["seed"],
        "source_sha":metadata["source_sha"],
        "source_model_sha256":source_sha,
        "initialization_sha256":metadata["initialization_sha256"],
        "restored_model_sha256":digest,
        "model_feature_schema":schema,"pre_learning_updates":0,
        "reference_post_learning_matches":metadata["completed_matches"],
        "champion_promoted":False}
    (output/"manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("models","battle-v5","series","out"):
        p.add_argument("--"+k,required=True)
    p.add_argument("--shard",type=int,required=True)
    a=p.parse_args()
    restore(a.models,a.battle_v5,a.series,a.out,a.shard)
