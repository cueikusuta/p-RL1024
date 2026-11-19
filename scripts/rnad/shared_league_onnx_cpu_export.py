"""Immutable battle GRU ONNX export for an offline Actor throughput study.

The accepted Match authority is intentionally NOT changed: Node ONNX Runtime
InferenceSession.run is asynchronous, whereas the current authority API and
policy.choose are synchronous. Candidate only, no new Actor model promoted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.rnad.neural import (RNADNetwork, STATE_FEATURES_V5,
    ACTION_FEATURES_V5, THREE_PHASE_FEATURE_SCHEMA_V5, load_exported_json,
    export_onnx, compare_onnx)


def run(actor, out):
    torch.set_num_threads(1)
    source=Path(actor)
    root=Path(out)
    root.mkdir(parents=True,exist_ok=True)
    net=RNADNetwork(state_features=STATE_FEATURES_V5,
                    action_features=ACTION_FEATURES_V5).eval()
    pinned=load_exported_json(
        net,source,expected_feature_schema=THREE_PHASE_FEATURE_SCHEMA_V5)
    artifact=root/"research-battle-only.onnx"
    export_onnx(net,artifact)
    maxima=compare_onnx(net,artifact)
    result={
        "schema":"rnad-i5-battle-onnx-cpu-candidate-v1",
        "source_actor_battle_sha256":pinned,
        "onnx_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "torch_onnx_max_abs":maxima,
        "offline_candidate_only":True,
        "authoritative_match_engine_changed":False,
        "champion_promoted":False,
    }
    (root/"manifest.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--actor-battle",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.actor_battle,a.out)
