"""Verify provenance and exact floated weights in an archived RNAD warmstart run.

Read-only verification. Requires the full immutable completed-Match PyTorch
checkpoint file, independently validates bytes, optimizer/scheduler state and
exported-policy weights against the versioned SHA256 manifest.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def tensor_hash(model_path):
    obj=json.loads(Path(model_path).read_text())
    if (obj.get("schema")!="rnad-small-gru-v1" or
        obj.get("state_features")!=32 or
        obj.get("action_features")!=16 or
        obj.get("hidden")!=128 or
        not isinstance(obj.get("weights"),dict) or
        len(obj["weights"])!=14):
        raise ValueError("Wrong exported model or model dimensions")
    h=hashlib.sha256()
    for name,item in sorted(obj["weights"].items()):
        shape=item["shape"]
        values=item["values"]
        if (not isinstance(name,str) or not isinstance(shape,list) or
            not all(type(x)==int and x>0 for x in shape) or
            not isinstance(values,list) or
            math.prod(shape)!=len(values) or
            not all(type(x) in (int,float) and math.isfinite(x) for x in values)):
            raise ValueError("Invalid model weight data")
        # Existing train.state_hash() uses the *literal two-byte* separator
        # backslash + ASCII zero, not a NUL. Preserve historical digests.
        sep=bytes([92,48])
        h.update(name.encode("utf8")+sep)
        h.update(str(tuple(shape)).encode("ascii")+sep)
        h.update(struct.pack("<"+"f"*len(values),*values))
    return h.hexdigest()


def verify(root,source_sha):
    root=Path(root)
    research=json.loads((root/"warmstart-report.json").read_text())
    teacher=json.loads((root/"teacher"/"teacher-report.json").read_text())
    selfplay=json.loads((root/"selfplay"/"report.json").read_text())
    manifest=json.loads((root/"selfplay"/"checkpoints"/"latest.json").read_text())
    teacher_file=root/"teacher"/"trained-model.json"
    rnad_file=root/"selfplay"/"final-model.json"
    exact_init_sha=hashlib.sha256(teacher_file.read_bytes()).hexdigest()
    if (research.get("schema")!="rnad-optional-warmstart-256-research-v1" or
        research.get("source_sha")!=source_sha or
        teacher.get("source_sha")!=source_sha or
        selfplay.get("source_sha")!=source_sha or
        research.get("teacher_matches")!=32 or
        research.get("teacher_updates")!=128 or
        research.get("selfplay_matches")!=256 or
        research.get("rnad_updates")!=256 or
        research.get("is_champion") is not False or
        research.get("initialization_json_sha256")!=exact_init_sha or
        selfplay.get("initialization_sha256")!=exact_init_sha or
        manifest.get("metadata",{}).get("initialization_sha256")!=exact_init_sha or
        manifest.get("metadata",{}).get("updates")!=256 or
        research.get("accepted_actions")!=manifest["metadata"].get("accepted_decisions") or
        tensor_hash(teacher_file)!=research.get("teacher_initial_model_sha256") or
        tensor_hash(rnad_file)!=research.get("rnad_trained_sha256") or
        selfplay.get("model_sha256")!=research.get("rnad_trained_sha256")):
        raise AssertionError("Invalid archived RNAD model/teacher/provenance")
    # A trained-policy export is NOT a restorable training checkpoint.
    # Verify the exact on-disk PyTorch payload, its optimizer state and
    # the identical Policy weights. No arbitrary pickle loading:
    # restore_checkpoint uses torch.load(..., weights_only=True).
    import copy
    import re
    import torch
    from scripts.rnad.neural import RNADNetwork, freeze_magnet
    from scripts.rnad.checkpoint import restore_checkpoint
    from scripts.rnad.train import state_hash
    ckpt=root/"selfplay"/"checkpoints"
    name=manifest.get("checkpoint_file","")
    if not isinstance(name,str) or not re.fullmatch(
            r"checkpoint-u256-[0-9a-f]{64}\.pt",name):
        raise AssertionError("Missing completed Match checkpoint filename")
    snapshot=ckpt/name
    if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=manifest.get("checkpoint_sha256"):
        raise AssertionError("Missing or corrupted full optimizer checkpoint")
    torch.set_num_threads(1)
    net=RNADNetwork()
    target=copy.deepcopy(net)
    magnet=freeze_magnet(net)
    optimizer=torch.optim.Adam(net.parameters(),lr=3e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=.98)
    restored=restore_checkpoint(
        ckpt,policy=net,value=target,magnet=magnet,
        optimizer=optimizer,scheduler=scheduler,restore_rng=False,
        expected={
            "model_id":"RNAD-001",
            "source_sha":source_sha,
            "training_code_sha":source_sha,
            "engine_snapshot_sha":research["engine_sha"],
            "observation_schema_version":"rnad-battle-features-v1",
            "initialization_sha256":exact_init_sha})
    if (state_hash(net)!=research["rnad_trained_sha256"] or
        restored.get("checkpoint_sha256")!=manifest["checkpoint_sha256"] or
        not optimizer.state or scheduler.last_epoch!=256):
        raise AssertionError("Saved optimizer / Policy / schedule cannot resume")
    return {
        "schema":"rnad-verified-archived-evaluation-v1",
        "source_sha":source_sha,
        "teacher_accepted_matches":32,
        "rnad_completed_matches":256,
        "rnad_accepted_actions":research["accepted_actions"],
        "teacher_model_sha256":research["teacher_initial_model_sha256"],
        "rnad_model_sha256":research["rnad_trained_sha256"],
        "evaluation_only":True,
        "optimizer_state_verified":True,
        "champion":False
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--source-sha",required=True)
    args=p.parse_args()
    print(json.dumps(verify(args.root,args.source_sha),indent=2))
