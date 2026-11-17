"""Export one SHA-registered frozen A selection actor for JS read-only inference.

This does NOT learn, promote, or export any teacher target as an inference
input. Include one actor-only parity fixture from the reused development block.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from scripts.rnad.selection_ab import (
    DecomposedTeacher,pack,read_events,SOURCE_TEACHER
)

SCHEMA="rnad-i5-A-public-actor-js-weights-v1"

def export(study_file,model_dir,dev_dir,family,out):
    torch.set_num_threads(1)
    study=json.loads(Path(study_file).read_bytes())
    if family=="A1024":
        if study.get("schema")!="rnad-i5-selection-A-1024-supervised-study-v1":
            raise ValueError("Invalid old A1024 study")
        chosen=study["selected_checkpoint_before_new_holdout"]
        row=study["trials"][chosen]
        source=study["teacher_source_sha"]
    elif family=="A2048":
        if study.get("schema")!="rnad-i5-selection-A-2048-supervised-study-v1":
            raise ValueError("Invalid new A2048 study")
        chosen=study["preselected_model"]
        row=study["models"][chosen]
        source=study["teacher_source_sha"]
    else:
        raise ValueError("Only frozen A1024 and A2048 can be exported")
    league_key="league_started" if family=="A1024" else "league_training_started"
    if (row["filename"]!=chosen or
        row["seed"] not in (90251,90252,90253) or
        not chosen.endswith(".pt") or
        study.get(league_key) is not False or
        study.get("champion_promoted") is not False):
        raise ValueError("Unexpected frozen model identity / policy")
    path=Path(model_dir)/chosen
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=row["sha256"]:
        raise ValueError("Model bytes are not pinned by study SHA")
    vocab=study["vocab"]
    model=DecomposedTeacher(vocab)
    state=torch.load(path,map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True)
    model.eval()
    examples,dev_vocab,engine=read_events(
        dev_dir,2472001,1,expected_sha=source)
    if (not examples or dev_vocab!=vocab or engine!=study["engine_sha"]):
        raise ValueError("Mismatch historical authority-accepted actor fixture")
    row_actor=examples[0][1]["actor"]
    with torch.no_grad():
        logits,_,_=model(pack([examples[0]]))
    tensors={}
    for k,t in state.items():
        if not bool(torch.isfinite(t).all()):
            raise ValueError("Nonfinite exported frozen tensor")
        tensors[k]={"shape":list(t.shape),"values":t.reshape(-1).tolist()}
    payload={
        "schema":SCHEMA,
        "source_model_filename":chosen,
        "source_model_sha256":digest,
        "source_sha":source,
        "engine_sha":engine,
        "teacher":SOURCE_TEACHER,
        "public_actor_vocabulary":vocab,
        "labels_as_inference_inputs":False,
        "champion_promoted":False,
        "tensors":tensors,
        "parity_fixture":{"actor":row_actor,"logits":logits[0].tolist()}
    }
    path=Path(out)
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(payload,ensure_ascii=False,separators=(",",":"),
                    allow_nan=False)+"\n").encode("utf-8")
    path.write_bytes(raw)
    result={"family":family,"file":str(path),
            "source_model_sha256":digest,
            "json_sha256":hashlib.sha256(raw).hexdigest(),
            "parity_decisions":len(payload["parity_fixture"]["logits"])}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("study","model-dir","dev-dir","family","out"):
        p.add_argument("--"+k,required=True)
    a=p.parse_args()
    export(a.study,a.model_dir,a.dev_dir,a.family,a.out)
