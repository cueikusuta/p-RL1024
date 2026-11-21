"""Seal the four-opponent B research profile from an already source-pinned v5 bootstrap.

Never mutate the source bootstrap, completed checkpoint, or current study Run.
The caller must first bridge the frozen full state to THIS code SHA by the
existing authenticated handoff; this step changes only the opponent registry.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import torch

from scripts.rnad.shared_league_contract import (
    PROFILE_B, PROFILE_B_FIXED, verify_manifest, canonical_hash,
)
from scripts.rnad.shared_league_bootstrap import digest, write_json
from scripts.rnad.train import current_git_sha

AQ_SHA = "5ccc11a3362c7f5a8874847e3afc1f19c539d4ef258bd82704f53d8a1db1946e"
B_SCHEMA = "rnad-i5-authorized-fixed-opponent-v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare(src, dst):
    src,dst=Path(src),Path(dst)
    if not src.is_dir() or dst.exists():
        raise ValueError("Only an existing pinned source and new output path are valid")
    old=read_json(src/"generation-manifest.json")
    verify_manifest(old)
    if old.get("league_opponent_profile") is not None or (
        old["source_sha"] != current_git_sha() or old["opponents"]["exploiter"]):
        raise ValueError("B preparation requires a source-pinned unused legacy profile")
    existing={e["id"]:e for e in old["opponents"]["fixed"]}
    if not all("fixed-"+x in existing for x in ("base256","c005")):
        raise ValueError("Missing immutable baseline fixed opponents")
    for name in ("base256","c005"):
        p=src/"fixed"/(name+"-v3.json")
        if not p.is_file() or digest(p)!=existing["fixed-"+name]["sha256"]:
            raise ValueError("Legacy fixed opponent drift: "+name)
    for directory,key in (("actor","main_sha256"),("magnet","magnet_sha256")):
        p=src/directory/"manifest.json"
        if not p.is_file() or digest(p)!=old[key]:
            raise ValueError("Model/manifest SHA mismatch: "+directory)
    ckpt_path=src/"checkpoint-g0.pt"
    index=read_json(src/"checkpoint-index.json")
    if not ckpt_path.is_file() or index.get("sha256")!=digest(ckpt_path):
        raise ValueError("Unpinned input checkpoint")
    ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=True)
    if ckpt.get("schema")!="rnad-shared-learner-full-state-v1" or (
        ckpt.get("source_sha")!=old["source_sha"] or
        ckpt.get("manifest_sha256")!=canonical_hash(old) or
        ckpt.get("main_sha256")!=old["main_sha256"] or
        ckpt.get("magnet_sha256")!=old["magnet_sha256"] or
        ckpt.get("generation")!=old["generation"] or
        ckpt.get("is_champion") is not False or ckpt.get("step")!=0):
        raise ValueError("Prior full-state bootstrap not source pinned")
    source=Path("src/ai/draft-core-trio.js")
    if not source.is_file():
        raise ValueError("Missing reviewed Core-Trio source")
    core_sha=digest(source)
    temporary=dst.with_name(dst.name+".incomplete")
    if temporary.exists():
        raise ValueError("Refusing stale incomplete B staging directory")
    shutil.copytree(src,temporary)
    try:
        fixed=temporary/"fixed"
        fresh=[]
        for kind in ("aqbase001","coretrioaq001"):
            name="fixed-"+kind
            p=fixed/(kind+"-v1.json")
            write_json(p,{
                "schema":B_SCHEMA,"kind":kind,
                "aq_model_sha256":AQ_SHA,
                "core_trio_source_sha256":core_sha,
                "epsilon":0.02,
                "source_git_sha":old["source_sha"],
                "research_only":True,
            })
            fresh.append({"id":name,"sha256":digest(p)})
        fresh.extend(copy.deepcopy(existing["fixed-"+n]) for n in ("base256","c005"))
        updated=copy.deepcopy(old)
        updated["opponents"]["fixed"]=fresh
        updated["opponents"]["exploiter"]=[]
        updated["league_opponent_profile"]=PROFILE_B
        assert {e["id"] for e in fresh}==set(PROFILE_B_FIXED)
        manifest_sha=verify_manifest(updated)
        (temporary/"generation-manifest.json").unlink()
        write_json(temporary/"generation-manifest.json",updated)
        ckpt["manifest_sha256"]=manifest_sha
        torch.save(ckpt,temporary/"checkpoint-g0.pt")
        index["sha256"]=digest(temporary/"checkpoint-g0.pt")
        (temporary/"checkpoint-index.json").unlink()
        write_json(temporary/"checkpoint-index.json",index)
        report_path=temporary/"handoff-report.json"
        if report_path.exists():
            report=read_json(report_path)
            report["new_generation_manifest_sha256"]=manifest_sha
            report["new_full_checkpoint_sha256"]=index["sha256"]
            report["b_profile_prepared"]=True
            report_path.unlink()
            write_json(report_path,report)
        pins=temporary/"file-pins.sha256"
        if pins.exists():pins.unlink()
        lines=sorted(
            f"{digest(p)}  {p.relative_to(temporary).as_posix()}"
            for p in temporary.rglob("*") if p.is_file()
        )
        pins.write_text("\n".join(lines)+"\n")
        temporary.rename(dst)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
    return {
        "generation":old["generation"],"source_sha":old["source_sha"],
        "manifest_sha256":manifest_sha,"fixed_ids":[e["id"] for e in fresh],
        "checkpoint_sha256":index["sha256"],
    }


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    print(json.dumps(prepare(args.src,args.out),sort_keys=True))
