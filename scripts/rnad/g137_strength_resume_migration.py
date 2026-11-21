"""Audited G137 full-state migration after completed G136; explicitly user-approved continuation.

Only SHA-pinned actor manifests and checkpoint provenance are rewritten.
Main/Target/Magnet/optimizer/scheduler/RNG tensors and prior seeds are immutable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import torch
from scripts.rnad.shared_league_bootstrap import digest, write_json, POLICY_ID
from scripts.rnad.shared_league_contract import canonical_hash, verify_manifest
from scripts.rnad.shared_league_handoff import _seal_bundle, ARTIFACT_SCHEMA

PINNED_OLD_SOURCE = "26bcb35e6a1e7e74daa7e05c74c4b381c751188b"
PINNED_ENGINE = "ea81165993110ee9fb3a09dcb97c6708c0d7397b"
PINNED_GENERATION = 137
PINNED_SEED = 9900001 + PINNED_GENERATION * 640
STUDY_BUDGET_SECONDS = 36000


def need(ok, message):
    if not ok:
        raise ValueError("G137 resume migration: " + message)


def read_json(path):
    return json.loads(Path(path).read_text())


def same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b


def verify_pins(root):
    entries = (root / "file-pins.sha256").read_text().splitlines()
    observed = set()
    for line in entries:
        parts = line.split("  ", 1)
        need(len(parts) == 2 and len(parts[0]) == 64, "Malformed pinned digest list")
        expected, relative = parts
        name = Path(relative)
        need(not name.is_absolute() and ".." not in name.parts and str(name) not in observed, "Unsafe or duplicate pin")
        observed.add(str(name))
        need((root / name).is_file() and digest(root / name) == expected, "Source pin mismatch: " + relative)
    actual = {str(x.relative_to(root)) for x in root.rglob("*") if x.is_file() and x.name != "file-pins.sha256"}
    need(observed == actual, "Unpinned or absent source bundle file")
    return len(observed)


def migrate(*, source, out):
    src, dst = Path(source), Path(out)
    need(src.is_dir() and not dst.exists(), "Missing source or destination already exists")
    pinned = verify_pins(src)
    current_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    need(len(current_sha)==40 and current_sha != PINNED_OLD_SOURCE, "New research code source not pinned")
    old = read_json(src / "generation-manifest.json")
    verify_manifest(old)
    need(old["generation"] == PINNED_GENERATION and old["seed_start"] == PINNED_SEED and
         old["source_sha"] == PINNED_OLD_SOURCE and old["engine_sha"] == PINNED_ENGINE and
         old["policy_id"] == POLICY_ID and not old["opponents"]["exploiter"], "Wrong completed G136 source provenance")
    need([r["id"] for r in old["opponents"]["historical"]] == ["past-g134", "past-g135", "past-g136"], "Unexpected history")
    need(len(old["opponents"]["fixed"]) == 3, "Frozen benchmark pool missing")
    index=read_json(src/"checkpoint-index.json")
    prev_file=src/"checkpoint-g0.pt"
    old_ckpt_sha=digest(prev_file)
    need(index["sha256"] == old_ckpt_sha and index["generation"] == PINNED_GENERATION, "G137 checkpoint SHA mismatch")
    cp=torch.load(prev_file, map_location="cpu", weights_only=True)
    need(cp["generation"] == PINNED_GENERATION and cp["step"] == 0 and
         cp["source_sha"] == PINNED_OLD_SOURCE and cp["engine_sha"] == PINNED_ENGINE and
         cp["manifest_sha256"] == canonical_hash(old) and
         cp["main_sha256"] == old["main_sha256"] and
         cp["magnet_sha256"] == old["magnet_sha256"] and
         cp.get("cumulative_optimizer_updates") == 137*40 and
         cp.get("cumulative_complete_matches") == 137*1280 and
         len(cp["completed_generation_match_ids"]) == 1280 and
         not cp["training_match_ids"] and cp["is_champion"] is False, "Incomplete historic checkpoint")
    preserved=("model","target","magnet","optimizer","scheduler","torch_rng","python_rng","completed_generation_match_ids")
    need(all(k in cp for k in preserved), "Incomplete optimizer/magnet/RNG state")

    dst.mkdir(parents=True)
    main_sha=_seal_bundle(src/"actor", dst/"actor",
                          expected_generation=PINNED_GENERATION,old_sha=PINNED_OLD_SOURCE,
                          new_sha=current_sha,engine=PINNED_ENGINE)
    magnet_sha=_seal_bundle(src/"magnet", dst/"magnet",
                            expected_generation=PINNED_GENERATION,old_sha=PINNED_OLD_SOURCE,
                            new_sha=current_sha,engine=PINNED_ENGINE)
    new_history=[]
    for i in (134,135,136):
        name=f"past-g{i}"
        row=next(r for r in old["opponents"]["historical"] if r["id"]==name)
        need(digest(src/"past"/name/"manifest.json")==row["sha256"], "Untrusted historical Head")
        h=_seal_bundle(src/"past"/name, dst/"past"/name,
                       expected_generation=i,old_sha=PINNED_OLD_SOURCE,
                       new_sha=current_sha,engine=PINNED_ENGINE)
        new_history.append({"id":name,"sha256":h})
    for row in old["opponents"]["fixed"]:
        ident=row["id"]
        need(ident.startswith("fixed-") and ident.removeprefix("fixed-").isalnum(), "Unapproved fixed opponent")
        name=ident.removeprefix("fixed-")+"-v3.json"
        source_file=src/"fixed"/name
        need(digest(source_file)==row["sha256"], "Frozen opponent modified")
        target=dst/"fixed"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_file,target)
        need(digest(target)==row["sha256"], "Copied opponent differs")
    for name in ("original-draft.json","original-selection-js.json"):
        shutil.copyfile(src/name,dst/name)
        need(digest(src/name)==digest(dst/name), "Source expert template modified")
    proposed={**old, "source_sha":current_sha, "main_sha256":main_sha,
              "policy_manifest_sha256":main_sha, "magnet_sha256":magnet_sha,
              "opponents":{"historical":new_history,"fixed":old["opponents"]["fixed"],"exploiter":[]}}
    new_manifest_sha=verify_manifest(proposed)
    write_json(dst/"generation-manifest.json",proposed)

    old_context=read_json(src/"continuation-context.json")
    need(old_context["schema"]==ARTIFACT_SCHEMA and
         old_context["new_continuation_source_sha"]==PINNED_OLD_SOURCE and
         old_context["source_run_id"]==44318145987 and
         old_context["budget_seconds"]==STUDY_BUDGET_SECONDS,
         "Wrong previous wave context")
    started=int(time.time())
    context={**old_context,
       "new_continuation_source_sha":current_sha,
       "new_budget_phase":"g137-user-approved-ten-hour-continuation",
       "original_generation":PINNED_GENERATION,
       "start_epoch":started,
       "deadline_epoch":started+STUDY_BUDGET_SECONDS,
       "previous_budget_context_sha256":digest(src/"continuation-context.json")}
    write_json(dst/"continuation-context.json",context)

    ckpt=dict(cp)
    ckpt["resume_previous_source_sha"]=PINNED_OLD_SOURCE
    ckpt["resume_previous_checkpoint_sha256"]=old_ckpt_sha
    ckpt["source_sha"]=current_sha
    ckpt["main_sha256"]=main_sha
    ckpt["magnet_sha256"]=magnet_sha
    ckpt["manifest_sha256"]=new_manifest_sha
    torch.save(ckpt,dst/"checkpoint-g0.pt")
    restored=torch.load(dst/"checkpoint-g0.pt",map_location="cpu",weights_only=True)
    need(all(same(cp[k],restored[k]) for k in preserved), "Full-state tensors or RNG changed across source migration")
    need(restored["source_sha"]==current_sha and
         restored["manifest_sha256"]==new_manifest_sha and
         restored["main_sha256"]==main_sha, "Migrated provenance not anchored")
    write_json(dst/"checkpoint-index.json",{
      **index,"sha256":digest(dst/"checkpoint-g0.pt"),
      "prior_generation_checkpoint_sha256":old_ckpt_sha,
      "research_only":True,"is_champion":False})
    write_json(dst/"migration-report.json",{
      "schema":"rnad-i5-g137-strength-resume-migration-v1",
      "old_checkpoint_sha256":old_ckpt_sha,
      "new_checkpoint_sha256":digest(dst/"checkpoint-g0.pt"),
      "source_sha":current_sha,"previous_source_sha":PINNED_OLD_SOURCE,
      "generation":136,"checked_file_pins":pinned,
      "verified_model_target_magnet_optimizer_scheduler_rng":True,
      "historical_opponents":[r["id"] for r in new_history],
      "research_only":True,"champion_promoted":False})
    pins=sorted((str(p.relative_to(dst)),digest(p)) for p in dst.rglob("*") if p.is_file())
    (dst/"file-pins.sha256").write_text("".join(f"{h}  {p}\n" for p,h in pins))
    print(json.dumps(read_json(dst/"migration-report.json")),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--out",required=True)
    a=parser.parse_args()
    migrate(source=a.source,out=a.out)
