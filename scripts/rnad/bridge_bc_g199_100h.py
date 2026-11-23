"""Source-pin the completed C-v6 G198 research checkpoint for the B+C 100h study.

No training here. Input is a complete, SHA-pinned, read-only, four-win
Match generation handoff artifact. On success its head weights and full
optimizer/RNG states are unchanged; only the NEW source identity is resealed.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path
import torch

from scripts.rnad.g137_strength_resume_migration import verify_pins, same
from scripts.rnad.shared_league_bootstrap import digest, write_json, POLICY_ID
from scripts.rnad.shared_league_contract import verify_manifest, canonical_hash
from scripts.rnad.shared_league_handoff import _seal_bundle

EXPECTED_SOURCE = "a98bd7cf73b7d3fc2e7d4a9b81764833843653a7"
EXPECTED_GENERATION = 199
EXPECTED_SEED = 9900001 + EXPECTED_GENERATION * 640


def check(ok, why):
    if not ok:
        raise ValueError("B/C source-bridge: " + why)


def read(path):
    return json.loads(Path(path).read_text())


def bridge(src, dst):
    src, dst = Path(src), Path(dst)
    check(src.is_dir() and not dst.exists(), "Source missing or output exists")
    pins = verify_pins(src)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, timeout=8).strip()
    check(len(head) == 40 and head != EXPECTED_SOURCE,
          "No new source identity to seal")
    old = read(src / "generation-manifest.json")
    verify_manifest(old)
    check(old["generation"] == EXPECTED_GENERATION and
          old["seed_start"] == EXPECTED_SEED and
          old["source_sha"] == EXPECTED_SOURCE and
          old["policy_id"] == POLICY_ID and
          old.get("league_opponent_profile") is None and
          old.get("battle_feature_schema", "rnad-three-phase-features-v5") ==
             "rnad-three-phase-features-v6" and
          not old["opponents"]["exploiter"], "Wrong frozen C-v6 G198→G199 checkpoint")
    engine = old["engine_sha"]
    index = read(src / "checkpoint-index.json")
    checkpoint_path = src / "checkpoint-g0.pt"
    check(checkpoint_path.is_file() and
          digest(checkpoint_path) == index.get("sha256") and
          index.get("generation") == EXPECTED_GENERATION,
          "Source optimizer checkpoint digest or generation invalid")
    previous = torch.load(checkpoint_path,
                          map_location="cpu", weights_only=True)
    keys = ("model", "target", "magnet", "optimizer", "scheduler",
            "torch_rng", "python_rng", "completed_generation_match_ids")
    check(previous.get("schema") == "rnad-shared-learner-full-state-v1" and
          previous.get("generation") == EXPECTED_GENERATION and
          previous.get("step") == 0 and previous.get("is_champion") is False and
          previous.get("source_sha") == EXPECTED_SOURCE and
          previous.get("engine_sha") == engine and
          previous.get("main_sha256") == old["main_sha256"] and
          previous.get("magnet_sha256") == old["magnet_sha256"] and
          previous.get("manifest_sha256") == canonical_hash(old) and
          isinstance(previous.get("training_match_ids"),list) and
          not previous["training_match_ids"] and
          all(k in previous for k in keys),
          "Untrusted or incomplete full-state research checkpoint")
    check(len(previous["completed_generation_match_ids"])==1280 and
          len(set(previous["completed_generation_match_ids"]))==1280,
          "Missing previously completed original 1280 accepted Matches")
    tmp=dst.with_name(dst.name+".incomplete")
    check(not tmp.exists(),"Previous partial migration path exists")
    shutil.copytree(src,tmp)
    try:
        verified = {}
        def reseal(path, expected_generation, old_sha):
            original = tmp / path
            sealed = tmp / (str(path) + ".resealed")
            digest_ = _seal_bundle(original, sealed,
                expected_generation=expected_generation,
                old_sha=old_sha,new_sha=head,engine=engine)
            shutil.rmtree(original)
            sealed.rename(original)
            return digest_
        main_sha=reseal(Path("actor"),EXPECTED_GENERATION,EXPECTED_SOURCE)
        magnet_sha=reseal(Path("magnet"),EXPECTED_GENERATION,EXPECTED_SOURCE)
        history=[]
        for entry in old["opponents"]["historical"]:
            name=entry["id"]
            check(name.startswith("past-g") and
                  name.removeprefix("past-g").isdigit(),
                  "Untrusted historical opponent registry ID")
            generation=int(name.removeprefix("past-g"))
            folder=Path("past") / name
            path=tmp / folder / "manifest.json"
            check(digest(path)==entry["sha256"],
                  "Historical weights no longer match exact manifest SHA")
            original_sha=read(path)["source_sha"]
            hist_sha=reseal(folder,generation,original_sha)
            history.append({"id":name,"sha256":hist_sha})
        for entry in old["opponents"]["fixed"]:
            ident=entry["id"]
            check(ident.startswith("fixed-") and
                  ident.removeprefix("fixed-").isalnum(),
                  "Unexpected fixed model ID")
            p=tmp/"fixed"/(ident.removeprefix("fixed-")+"-v3.json")
            check(p.is_file() and digest(p)==entry["sha256"],
                  "Fixed legacy opponent changed")
        manifest={**old,"source_sha":head,
                  "main_sha256":main_sha,"policy_manifest_sha256":main_sha,
                  "magnet_sha256":magnet_sha,
                  "opponents":{**old["opponents"],"historical":history}}
        manifest_sha=verify_manifest(manifest)
        (tmp/"generation-manifest.json").unlink()
        write_json(tmp/"generation-manifest.json",manifest)
        new_cp=copy.deepcopy(previous)
        new_cp["source_sha"]=head
        new_cp["main_sha256"]=main_sha
        new_cp["magnet_sha256"]=magnet_sha
        new_cp["manifest_sha256"]=manifest_sha
        new_cp["research_source_previous_sha"]=EXPECTED_SOURCE
        new_cp["research_source_previous_checkpoint_sha"]=digest(checkpoint_path)
        (tmp/"checkpoint-g0.pt").unlink()
        torch.save(new_cp,tmp/"checkpoint-g0.pt")
        check(all(same(previous[k],new_cp[k]) for k in keys),
              "Model/Target/Magnet/Optimizer/Scheduler/RNG drift in bridge")
        idx=read(tmp/"checkpoint-index.json")
        idx["sha256"]=digest(tmp/"checkpoint-g0.pt")
        (tmp/"checkpoint-index.json").unlink()
        write_json(tmp/"checkpoint-index.json",idx)
        context_path=tmp/"continuation-context.json"
        if context_path.exists():
            context=read(context_path)
            check(context.get("new_continuation_source_sha")==EXPECTED_SOURCE,
                  "Continuation context was not previous source")
            began=int(time.time())
            context.update({"new_continuation_source_sha":head,
                "start_epoch":began,"deadline_epoch":began+360000,
                "budget_seconds":360000,
                "source_run_id":44897382403,
                "requested_approx_hours":100,
                "new_budget_phase":"B+C combined 100h research",
                "original_generation":EXPECTED_GENERATION})
            context_path.unlink()
            write_json(context_path,context)
        report={
            "schema":"rnad-i5-bc-source-bridge-v1",
            "old_source_sha":EXPECTED_SOURCE,
            "new_source_sha":head,"generation":EXPECTED_GENERATION,
            "old_full_checkpoint_sha256":digest(checkpoint_path),
            "new_full_checkpoint_sha256":idx["sha256"],
            "unchanged_state_keys":list(keys),
            "initial_file_pin_count":pins,
            "manifest_sha256":manifest_sha,"research_only":True,
            "champion_promoted":False}
        write_json(tmp/"bc-source-bridge-report.json",report)
        oldpins=tmp/"file-pins.sha256"
        oldpins.unlink()
        files=sorted((str(p.relative_to(tmp)),digest(p))
                     for p in tmp.rglob("*") if p.is_file())
        oldpins.write_text("".join(f"{h}  {p}\n" for p,h in files))
        verify_pins(tmp)
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True)
        raise
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    print(json.dumps(bridge(args.source,args.out),sort_keys=True))
