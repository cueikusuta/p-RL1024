"""Verified, fail-closed generation-to-generation handoff for one shared R-NaD Main.

No optimizer is reinitialized, no weights are averaged, and no previous model
file is modified. Only the Actor bundle's source *manifest* is resealed for
the fixed continuation code SHA; all four model weight file SHA256s remain
exactly unchanged. Full state passes to the next generation unmodified except
for explicit provenance, completed-Match archival, and fresh generation IDs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import torch

from scripts.rnad.shared_league_bootstrap import (
    MODEL_SCHEMA, POLICY_ID, digest, write_json
)
from scripts.rnad.shared_league_contract import (
    ACTORS, MATCHES_PER_GENERATION, SEEDS_PER_ACTOR, SCHEMA, POLICY_SCHEMA,
    OPTIMIZER_BATCH_MATCHES, OPTIMIZER_STEPS, canonical_hash, verify_manifest
)

ARTIFACT_SCHEMA = "rnad-i5-shared-league-handoff-v1"
STATE_SCHEMA = "rnad-shared-learner-full-state-v1"
SEED_STRIDE = ACTORS * SEEDS_PER_ACTOR
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
BUNDLE_FILES = ("draft", "selection", "battle", "values")
HISTORY_DEPTH = 3
DEFAULT_BUDGET_SECONDS = 36000


def _assert(ok, message):
    if not ok:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()


def _engine_sha():
    return _read("provenance/manifest.json")["sourceCommit"]


def _check_full_match_ids(ids, previous):
    _assert(type(ids) is list and len(ids) == MATCHES_PER_GENERATION and
            len(set(ids)) == MATCHES_PER_GENERATION,
            "Prior generation lacks 1280 distinct accepted complete Match IDs")
    pattern = re.compile(r"g(\d+)-a(\d+)-s(\d+)-seat([01])\Z")
    expected = {f"g{previous['generation']}-a{actor}-s{seed}-seat{seat}"
                for actor in range(ACTORS)
                for seed in range(previous["seed_start"] + actor*SEEDS_PER_ACTOR,
                                  previous["seed_start"] + (actor+1)*SEEDS_PER_ACTOR)
                for seat in (0, 1)}
    _assert(set(ids) == expected and all(pattern.fullmatch(x) for x in ids),
            "Previous generation Match population missing, repeated or mislabeled")


def _copy_immutable_file(src, dst):
    src, dst = Path(src), Path(dst)
    _assert(src.is_file() and not dst.exists(),
            "Missing or overwritten immutable provenance file: " + str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    _assert(digest(src) == digest(dst), "Unexpected byte changes on copy")


def _seal_bundle(src, dst, *, expected_generation, old_sha, new_sha, engine):
    """Only rewrite source_commit in the immutable manifest; preserve weight bytes."""
    src, dst = Path(src), Path(dst)
    before = _read(src/"manifest.json")
    _assert(before.get("schema") == MODEL_SCHEMA and
            before.get("generation") == expected_generation and
            before.get("policy_id") == POLICY_ID and
            before.get("source_sha") == old_sha and
            before.get("engine_sha") == engine and
            set(before.get("files", {})) == set(BUNDLE_FILES),
            "Previous frozen Actor bundle identity is not verifiable")
    dst.mkdir(parents=True, exist_ok=False)
    for name in BUNDLE_FILES:
        filename = name + ".json"
        frozen = before["files"][name]
        _assert(isinstance(frozen, str) and HEX64.fullmatch(frozen) and
                digest(src/filename) == frozen,
                "Corrupt previous-generation frozen Head: " + name)
        _copy_immutable_file(src/filename, dst/filename)
        _assert(digest(dst/filename) == frozen,
                "Frozen Head bytes changed during generation handoff")
    after = dict(before)
    after["source_sha"] = new_sha
    write_json(dst/"manifest.json", after)
    _assert(_read(dst/"manifest.json")["files"] == before["files"],
            "Generation handoff modified model tensors")
    return digest(dst/"manifest.json")


def transition(*, previous_bootstrap, previous_result, out,
               budget_seconds=DEFAULT_BUDGET_SECONDS):
    prev_root, result_root, out_root = (
        Path(previous_bootstrap), Path(previous_result), Path(out))
    _assert(not out_root.exists(), "No overwriting generation handoff")
    old = _read(prev_root/"generation-manifest.json")
    verify_manifest(old)
    old_gen = old["generation"]
    new_gen = old_gen + 1
    _assert(new_gen <= 1000, "Bounded research generation exceeded")
    head, engine = _git_sha(), _engine_sha()
    _assert(HEX40.fullmatch(head) and old["engine_sha"] == engine,
            "Research code checkout or authority engine mismatch")
    old_sha = old["source_sha"]
    _assert(HEX40.fullmatch(old_sha),
            "Previous trained source SHA not registered")
    # The completed previous generation is an immutable input, not an
    # alternative bootstrap candidate that can be selected by its performance.
    index = _read(result_root/f"generation{new_gen}-index.json")
    report = _read(result_root/f"generation{new_gen}-report.json")
    _assert(index.get("schema") == STATE_SCHEMA and
            index.get("generation") == new_gen and
            index.get("optimizer_updates") == OPTIMIZER_STEPS and
            index.get("accepted_complete_matches") == MATCHES_PER_GENERATION and
            index.get("research_only") is True and
            index.get("champion_promoted") is False and
            index.get("source_manifest_sha256") == canonical_hash(old) and
            report.get("generation") == old_gen and
            report.get("optimizer_updates_completed") == OPTIMIZER_STEPS and
            report.get("accepted_complete_four_win_matches") ==
                MATCHES_PER_GENERATION and
            report.get("research_only") is True and
            report.get("champion_promoted") is False and
            report.get("single_writer_main_sha256") == index["main_sha256"] and
            report.get("frozen_magnet_sha256") == index["magnet_sha256"],
            "Previous research generation was incomplete or promoted")
    checkpoint_file = result_root/f"checkpoint-g{new_gen}.pt"
    _assert(HEX64.fullmatch(index.get("full_checkpoint_sha256", "")) and
            digest(checkpoint_file) == index["full_checkpoint_sha256"],
            "Prior full optimizer/target/magnet/RNG checkpoint SHA mismatch")
    ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    _assert(ckpt.get("schema") == STATE_SCHEMA and
            ckpt.get("policy_id") == POLICY_ID and
            ckpt.get("generation") == new_gen and
            ckpt.get("step") == 0 and
            ckpt.get("main_sha256") == index["main_sha256"] and
            ckpt.get("source_sha") == old_sha and
            ckpt.get("engine_sha") == engine and
            ckpt.get("is_champion") is False and
            isinstance(ckpt.get("model"), dict) and
            isinstance(ckpt.get("target"), dict) and
            isinstance(ckpt.get("magnet"), dict) and
            isinstance(ckpt.get("optimizer"), dict) and
            isinstance(ckpt.get("scheduler"), dict) and
            isinstance(ckpt.get("torch_rng"), torch.Tensor) and
            isinstance(ckpt.get("python_rng"), tuple),
            "Previous checkpoint is not a complete shared Learner state")
    _check_full_match_ids(ckpt.get("completed_generation_match_ids"), old)
    _assert("training_match_ids" in ckpt, "Missing prior training journal")
    if ckpt["training_match_ids"]:
        _check_full_match_ids(ckpt["training_match_ids"], old)
    _assert(index["main_sha256"] ==
            digest(result_root/f"generation{new_gen}-actor"/"manifest.json") and
            index["magnet_sha256"] ==
            digest(result_root/f"generation{new_gen}-magnet"/"manifest.json"),
            "Prior actor/magnet manifests do not match full-state report")
    _assert(old["seed_start"] + 2*SEED_STRIDE <= 0xffffffff,
            "Research seed plan exhausted")

    # One start time across all waves; never extend the deadline on restart.
    context_path = prev_root/"continuation-context.json"
    if context_path.exists():
        context = _read(context_path)
        _assert(context.get("schema") == ARTIFACT_SCHEMA and
                context.get("deadline_epoch", -1) >
                    context.get("start_epoch", 0) and
                context.get("budget_seconds") == budget_seconds and
                context.get("source_run_id") == 44318145987,
                "Prior wave time budget changed or reset")
    else:
        _assert(old_gen == 0,
                "Later generations may not silently restart wall-clock budget")
        started = int(time.time())
        context = {
            "schema": ARTIFACT_SCHEMA,
            "source_run_id": 44318145987,
            "budget_seconds": budget_seconds,
            "start_epoch": started,
            "deadline_epoch": started+budget_seconds,
            "original_generation": 1,
            "requested_approx_hours": 10,
            "original_league_training_source_sha": old_sha,
            "new_continuation_source_sha": head,
        }
    _assert(context["new_continuation_source_sha"] == head,
            "Continuation source code changed between waves")
    out_root.mkdir(parents=True, exist_ok=False)
    # All previous historical opponents remain frozen, and are sampled only
    # by immutable ID/SHA. Add the just-finished *previous Main* to the pool.
    prior_past = prev_root/"past"
    all_past = {}
    for entry in old["opponents"]["historical"]:
        name=entry["id"]
        if not name.startswith("past-g"):
            _assert(old_gen==0 and name=="integrated-initial-snapshot-v0",
                    "Unregistered previous historical pool")
            continue
        n=int(name.split("-g")[1])
        if n>=new_gen-HISTORY_DEPTH:
            src=prior_past/name
            _assert(digest(src/"manifest.json")==entry["sha256"],
                    "Previous historical registry SHA changed")
            all_past[n]=(src,old_sha)
    # Current generation Main was immutable for all previous 20 Actors.
    previous_actor_sha=old["main_sha256"]
    _assert(digest(prev_root/"actor"/"manifest.json")==previous_actor_sha,
            "Previous Main frozen artifact SHA changed")
    all_past[old_gen]=(prev_root/"actor",old_sha)

    old_actor_sha = _seal_bundle(
        result_root/f"generation{new_gen}-actor", out_root/"actor",
        expected_generation=new_gen, old_sha=old_sha, new_sha=head,
        engine=engine)
    magnet_sha = _seal_bundle(
        result_root/f"generation{new_gen}-magnet", out_root/"magnet",
        expected_generation=new_gen, old_sha=old_sha, new_sha=head,
        engine=engine)
    history=[]
    for gen,(src,source_sha) in sorted(all_past.items()):
        if gen < new_gen-HISTORY_DEPTH:
            continue
        ident=f"past-g{gen}"
        sha=_seal_bundle(src,out_root/"past"/ident,
                         expected_generation=gen,old_sha=source_sha,
                         new_sha=head,engine=engine)
        history.append({"id": ident,"sha256": sha})
    _assert(history and history[-1]["id"] == f"past-g{old_gen}",
            "Missing immediately previous immutable Main in opponent history")

    frozen=[]
    for entry in old["opponents"]["fixed"]:
        ident=entry["id"]
        _assert(ident.startswith("fixed-") and
                re.fullmatch(r"fixed-[a-z0-9]+",ident),
                "Unexpected frozen legacy identifier")
        name=ident.removeprefix("fixed-")
        source=prev_root/"fixed"/(name+"-v3.json")
        dest=out_root/"fixed"/(name+"-v3.json")
        _assert(digest(source)==entry["sha256"],
                "Frozen fixed benchmark bytes changed")
        _copy_immutable_file(source,dest)
        frozen.append(copy.deepcopy(entry))
    _assert(len(frozen) >= 1, "No fixed benchmark opponent")

    for name in ("original-draft.json","original-selection-js.json"):
        _copy_immutable_file(prev_root/name,out_root/name)

    # New generation seed range is disjoint from every earlier fixed window.
    next_manifest = {
        **old,
        "generation": new_gen,
        "seed_start": old["seed_start"]+SEED_STRIDE,
        "source_sha": head,
        "policy_manifest_sha256": old_actor_sha,
        "main_sha256": old_actor_sha,
        "magnet_sha256": magnet_sha,
        "opponents": {
            "historical": history,
            "fixed": frozen,
            "exploiter": [],
        },
    }
    manifest_sha=verify_manifest(next_manifest)
    write_json(out_root/"generation-manifest.json",next_manifest)

    ckpt["handoff_previous_checkpoint_sha256"]=index["full_checkpoint_sha256"]
    ckpt["handoff_previous_manifest_sha256"]=canonical_hash(old)
    ckpt["handoff_previous_source_sha"]=old_sha
    ckpt["source_sha"]=head
    ckpt["manifest_sha256"]=manifest_sha
    ckpt["main_sha256"]=old_actor_sha
    ckpt["magnet_sha256"]=magnet_sha
    ckpt["generation"]=new_gen
    ckpt["step"]=0
    ckpt["training_match_ids"]=[]
    ckpt.pop("order_sha256",None)
    ckpt["is_champion"]=False
    # No reset of optimizer moments, scheduler step, Target or Magnet.
    # Keep RNG state and completed generation journal intact.
    output_checkpoint=out_root/"checkpoint-g0.pt"
    torch.save(ckpt,output_checkpoint)
    write_json(out_root/"checkpoint-index.json",{
        "schema":"rnad-shared-learner-index-v1",
        "checkpoint_file":output_checkpoint.name,
        "sha256":digest(output_checkpoint),
        "generation":new_gen,
        "applied_matches":0,
        "completed_steps":0,
        "prior_generation_checkpoint_sha256":index["full_checkpoint_sha256"],
        "research_only":True,"is_champion":False,
    })
    write_json(out_root/"continuation-context.json", context)
    write_json(out_root/"handoff-report.json",{
        "schema":ARTIFACT_SCHEMA,
        "generation":new_gen,
        "previous_generation":old_gen,
        "source_sha":head,
        "previous_source_sha":old_sha,
        "engine_sha":engine,
        "main_sha256":old_actor_sha,
        "magnet_sha256":magnet_sha,
        "prior_checkpoint_sha256":index["full_checkpoint_sha256"],
        "new_full_checkpoint_sha256":digest(output_checkpoint),
        "new_generation_manifest_sha256":manifest_sha,
        "exact_previous_complete_matches":MATCHES_PER_GENERATION,
        "prior_optimizer_updates":OPTIMIZER_STEPS,
        "optimizer_full_state_preserved":True,
        "target_full_state_preserved":True,
        "magnet_full_state_preserved":True,
        "rng_full_state_preserved":True,
        "new_seed_start":next_manifest["seed_start"],
        "historical_pool":history,
        "next_wall_clock_deadline_epoch":context["deadline_epoch"],
        "research_only":True,"champion_promoted":False,
    })
    # Exact size/byte locks are transport safety, not substitute for Match receipts.
    files=[]
    for p in out_root.rglob("*"):
        if p.is_file() and p.name!="file-pins.sha256":
            files.append((str(p.relative_to(out_root)),digest(p)))
    (out_root/"file-pins.sha256").write_text(
        "".join(f"{sha}  {name}\n" for name,sha in sorted(files)))
    print(json.dumps({
        "generation":new_gen,
        "main_sha256":old_actor_sha,
        "magnet_sha256":magnet_sha,
        "prev_checkpoint_sha256":index["full_checkpoint_sha256"],
        "verified_prior_complete_matches":MATCHES_PER_GENERATION,
        "kept_optimizer_target_magnet_rng":True,
        "next_seed_start":next_manifest["seed_start"],
        "deadline_epoch":context["deadline_epoch"],
        "research_only":True,"champion_promoted":False,
    }),flush=True)
    return next_manifest


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for name in ("previous-bootstrap","previous-result","out"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--budget-seconds",type=int,
                   default=DEFAULT_BUDGET_SECONDS)
    a=p.parse_args()
    transition(previous_bootstrap=a.previous_bootstrap,
               previous_result=a.previous_result,out=a.out,
               budget_seconds=a.budget_seconds)
