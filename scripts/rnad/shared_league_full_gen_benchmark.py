"""Strict historical G0 full-generation CPU throughput comparison (research only).

The accepted 1,280 Matches and immutable initial checkpoint are pinned to the
original 2026-11-18 source SHA. Different code is evaluated as a BENCHMARK,
not authority replay or a new legal training generation. Checkpoints and
optimizer updates remain in TemporaryDirectory and are never published.
"""
from __future__ import annotations

import argparse
import os
import json
import math
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.rnad.shared_league_bootstrap import digest
from scripts.rnad.shared_league_contract import (
    ACTORS, MATCHES_PER_ACTOR, MATCHES_PER_GENERATION,
    verify_actor, verify_manifest,
)
from scripts.rnad.shared_league_learner import (
    _load, _models, _match_loss, _order, _atomic_checkpoint,
)
from scripts.rnad.shared_league_parallel import ParallelGradientPool

HISTORIC_SOURCE_SHA = "0f348495dd381dbfd299e00c0346f61351212075"
HISTORIC_GENERATION = 0


def _collect_original(root, actor_dir, verified_dir):
    start = time.perf_counter()
    manifest = json.loads((root/"generation-manifest.json").read_text())
    verify_manifest(manifest)
    if (manifest["source_sha"] != HISTORIC_SOURCE_SHA
            or manifest["generation"] != HISTORIC_GENERATION):
        raise ValueError("This benchmark only accepts immutable original G0 provenance")
    ids, meta, roles = set(), [], Counter()
    for actor_id in range(ACTORS):
        source = actor_dir/f"actor-{actor_id}"/f"actor-{actor_id}.json.gz"
        with __import__("gzip").open(source, "rt") as f:
            payload = json.load(f)
        if payload.get("actor_id") != actor_id:
            raise ValueError("Wrong authoritative Actor ID")
        seen = verify_actor(manifest, payload)
        if len(seen) != MATCHES_PER_ACTOR or seen & ids:
            raise ValueError("Missing or duplicate authoritative Matches")
        ids.update(seen)
        roles[payload["role"]] += 1
        for record in payload["matches"]:
            mid = record["match_id"]
            (verified_dir/(mid+".json")).write_text(
                json.dumps(record, separators=(",", ":"), allow_nan=False),
                encoding="utf-8")
            meta.append({"match_id":mid,"role":payload["role"]})
        del payload
    expected={"self":8,"historical":8,"fixed":4}
    if manifest["opponents"]["exploiter"]:
        expected={"self":8,"historical":6,"fixed":4,"exploiter":2}
    if len(ids)!=MATCHES_PER_GENERATION or dict(roles)!=expected:
        raise ValueError("Historical full-generation actor quota mismatch")
    return meta, roles, time.perf_counter()-start


def _run(bootstrap, ordering, verified_dir, backend, tempdir):
    tempdir.mkdir(parents=True, exist_ok=False)
    ckpt_path=bootstrap/"checkpoint-g0.pt"
    ckpt_index=json.loads((bootstrap/"checkpoint-index.json").read_text())
    if digest(ckpt_path)!=ckpt_index["sha256"]:
        raise ValueError("Corrupt pinned G0 full-state checkpoint")
    checkpoint=torch.load(ckpt_path,map_location="cpu",weights_only=True)
    if (checkpoint.get("source_sha")!=HISTORIC_SOURCE_SHA or
            checkpoint.get("generation")!=0):
        raise ValueError("Wrong historical full-state checkpoint")
    model,target,magnet,opt,scheduler=_models(bootstrap,checkpoint)
    torch.set_num_threads(1)
    if backend in ("cpu-parallel-vector", "cpu-parallel-vector-reused-tensors"):
        pool=ParallelGradientPool(model,target,magnet,workers=4,vector=True)
    else:
        pool=None
    step_total,grad_total,save_total=[],[],[]
    mean_losses,max_ratios=[],[]
    try:
        for step in range(40):
            batch=ordering[step*32:(step+1)*32]
            if len(batch)!=32 or len(set(row["match_id"] for row in batch))!=32:
                raise ValueError("Benchmark must use exactly 32 new Matches per update")
            opt.zero_grad(set_to_none=True)
            start=time.perf_counter()
            if pool is None:
                losses,ratios=[],[]
                for choice in batch:
                    record=_load(verified_dir/(choice["match_id"]+".json"))
                    loss,ratio=_match_loss(model,target,magnet,record,
                                            role=choice["role"])
                    (loss/32).backward()
                    losses.append(float(loss.detach()))
                    ratios.append(ratio)
            else:
                task=[(str(verified_dir/(row["match_id"]+".json")),
                       row["role"],None) for row in batch]
                losses,ratios=pool.step_gradients(task)
            grad_seconds=time.perf_counter()-start
            if len(losses)!=32 or len(ratios)!=32 or (
                    any(not math.isfinite(v) for v in losses+ratios)):
                raise ValueError("Missing or nonfinite full Match losses")
            mean_losses.append(sum(losses)/len(losses))
            max_ratios.append(max(ratios))
            norm=torch.nn.utils.clip_grad_norm_(
                model.parameters(),10.0,error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite full-generation shared gradient")
            opt.step()
            scheduler.step()
            with torch.no_grad():
                for dest,source in zip(target.parameters(),model.parameters()):
                    dest.mul_(.99).add_(source,alpha=.01)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite research checkpoint parameters")
            checkpoint.update({
                "step":step+1,
                "model":{k:v.clone() for k,v in model.state_dict().items()},
                "target":{k:v.clone() for k,v in target.state_dict().items()},
                "magnet":magnet.state_dict(),
                "optimizer":opt.state_dict(),
                "scheduler":scheduler.state_dict(),
                "torch_rng":torch.get_rng_state(),
                "python_rng":__import__("random").getstate(),
                "training_match_ids":[x["match_id"] for x in ordering[:(step+1)*32]],
            })
            save_start=time.perf_counter()
            _atomic_checkpoint(tempdir/"disposable-replay.pt",checkpoint)
            save_seconds=time.perf_counter()-save_start
            step_total.append(time.perf_counter()-start)
            grad_total.append(grad_seconds)
            save_total.append(save_seconds)
            print(json.dumps({
                "experimental_backend":backend, "optimizer_step":step+1,
                "accepted_complete_matches_examined":(step+1)*32,
                "wall_seconds":round(step_total[-1],4),
                "research_only":True}),flush=True)
    finally:
        if pool is not None:
            pool.close()
    return {
        "backend":backend,
        "total_40_update_seconds":round(sum(step_total),3),
        "total_grad_seconds":round(sum(grad_total),3),
        "total_checkpoint_save_seconds":round(sum(save_total),3),
        "median_step_seconds":round(statistics.median(step_total),4),
        "accepted_complete_matches":len(ordering),
        "optimizer_updates":len(step_total),
        "mean_loss_per_step":mean_losses,
        "maximum_ratio_per_step":max_ratios,
        "model":{k:v.detach().clone() for k,v in model.state_dict().items()},
        "full_state":{
            "step":checkpoint["step"],
            "training_match_ids":checkpoint["training_match_ids"],
            "model":checkpoint["model"],
            "target":checkpoint["target"],
            "magnet":checkpoint["magnet"],
            "optimizer":checkpoint["optimizer"],
            "scheduler":checkpoint["scheduler"],
            "torch_rng":checkpoint["torch_rng"],
            "python_rng":checkpoint["python_rng"],
        },
    }



def _audit_full_state(reference, candidate):
    """Fail closed on checkpoint/optimizer/scheduler/RNG and Match ordering drift.

    Unlike comparing only the final Main, this detects a broken Target EMA,
    frozen Magnet, Adam moments, scheduler, replay order or RNG continuation.
    Tolerance reflects the existing parallel CPU floating-point parity gate.
    """
    a, b = reference["full_state"], candidate["full_state"]
    if a["step"] != 40 or b["step"] != 40:
        raise AssertionError("Incomplete 40-update checkpoint")
    if a["training_match_ids"] != b["training_match_ids"] or (
            len(a["training_match_ids"]) != 1280 or
            len(set(a["training_match_ids"])) != 1280):
        raise AssertionError("Mismatch in complete accepted Match ordering")
    if a["scheduler"] != b["scheduler"]:
        raise AssertionError("Scheduler state changed")
    if a["python_rng"] != b["python_rng"] or not torch.equal(
            a["torch_rng"], b["torch_rng"]):
        raise AssertionError("Full-state RNG continuation changed")
    maxima = {}
    for component in ("model", "target", "magnet"):
        old, new = a[component], b[component]
        if set(old) != set(new):
            raise AssertionError(component + " keys mismatch")
        widest = 0.0
        for key in old:
            torch.testing.assert_close(
                old[key], new[key], rtol=4e-3, atol=4e-5)
            widest = max(widest, float((old[key]-new[key]).abs().max()))
        maxima[component] = widest
    old_opt, new_opt = a["optimizer"], b["optimizer"]
    if old_opt["param_groups"] != new_opt["param_groups"] or (
            set(old_opt["state"]) != set(new_opt["state"])):
        raise AssertionError("Adam optimizer groups or slots changed")
    moments = 0
    moment_maxima = {"exp_avg":0.0, "exp_avg_sq":0.0}
    for key, old_slot in old_opt["state"].items():
        new_slot = new_opt["state"][key]
        if set(old_slot) != set(new_slot):
            raise AssertionError("Adam moment fields changed")
        for field, first in old_slot.items():
            second = new_slot[field]
            if field == "step":
                if (not torch.equal(first, second) or first.numel() != 1
                        or int(first.item()) != 40):
                    raise AssertionError("Adam step counter changed or incomplete")
            elif field in moment_maxima:
                if first.shape != second.shape or first.dtype != second.dtype:
                    raise AssertionError("Adam moment shape/dtype changed")
                if not torch.isfinite(first).all() or not torch.isfinite(second).all():
                    raise AssertionError("Nonfinite Adam moment")
                atol = 4e-6 if field == "exp_avg" else 4e-10
                torch.testing.assert_close(first, second, rtol=4e-3, atol=atol)
                moment_maxima[field] = max(
                    moment_maxima[field], float((first-second).abs().max()))
                moments += 1
            elif isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
                if not torch.equal(first, second):
                    raise AssertionError("Unexpected Adam tensor drift")
            elif first != second:
                raise AssertionError("Unexpected Adam metadata drift")
    if not moments:
        raise AssertionError("No Adam state moments audited")
    if len(reference["mean_loss_per_step"]) != 40 or (
            len(candidate["mean_loss_per_step"]) != 40):
        raise AssertionError("Incomplete loss history")
    loss_differences = [
        abs(x-y) for x,y in zip(
            reference["mean_loss_per_step"],candidate["mean_loss_per_step"])]
    ratio_differences = [
        abs(x-y) for x,y in zip(
            reference["maximum_ratio_per_step"],candidate["maximum_ratio_per_step"])]
    if max(loss_differences) > 1e-3 or max(ratio_differences) > 1e-5:
        raise AssertionError("Loss or V-trace ratio drift over 40 updates")
    return {"full_state_parity_passed":True,
            "all_1280_accepted_match_ids_in_identical_order":True,
            "model_target_magnet_max_abs_diff":maxima,
            "adam_moment_tensors_compared":moments,
            "adam_moment_max_abs_diff":moment_maxima,
            "scheduler_equal":True, "torch_rng_equal":True,
            "python_rng_equal":True,
            "max_40_step_mean_loss_abs_diff":max(loss_differences),
            "max_40_step_ratio_abs_diff":max(ratio_differences)}


def compare(bootstrap, actors, *, strict=False, compare_packing=False):
    root=Path(bootstrap)
    with tempfile.TemporaryDirectory(prefix="rnad-g0-observational-") as tmp:
        scratch=Path(tmp)
        verified_dir=scratch/"accepted"
        verified_dir.mkdir()
        meta,roles,preparation_seconds=_collect_original(
            root,Path(actors),verified_dir)
        ordered=_order(meta)
        if len(ordered)!=1280:
            raise ValueError("Wrong exact original one-generation ordering")
        if compare_packing:
            # Same verified 1280/40-update inputs, frozen original G0 and
            # exactly one 4-worker optimizer stream in each independent run.
            with patch.dict(os.environ, {"RNAD_REUSE_FROZEN_PHASE_TENSORS":"0"}):
                scalar=_run(root,ordered,verified_dir,
                            "cpu-parallel-vector",scratch/"vector-reference")
            with patch.dict(os.environ, {"RNAD_REUSE_FROZEN_PHASE_TENSORS":"1"}):
                optimized=_run(root,ordered,verified_dir,
                               "cpu-parallel-vector-reused-tensors",
                               scratch/"vector-reused")
        else:
            scalar=_run(root,ordered,verified_dir,
                        "original-scalar",scratch/"scalar")
            optimized=_run(root,ordered,verified_dir,
                           "cpu-parallel-vector",scratch/"parallel")
        largest=max(float((v-optimized["model"][k]).abs().max())
                    for k,v in scalar["model"].items())
        checksum_equal=all(torch.equal(v,optimized["model"][k])
                           for k,v in scalar["model"].items())
        speed=scalar["total_40_update_seconds"]/optimized[
            "total_40_update_seconds"]
        def simple(x):
            return {k:v for k,v in x.items()
                    if k not in ("model","full_state")}
        result={
            "schema":("rnad-i5-g0-1280-match-40-update-packing-parity-v1"
                       if compare_packing else
                       "rnad-i5-original-g0-1280-match-40-step-isolated-timing-v1"),
            "compare_reused_phase_tensors":compare_packing,
            "original_source_sha":HISTORIC_SOURCE_SHA,
            "observational_new_code_not_valid_checkpoint_continuation":True,
            "one_shared_optimizer_in_each_trial":True,
            "actor_policy_frozen":True,
            "strict_verification_of_original_match_artifacts":True,
            "actor_role_quota":dict(roles),
            "accepted_complete_matches":1280,
            "preparation_seconds":round(preparation_seconds,3),
            "reference":simple(scalar),
            "experimental":simple(optimized),
            "max_final_model_parameter_abs_difference":largest,
            "final_models_bitwise_equal":checksum_equal,
            "one_trial_40_step_speed_ratio":round(speed,4),
            "full_state_audit":(_audit_full_state(scalar,optimized)
                                if compare_packing else None),
            "full_generation_actor_collection_timed":False,
            "same_32_match_group_40_updates_tested":True,
            "twofold_wall_speed_proven":False,
            "strength_comparison_performed":False,
            "champion_promoted":False,
            "main_or_workflow_changed":False,
        }
        if strict and (largest>0.005 or not math.isfinite(largest)):
            raise ValueError("Excess final full-generation model divergence")
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bootstrap",required=True)
    p.add_argument("--actors",required=True)
    p.add_argument("--strict",action="store_true")
    p.add_argument("--compare-packing",action="store_true")
    a=p.parse_args()
    print(json.dumps(compare(a.bootstrap,a.actors,strict=a.strict,
                             compare_packing=a.compare_packing),
                     indent=2),flush=True)


if __name__=="__main__":
    main()
