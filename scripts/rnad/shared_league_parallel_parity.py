"""Independent real-authority 32-Match CPU gradient-parallel research gate.

No checkpoint or opponent is promoted; compare an unchanged 32-Match optimizer
step using pinned G0 Match receipts. Each worker reads only validated Matches.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
import time
from unittest.mock import patch
from pathlib import Path

import torch

from scripts.rnad.shared_league_bootstrap import digest
from scripts.rnad.shared_league_contract import verify_actor, verify_manifest
from scripts.rnad.shared_league_learner import _match_loss, _models
from scripts.rnad.shared_league_parallel import ParallelGradientPool
from scripts.rnad.shared_league_batched import batched_match_losses


def benchmark(bootstrap, actor_bundle, *, processes=4):
    torch.set_num_threads(1)
    root = Path(bootstrap)
    manifest = json.loads((root/"generation-manifest.json").read_text())
    verify_manifest(manifest)
    with gzip.open(actor_bundle, "rt") as f:
        payload = json.load(f)
    verified = verify_actor(manifest, payload)
    if len(verified) != 64 or payload["actor_id"] != 0:
        raise ValueError("Need authoritative 64-Match actor-0 bundle")
    if digest(root/"checkpoint-g0.pt") != json.loads(
            (root/"checkpoint-index.json").read_text())["sha256"]:
        raise ValueError("Pinned bootstrap checkpoint digest mismatch")
    checkpoint = torch.load(root/"checkpoint-g0.pt", map_location="cpu",
                            weights_only=True)
    rows = sorted(payload["matches"], key=lambda r: r["match_id"])[:32]
    if len({r["match_id"] for r in rows}) != 32:
        raise ValueError("Not 32 unique accepted completed Matches")

    serial, target, magnet, opt, scheduler = _models(root, checkpoint)
    opt.zero_grad(set_to_none=True)
    started = time.perf_counter()
    losses, ratios = [], []
    for row in rows:
        loss, ratio = _match_loss(serial, target, magnet, row,
                                  role=payload["role"])
        (loss/32).backward()
        losses.append(float(loss.detach()))
        ratios.append(ratio)
    scalar_seconds = time.perf_counter() - started
    serial_grads = [p.grad.detach().clone() for p in serial.parameters()]
    torch.nn.utils.clip_grad_norm_(serial.parameters(), 10.,
                                   error_if_nonfinite=True)
    opt.step()
    scheduler.step()

    parallel, p_target, p_magnet, p_opt, p_schedule = _models(root, checkpoint)
    with tempfile.TemporaryDirectory() as tmp:
        tasks = []
        for row in rows:
            path = Path(tmp)/(row["match_id"]+".json")
            path.write_text(json.dumps(row, separators=(",",":"),
                                       allow_nan=False))
            tasks.append((str(path), payload["role"], None))
        start = time.perf_counter()
        with ParallelGradientPool(
                parallel, p_target, p_magnet, workers=processes) as workers:
            init_seconds = time.perf_counter()-start
            start_work = time.perf_counter()
            other_losses, other_ratios = workers.step_gradients(tasks)
            parallel_seconds = time.perf_counter()-start_work
            parallel_grads = [p.grad.detach().clone()
                              for p in parallel.parameters()]
        for left, right in zip(serial_grads, parallel_grads):
            torch.testing.assert_close(
                left, right, rtol=4e-3, atol=4e-5)
        torch.nn.utils.clip_grad_norm_(parallel.parameters(), 10.,
                                       error_if_nonfinite=True)
        p_opt.step()
        p_schedule.step()
        for p, q in zip(serial.parameters(), parallel.parameters()):
            torch.testing.assert_close(p, q, rtol=4e-3, atol=4e-5)
    if abs(sum(losses)-sum(other_losses)) > 1e-3 or (
            abs(max(ratios)-max(other_ratios)) > 1e-5):
        raise AssertionError("Loss or V-trace ratios drifted")
    # Additional full optimizer step on the SAME 32 real accepted Matches.
    # Compare all parameter gradients, not only batch loss magnitudes.
    vector, vtarget, vmagnet, vopt, vschedule = _models(root, checkpoint)
    vopt.zero_grad(set_to_none=True)
    vector_start = time.perf_counter()
    vector_losses, vector_ratios = [], []
    for start in range(0, 32, 4):
        block = [(row, payload["role"]) for row in rows[start:start+4]]
        results = batched_match_losses(vector, vtarget, vmagnet, block)
        (sum(loss for loss,_ in results)/32).backward()
        vector_losses.extend(float(loss.detach()) for loss,_ in results)
        vector_ratios.extend(ratio for _,ratio in results)
    vector_seconds = time.perf_counter() - vector_start
    for reference_grad, vec_param in zip(serial_grads, vector.parameters()):
        torch.testing.assert_close(
            reference_grad, vec_param.grad, rtol=4e-3, atol=4e-5)
    torch.nn.utils.clip_grad_norm_(vector.parameters(), 10.,
                                   error_if_nonfinite=True)
    vopt.step()
    vschedule.step()
    for p, q in zip(serial.parameters(), vector.parameters()):
        torch.testing.assert_close(p, q, rtol=4e-3, atol=4e-5)
    if abs(sum(losses)-sum(vector_losses)) > 1e-3 or (
            abs(max(ratios)-max(vector_ratios)) > 1e-5):
        raise AssertionError("Vectorized full 32 Match objective drifted")
    vector_parallel, vt, vm, vo, vs = _models(root, checkpoint)
    with tempfile.TemporaryDirectory() as tmp:
        tasks=[]
        for row in rows:
            path=Path(tmp)/(row["match_id"]+".json")
            path.write_text(json.dumps(row, separators=(",",":"),allow_nan=False))
            tasks.append((str(path),payload["role"],None))
        with ParallelGradientPool(vector_parallel, vt, vm,
                                  workers=processes, vector=True) as workers:
            begun=time.perf_counter()
            vp_losses,vp_ratios=workers.step_gradients(tasks)
            vector_parallel_seconds=time.perf_counter()-begun
        for left,right in zip(serial_grads, vector_parallel.parameters()):
            torch.testing.assert_close(left,right.grad,rtol=4e-3,atol=4e-5)
        # Optional fine-grained task scheduler: same 32 immutable accepted
        # Matches, identical single optimizer, checked against scalar reference.
        dynamic, dt, dm, do, ds = _models(root, checkpoint)
        with ParallelGradientPool(dynamic, dt, dm, workers=processes,
                                  vector=True, task_matches=4) as workers:
            begun=time.perf_counter()
            dynamic_losses,dynamic_ratios=workers.step_gradients(tasks)
            dynamic_seconds=time.perf_counter()-begun
        for left,right in zip(serial_grads, dynamic.parameters()):
            torch.testing.assert_close(left,right.grad,rtol=4e-3,atol=4e-5)
        # Candidate: one 8-Match vector microbatch per worker, on the
        # SAME authoritative 32 rows, with the same frozen Main/Target/Magnet.
        vector8, t8, m8, o8, s8 = _models(root, checkpoint)
        with ParallelGradientPool(vector8, t8, m8, workers=processes,
                                  vector=True, vector_batch=8) as workers:
            begun=time.perf_counter()
            eight_losses,eight_ratios=workers.step_gradients(tasks)
            eight_seconds=time.perf_counter()-begun
        for left,right in zip(serial_grads, vector8.parameters()):
            torch.testing.assert_close(left,right.grad,rtol=4e-3,atol=4e-5)
        # Separate NumPy CPU target hypothesis: all-action Q is detached,
        # but all 32 full-Match gradients and the single optimizer step
        # must still match the unchanged scalar Torch V-trace reference.
        fast, ft, fm, fo, fs = _models(root, checkpoint)
        with patch.dict(os.environ, {"RNAD_FAST_RNAD_TARGETS": "1"}):
            with ParallelGradientPool(fast, ft, fm, workers=processes,
                                      vector=True) as workers:
                begun=time.perf_counter()
                fast_losses,fast_ratios=workers.step_gradients(tasks)
                fast_seconds=time.perf_counter()-begun
        for left,right in zip(serial_grads, fast.parameters()):
            torch.testing.assert_close(left,right.grad,rtol=4e-3,atol=4e-5)
    torch.nn.utils.clip_grad_norm_(fast.parameters(),10.,
                                   error_if_nonfinite=True)
    fo.step()
    fs.step()
    for p,q in zip(serial.parameters(),fast.parameters()):
        torch.testing.assert_close(p,q,rtol=4e-3,atol=4e-5)
    if abs(sum(losses)-sum(fast_losses)) > 1e-3 or (
            abs(max(ratios)-max(fast_ratios)) > 1e-5):
        raise AssertionError("NumPy CPU all-action R-NaD target objective changed")
    torch.nn.utils.clip_grad_norm_(vector8.parameters(),10.,
                                   error_if_nonfinite=True)
    o8.step()
    s8.step()
    for p,q in zip(serial.parameters(),vector8.parameters()):
        torch.testing.assert_close(p,q,rtol=4e-3,atol=4e-5)
    if abs(sum(losses)-sum(eight_losses)) > 1e-3 or (
            abs(max(ratios)-max(eight_ratios)) > 1e-5):
        raise AssertionError("8-Match vector gradient objective changed")
    torch.nn.utils.clip_grad_norm_(dynamic.parameters(),10.,
                                   error_if_nonfinite=True)
    do.step()
    ds.step()
    for p,q in zip(serial.parameters(),dynamic.parameters()):
        torch.testing.assert_close(p,q,rtol=4e-3,atol=4e-5)
    if abs(sum(losses)-sum(dynamic_losses)) > 1e-3 or (
            abs(max(ratios)-max(dynamic_ratios)) > 1e-5):
        raise AssertionError("Dynamic parallel gradient objective changed")
    torch.nn.utils.clip_grad_norm_(vector_parallel.parameters(),10.,
                                   error_if_nonfinite=True)
    vo.step()
    vs.step()
    for p,q in zip(serial.parameters(),vector_parallel.parameters()):
        torch.testing.assert_close(p,q,rtol=4e-3,atol=4e-5)
    if abs(sum(losses)-sum(vp_losses)) > 1e-3 or (
            abs(max(ratios)-max(vp_ratios)) > 1e-5):
        raise AssertionError("Parallel vector objective changed")
    return {
        "schema":"rnad-i5-authoritative-parallel-gradient-research-v1",
        "source_sha": manifest["source_sha"],
        "accepted_matches":32,
        "accepted_decisions":sum(len(r["events"]) for r in rows),
        "worker_count":processes,
        "serial_32_match_gradient_seconds":round(scalar_seconds, 4),
        "worker_init_seconds":round(init_seconds, 4),
        "parallel_32_match_gradient_seconds":round(parallel_seconds, 4),
        "vector_microbatch4_gradient_seconds":round(vector_seconds, 4),
        "parallel_vector_microbatch4_gradient_seconds":
            round(vector_parallel_seconds, 4),
        "parallel_vector_microbatch8_gradient_seconds":
            round(eight_seconds, 4),
        "single_trial_ratio_microbatch4_over_8":
            round(vector_parallel_seconds / eight_seconds, 4),
        "parallel_vector_numpy_targets_seconds": round(fast_seconds, 4),
        "single_trial_ratio_torch_over_numpy_targets":
            round(vector_parallel_seconds / fast_seconds, 4),
        "dynamic_task_matches":4,
        "parallel_dynamic_microbatch4_gradient_seconds":
            round(dynamic_seconds, 4),
        "single_trial_speed_ratio_fixed_over_dynamic":
            round(vector_parallel_seconds/dynamic_seconds, 4),
        "single_trial_speed_ratio_scalar_over_parallel_vector":
            round(scalar_seconds/vector_parallel_seconds, 4),
        "single_trial_speed_ratio_scalar_over_vector":
            round(scalar_seconds/vector_seconds, 4),
        "single_trial_speed_ratio_not_established":
            round(scalar_seconds / parallel_seconds, 4),
        "gradients_and_optimizer_step_numerically_equivalent":True,
        "strength_evaluated":False,"generation_completed":False,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bootstrap",required=True)
    parser.add_argument("--actor",required=True)
    parser.add_argument("--workers",type=int,choices=(2,4),default=4)
    a=parser.parse_args()
    print(json.dumps(benchmark(a.bootstrap,a.actor,processes=a.workers),
                     indent=2),flush=True)


if __name__=="__main__":
    main()
