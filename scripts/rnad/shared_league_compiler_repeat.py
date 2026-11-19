"""Compare cold TorchInductor compilation against steady reused head execution.

Repeated forward/backward uses the SAME accepted observations, frozen model
versions and optimizer-free snapshots. Never count repeats as independent
Matches or evidence of stronger training policies.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from scripts.rnad.shared_league_batched import batched_match_losses
from scripts.rnad.shared_league_learner import _models
from scripts.rnad.shared_league_vector_parity import _data


def bench(bootstrap, smoke, *, repetitions=5):
    if repetitions not in (5, 8, 10):
        raise ValueError("Bounded compilation repetitions only")
    torch.set_num_threads(1)
    root=Path(bootstrap)
    rows=_data(root,Path(smoke))
    checkpoint=torch.load(root/"checkpoint-g0.pt",
                          map_location="cpu",weights_only=True)
    eager,et,em,eo,_=_models(root,checkpoint)
    compiled,ct,cm,co,_=_models(root,checkpoint)

    def trial(model,target,magnet,optimizer,flag):
        os.environ["RNAD_TORCH_COMPILE"]=flag
        optimizer.zero_grad(set_to_none=True)
        start=time.perf_counter()
        vals=batched_match_losses(model,target,magnet,rows)
        (sum(loss for loss,_ in vals)/len(rows)).backward()
        return time.perf_counter()-start,(
            [float(loss.detach()) for loss,_ in vals],
            [p.grad.detach().clone() for p in model.parameters()])

    eager_trials=[]
    eager_first=None
    for i in range(repetitions):
        sec,values=trial(eager,et,em,eo,"0")
        eager_trials.append(sec)
        if i==0:
            eager_first=values
    compiled_trials=[]
    compiled_first=None
    for i in range(repetitions):
        sec,values=trial(compiled,ct,cm,co,"1")
        compiled_trials.append(sec)
        if i==0:
            compiled_first=values
    for a,b in zip(eager_first[0],compiled_first[0]):
        if abs(a-b)>2e-4:
            raise AssertionError("Warm TorchInductor value parity")
    for a,b in zip(eager_first[1],compiled_first[1]):
        torch.testing.assert_close(a,b,rtol=3e-3,atol=3e-5)
    baseline=statistics.median(eager_trials[1:])
    hot=statistics.median(compiled_trials[1:])
    cold=compiled_trials[0]
    return {
        "schema":"rnad-i5-pinned-match-cpu-torchinductor-hot-v1",
        "accepted_real_matches_per_repeat":len(rows),
        "same_matches_reused_for_microbenchmark_only":True,
        "trials":repetitions,
        "eager_seconds":eager_trials,
        "compile_enabled_seconds_including_cold":compiled_trials,
        "eager_hot_median_seconds":baseline,
        "compiled_hot_median_seconds":hot,
        "first_compiled_call_seconds":cold,
        "hot_ratio_eager_over_compiled":baseline/hot,
        "estimated_cache_calls_to_recover_cold_overhead": (
            (cold-hot)/(baseline-hot) if baseline>hot else None),
        "full_generation_speed_claim":False,
        "strength_evaluation":False,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bootstrap",required=True)
    p.add_argument("--smoke",required=True)
    a=p.parse_args()
    print(json.dumps(bench(a.bootstrap,a.smoke),indent=2),flush=True)
