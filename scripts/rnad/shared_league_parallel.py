"""Opt-in CPU-parallel R-NaD gradient calculation with ONE authoritative writer.

Workers only read pinned full Matches and compute gradients. Shared Main
parameters are frozen during Pool.map; the parent alone clips, runs Adam,
updates target/magnet and writes the full-state checkpoint. No asynchronous
policy update or off-policy/stale-parameter training is allowed.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.multiprocessing as mp

from scripts.rnad.shared_league_learner import _load, _match_loss

_WORKER_NETS = None


def _worker_init(model, target, magnet):
    global _WORKER_NETS
    torch.set_num_threads(1)
    _WORKER_NETS = (model, target, magnet)


def _grad_chunk(rows):
    if _WORKER_NETS is None:
        raise RuntimeError("Shared CPU worker not initialized")
    model, target, magnet = _WORKER_NETS
    model.zero_grad(set_to_none=True)
    losses, ratios = [], []
    for path, role, cache_sha256 in rows:
        if cache_sha256 is None:
            match = _load(path)
        else:
            from scripts.rnad.shared_league_bootstrap import digest
            if digest(path) != cache_sha256:
                raise ValueError("Worker encountered mismatched strict tensor cache")
            match = torch.load(path, map_location="cpu", weights_only=True)
        loss, ratio = _match_loss(model, target, magnet, match, role=role)
        (loss / 32).backward()
        losses.append(float(loss.detach()))
        ratios.append(ratio)
        del match
    tensors = []
    for param in model.parameters():
        tensors.append(
            param.grad.detach().reshape(-1).numpy()
            if param.grad is not None else
            np.zeros(param.numel(), dtype=np.float32))
    flat = np.concatenate(tensors).astype(np.float32, copy=False)
    if not np.isfinite(flat).all():
        raise ValueError("Worker produced nonfinite gradient")
    return flat, losses, ratios


class ParallelGradientPool:
    def __init__(self, model, target, magnet, *, workers=4):
        if type(workers) is not int or workers not in (2, 4):
            raise ValueError("Use 2 or 4 bounded hosted CPU workers")
        torch.set_num_threads(1)
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Start worker pool before accumulating gradients")
        # PyTorch CPU tensor storage shared with spawned workers. The worker
        # autograd .grad buffers are *not* shared: reduction is explicit.
        for network in (model, target, magnet):
            network.share_memory()
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            workers, initializer=_worker_init,
            initargs=(model, target, magnet))
        self.workers = workers
        self.model = model

    def step_gradients(self, rows):
        if len(rows) != 32:
            raise ValueError("Exactly 32 complete Match rows per optimizer step")
        chunks = [rows[i::self.workers] for i in range(self.workers)]
        result = self.pool.map(_grad_chunk, chunks)
        # Sum in a bounded deterministic order; independent task completion
        # order has no effect on gradient reduction order.
        flat = np.zeros_like(result[0][0])
        for grad, _, _ in result:
            flat += grad
        offset = 0
        for param in self.model.parameters():
            n = param.numel()
            param.grad = torch.from_numpy(flat[offset:offset+n].copy()).view_as(param)
            offset += n
        if offset != flat.size or not np.isfinite(flat).all():
            raise ValueError("Wrong worker gradient dimensions")
        losses = [v for _, part, _ in result for v in part]
        ratios = [v for _, _, part in result for v in part]
        return losses, ratios

    def close(self):
        self.pool.close()
        self.pool.join()

    def terminate(self):
        self.pool.terminate()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, ty, exc, tb):
        if ty is None:
            self.close()
        else:
            self.terminate()


def configured_workers():
    try:
        count = int(os.environ.get("RNAD_LEARNER_CPU_WORKERS", "4"))
    except ValueError as exc:
        raise ValueError("Invalid hosted CPU worker count") from exc
    if count not in (2, 4):
        raise ValueError("RNAD_LEARNER_CPU_WORKERS must be 2 or 4")
    return count
