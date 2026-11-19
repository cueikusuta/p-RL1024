"""Fail-closed, opt-in CPU staging of one authoritative 20-Actor generation.

Every immutable Actor is fully verified before ANY optimizer update. Distinct
processes can validate separate Actors, but the single parent assembles the
complete 20/20 barrier in fixed Actor-ID order. No model is updated here.
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.rnad.shared_league_contract import (
    ACTORS, MATCHES_PER_ACTOR, verify_actor, verify_manifest,
)


def _rss_kib():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        return 0


def _stage_actor(args):
    actor_id, source_path, manifest, destination, tensor_cache, fast_json = args
    source_path, destination = Path(source_path), Path(destination)
    timings = {"decompress": 0.0, "parse": 0.0, "verify": 0.0,
               "serialize": 0.0, "write": 0.0}
    start = time.perf_counter()
    if fast_json:
        with gzip.open(source_path, "rb") as source:
            raw = source.read()
        timings["decompress"] = time.perf_counter() - start
        parse_start = time.perf_counter()
        payload = json.loads(raw)
        del raw
        timings["parse"] = time.perf_counter() - parse_start
    else:
        # Keep the original streaming decoder as a measurable reference.
        with gzip.open(source_path, "rt") as source:
            payload = json.load(source)
        timings["decompress"] = time.perf_counter() - start
    if payload.get("actor_id") != actor_id:
        raise ValueError("Actor artifact ID mismatch")
    verify_start = time.perf_counter()
    seen = verify_actor(manifest, payload)
    timings["verify"] = time.perf_counter() - verify_start
    if len(seen) != MATCHES_PER_ACTOR:
        raise ValueError("Incomplete accepted Actor shard")
    role = payload["role"]
    meta = []
    for row in payload["matches"]:
        name = row["match_id"]
        if tensor_cache:
            import torch
            from scripts.rnad.shared_league_bootstrap import digest
            from scripts.rnad.shared_league_learner import _verified_phase_tensors
            target = destination / (name + ".pt")
            start = time.perf_counter()
            packed = _verified_phase_tensors(row)
            timings["serialize"] += time.perf_counter() - start
            start = time.perf_counter()
            torch.save(packed, target)
            checksum = digest(target)
            timings["write"] += time.perf_counter() - start
            meta.append({"match_id": name, "role": role,
                         "tensor_cache_sha256": checksum})
        elif fast_json:
            start = time.perf_counter()
            encoded = json.dumps(
                row, separators=(",", ":"), allow_nan=False).encode("utf-8")
            timings["serialize"] += time.perf_counter() - start
            start = time.perf_counter()
            (destination / (name + ".json")).write_bytes(encoded)
            timings["write"] += time.perf_counter() - start
            meta.append({"match_id": name, "role": role})
        else:
            # Byte-identical reference to the pre-optimization implementation.
            start = time.perf_counter()
            with (destination / (name + ".json")).open(
                    "w", encoding="utf-8") as sink:
                json.dump(row, sink, separators=(",", ":"), allow_nan=False)
            timings["write"] += time.perf_counter() - start
            meta.append({"match_id": name, "role": role})
    del payload
    return {"actor_id": actor_id, "ids": seen, "role": role,
            "meta": meta, "timings": timings, "rss_kib": _rss_kib()}


def stage_verified_actors(manifest, paths, verified, *,
                          workers=1, fast_json=False, profile=False,
                          return_metrics=False):
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("RNAD_PREP_WORKERS must be 1, 2 or 4")
    if type(fast_json) is not bool or type(profile) is not bool:
        raise ValueError("Invalid staging flags")
    verify_manifest(manifest)
    if len(paths) != ACTORS or any(not Path(p).is_file() for p in paths):
        raise ValueError("Cannot stage incomplete 20/20 Actor set")
    verified = Path(verified)
    verified.mkdir(parents=True, exist_ok=True)
    args = [(i, str(path), manifest, str(verified), False, fast_json)
            for i, path in enumerate(paths)]
    start = time.perf_counter()
    if workers == 1:
        results = [_stage_actor(arg) for arg in args]
    else:
        # Spawn avoids inheriting any PyTorch autograd or filesystem state.
        with ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(_stage_actor, args))
    match_ids, meta, roles = set(), [], Counter()
    for i, item in enumerate(results):
        if item["actor_id"] != i or item["ids"] & match_ids:
            raise ValueError("Missing/duplicated Actor identity at strict barrier")
        match_ids.update(item["ids"])
        meta.extend(item["meta"])
        roles[item["role"]] += 1
    if len(match_ids) != ACTORS * MATCHES_PER_ACTOR:
        raise ValueError("Missing accepted Match at 20/20 staging barrier")
    wall = time.perf_counter() - start
    metrics = {
        "schema": "rnad-learner-staging-profile-v1",
        "actors": len(results), "accepted_matches": len(match_ids),
        "workers": workers, "fast_json": fast_json,
        "wall_seconds": round(wall, 4),
        "max_worker_rss_kib": max(item["rss_kib"] for item in results),
        "sum_actor_phase_seconds": {
            k: round(sum(item["timings"][k] for item in results), 4)
            for k in ("decompress", "parse", "verify", "serialize", "write")
        },
        "research_only": True, "strength_claim": False,
    }
    if profile:
        print(json.dumps(metrics, sort_keys=True), flush=True)
    if return_metrics:
        return match_ids, meta, roles, metrics
    return match_ids, meta, roles


def staging_config():
    value = os.environ.get("RNAD_PREP_WORKERS", "1")
    fast = os.environ.get("RNAD_PREP_FAST_JSON", "0")
    prof = os.environ.get("RNAD_PROFILE_STAGING", "0")
    if value not in ("1", "2", "4") or fast not in ("0", "1") or prof not in ("0", "1"):
        raise ValueError("Invalid bounded R-NaD Learner staging configuration")
    return int(value), fast == "1", prof == "1"
