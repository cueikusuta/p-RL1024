"""Bounded concurrent retrieval of the 20 immutable GitHub Actor artifacts.

Only transport parallelism changes. Strict receipt and manifest checks remain
in the single Learner and no update starts until all 20 files have arrived.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.rnad.shared_league_artifact_recovery import artifact_name, MAX_RECOVERY_GENERATION

REPO = "cueikusuta/p-RL1024"
MAX_ATTEMPTS = 7


def download_actors(*, generation, run_id, resolved, output, workers=4,
                    run=subprocess.run, sleep=time.sleep):
    if type(generation) is not int or not 0 <= generation <= MAX_RECOVERY_GENERATION:
        raise ValueError("Invalid Actor generation")
    if type(run_id) is not int or run_id < 1:
        raise ValueError("Invalid immutable source run")
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("Artifact download workers must be 1, 2 or 4")
    if not isinstance(resolved, list) or len(resolved) != 20:
        raise ValueError("Expected exactly 20 resolved Actor artifact names")
    for actor_id, name in enumerate(resolved):
        allowed = {artifact_name(generation, actor_id),
                   artifact_name(generation, actor_id, recovered=True)}
        if not isinstance(name, str) or name not in allowed:
            raise ValueError("Wrong Actor source artifact identity")
    if len(set(resolved)) != 20:
        raise ValueError("Duplicate immutable artifact")
    root = Path(output)
    if root.exists() and not root.is_dir():
        raise ValueError("Artifact destination is not a directory")
    root.mkdir(parents=True, exist_ok=True)

    def retrieve(actor_id):
        name = resolved[actor_id]
        destination = root / f"actor-{actor_id}"
        expected = destination / f"actor-{actor_id}.json.gz"
        for attempt in range(MAX_ATTEMPTS):
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir()
            try:
                run(["gh", "run", "download", str(run_id), "--repo", REPO,
                     "--name", name, "--dir", str(destination)],
                    check=True, timeout=300, stdout=subprocess.DEVNULL)
                if not expected.is_file() or expected.stat().st_size == 0:
                    raise ValueError("Missing immutable Actor bundle after download")
                return actor_id, expected.stat().st_size
            except (OSError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired, ValueError):
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                sleep(min(10 << attempt, 180))
        raise RuntimeError("Unreachable Actor artifact retry exhausted")

    start = time.monotonic()
    completed = {}
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(retrieve, i) for i in range(20)]
        for future in as_completed(futures):
            try:
                actor_id, size = future.result()
                completed[actor_id] = size
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise RuntimeError("Incomplete 20/20 immutable Actor downloads") from errors[0]
    if set(completed) != set(range(20)):
        raise RuntimeError("Missing one or more Actor artifacts")
    return {
        "schema": "rnad-i5-parallel-actor-artifact-transport-v1",
        "generation": generation,
        "source_run_id": run_id,
        "actors_downloaded": len(completed),
        "download_workers": workers,
        "elapsed_seconds": round(time.monotonic()-start, 3),
        "strict_trajectory_verification_pending": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--resolved", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    a = ap.parse_args()
    source = json.loads(Path(a.resolved).read_text(encoding="utf-8"))
    if not isinstance(source, dict) or set(source) != {"source_artifacts"}:
        raise ValueError("Wrong strict Artifact resolver output")
    report = download_actors(
        generation=a.generation, run_id=a.run_id,
        resolved=source["source_artifacts"], output=a.out, workers=a.workers)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
