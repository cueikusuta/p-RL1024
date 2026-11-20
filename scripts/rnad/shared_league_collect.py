"""20 read-only actor shards, each exact 32 seeds x paired seats.

On failure, previously completed matches are immutable and can be reused only
when the pinned generation manifest, seed, seat and model SHA all verify.
The data originates solely from the JS receipt-backed authority Match runner.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.rnad.shared_league_contract import (
    ACTOR_SCHEMA, MATCHES_PER_ACTOR, SEEDS_PER_ACTOR, _opponent,
    _verify_events, canonical_hash, verify_actor, verify_manifest,
)
def current_git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   text=True, timeout=5).strip()


def frozen_engine_sha():
    return json.loads(Path("provenance/manifest.json").read_text())["sourceCommit"]


def read_gzip(path):
    with gzip.open(path, "rt") as source:
        return json.load(source)


def write_gzip_once(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt", compresslevel=6) as output:
        json.dump(data, output, separators=(",", ":"), allow_nan=False)


def _historical_opponent_path(root, manifest, opponent):
    """G0's SHA-pinned initial snapshot is the immutable Main Actor bundle.

    Subsequent generations use explicit past-gN directories, never an
    unverified substitute or a fallback when frozen data is missing.
    """
    ident = opponent["id"]
    if not isinstance(ident, str) or "/" in ident or ".." in ident or "\\\\" in ident:
        raise ValueError("Untrusted frozen historical opponent identifier")
    is_genesis = (manifest["generation"] == 0 and
                  ident == "integrated-initial-snapshot-v0")
    path = (Path(root) / "actor" if is_genesis else
            Path(root) / "past" / ident)
    source = path / "manifest.json"
    if (not source.is_file() or
        hashlib.sha256(source.read_bytes()).hexdigest() != opponent["sha256"] or
        (is_genesis and opponent["sha256"] != manifest["main_sha256"])):
        raise ValueError("Frozen historical generation differs from approved registry")
    return path


def _verified_match(path, *, seed, seat, match_id, manifest, opponent,
                    verify_events=True):
    row = read_gzip(path)
    if (row.get("match_id") != match_id or row.get("seed") != seed or
        row.get("learner_seat") != seat or row.get("policy_id") !=
            manifest["policy_id"] or row.get("source_sha") !=
            manifest["source_sha"] or row.get("engine_sha") !=
            manifest["engine_sha"] or row.get("main_sha256") !=
            manifest["main_sha256"] or row.get("magnet_sha256") !=
            manifest["magnet_sha256"] or row.get("opponent_sha256") !=
            opponent["sha256"] or row.get("completed") is not True or
        row.get("score", [0, 0])[row["winner"]] != 4):
        raise ValueError("Stale or incomplete accepted Match on retry")
    if verify_events:
        _verify_events(row)
    return row


def _batch_chunks(jobs, workers):
    """Deterministic, disjoint bounded work: each Node stays alive per chunk."""
    if workers not in (1, 2, 4):
        raise ValueError("Actor match workers must be 1, 2 or 4")
    return [part for i in range(workers) if (part := jobs[i::workers])]


def _run_actor_jobs(cmd, output, jobs, workers):
    chunks = _batch_chunks(jobs, workers)
    commands = []
    for index, chunk in enumerate(chunks):
        name = ("actor-batch-jobs.json" if workers == 1 else
                f"actor-batch-jobs-{index}.json")
        batch_file = output / name
        batch_file.write_text(json.dumps({
            "schema": "rnad-shared-actor-batch-jobs-v1", "jobs": chunk},
            separators=(",", ":")), encoding="utf-8")
        commands.append(cmd + ["--batch-jobs", str(batch_file)])
    if len(commands) == 1:
        subprocess.run(commands[0], check=True, timeout=1800)
        return
    # Only independent seed/seat matches run in parallel. Model snapshots
    # are read-only, outputs use exclusive creation, and retries validate
    # every completed authority artifact before accepting it.
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(
            subprocess.run, command, check=True, timeout=1800)
            for command in commands]
        first_error = None
        for future in futures:
            try:
                future.result()
            except (subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def collect(*, bootstrap, actor_id, out, smoke=False, match_workers=1):
    _batch_chunks([], match_workers)
    root = Path(bootstrap)
    manifest = json.loads((root / "generation-manifest.json").read_text())
    manifest_sha = verify_manifest(manifest)
    if (manifest["source_sha"] != current_git_sha() or
        manifest["engine_sha"] != frozen_engine_sha()):
        raise ValueError("Wrong frozen repository or game-engine checkout")
    role, opponent = _opponent(manifest, actor_id)
    output = Path(out)
    if output.exists() and not output.is_dir():
        raise ValueError("Actor output path must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        "generation": manifest["generation"],
        "actor_id": actor_id, "policy_id": manifest["policy_id"],
        "role": role, "main_sha256": manifest["main_sha256"],
        "magnet_sha256": manifest["magnet_sha256"],
        "opponent_sha256": opponent["sha256"],
        "legacy_sha256": opponent["sha256"] if role == "fixed" else None,
    }
    meta_file = output / "actor-meta.json"
    payload_bytes = json.dumps(meta, sort_keys=True).encode()
    if meta_file.exists():
        if meta_file.read_bytes() != payload_bytes:
            raise ValueError("Attempted to reuse stale actor metadata")
    else:
        meta_file.write_bytes(payload_bytes)
    opponent_path = None
    historical_path = None
    if role == "fixed":
        name = opponent["id"].removeprefix("fixed-")
        opponent_path = root / "fixed" / (name + "-v3.json")
        if (not opponent_path.is_file() or
            hashlib.sha256(opponent_path.read_bytes()).hexdigest()
            != opponent["sha256"]):
            raise ValueError("Frozen fixed opponent content absent or modified")
    if role == "historical":
        historical_path = _historical_opponent_path(root, manifest, opponent)
    magnet_path = root / "magnet"
    if not magnet_path.is_dir():
        # Initial generation 0 used the same pinned Main and Magnet.
        if manifest["generation"] != 0:
            raise ValueError("Missing next-generation frozen Magnet bundle")
        magnet_path = root / "actor"
    if hashlib.sha256((magnet_path/"manifest.json").read_bytes()).hexdigest() != manifest["magnet_sha256"]:
        raise ValueError("Frozen Magnet bytes mismatch generation manifest")
    n = 1 if smoke else SEEDS_PER_ACTOR
    # One persistent Node worker per Actor shard: fresh accepted Match and
    # seat-local GRU memories per job, immutable verified heads reused in Node.
    slots = []
    for seed in range(
        manifest["seed_start"] + actor_id * SEEDS_PER_ACTOR,
        manifest["seed_start"] + actor_id * SEEDS_PER_ACTOR + n
    ):
        for seat in (0, 1):
            match_id = f"g{manifest['generation']}-a{actor_id}-s{seed}-seat{seat}"
            path = output / "accepted" / (match_id + ".json.gz")
            path.parent.mkdir(parents=True, exist_ok=True)
            slots.append((seed, seat, match_id, path))
    cmd = [
        "node", "scripts/rnad/shared_league_actor.mjs",
        "--meta", str(meta_file), "--main", str(root / "actor"),
        "--magnet", str(magnet_path),
    ]
    if opponent_path is not None:
        cmd.extend(["--legacy", str(opponent_path)])
    elif historical_path is not None:
        cmd.extend(["--opponent", str(historical_path)])
    for attempt in range(3):
        missing = []
        for seed, seat, match_id, path in slots:
            if path.is_file():
                _verified_match(path, seed=seed, seat=seat, match_id=match_id,
                                manifest=manifest, opponent=opponent)
            else:
                missing.append({"seed": seed, "seat": seat, "out": str(path)})
        if not missing:
            break
        try:
            _run_actor_jobs(cmd, output, missing, match_workers)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Preserve only complete verified Matches from a partially failed
            # batch. The next attempt regenerates exactly the missing pairs.
            for seed, seat, match_id, path in slots:
                if path.is_file():
                    try:
                        _verified_match(path, seed=seed, seat=seat,
                                        match_id=match_id,
                                        manifest=manifest, opponent=opponent)
                    except (ValueError, KeyError, IndexError, OSError,
                            EOFError, gzip.BadGzipFile, json.JSONDecodeError):
                        path.unlink()
            if attempt == 2:
                raise
    matches = []
    for seed, seat, match_id, path in slots:
        if not path.is_file():
            raise ValueError("Actor batch did not finish required seed/seat")
        # Full-shard verify_actor below performs the strict event validation
        # once for all 64 Matches before any bundle is accepted or published.
        # Smoke lacks that full-shard barrier and therefore always validates.
        row = _verified_match(path, seed=seed, seat=seat, match_id=match_id,
                              manifest=manifest, opponent=opponent,
                              verify_events=smoke)
        matches.append(row)
        print(json.dumps({"actor_id": actor_id, "seed": seed,
                          "learner_seat": seat, "accepted_matches":len(matches),
                          "accepted_decisions": len(row["events"])}), flush=True)
    if smoke:
        return {"smoke_complete_matches": len(matches), "actor_id": actor_id,
                "winner_scores": [r["score"] for r in matches],
                "not_a_training_generation": True}
    payload = {
        "schema": ACTOR_SCHEMA, "generation": manifest["generation"],
        "actor_id": actor_id, "policy_id": manifest["policy_id"],
        "manifest_sha256": manifest_sha,
        "main_sha256": manifest["main_sha256"],
        "magnet_sha256": manifest["magnet_sha256"],
        "source_sha": manifest["source_sha"],
        "engine_sha": manifest["engine_sha"],
        "opponent_id": opponent["id"],
        "opponent_sha256": opponent["sha256"],
        "role": role, "matches": matches,
    }
    ids = verify_actor(manifest, payload)
    if len(ids) != MATCHES_PER_ACTOR:
        raise ValueError("Incomplete 64 Match actor output")
    write_gzip_once(output / f"actor-{actor_id}.json.gz", payload)
    return {"actor_id": actor_id, "accepted_matches": len(matches),
            "role": role, "manifest_sha256": manifest_sha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--actor-id", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--match-workers", type=int, choices=(1, 2, 4),
                        default=1, help="Opt-in bounded Node processes per Actor")
    args = parser.parse_args()
    print(json.dumps(collect(
        bootstrap=args.bootstrap, actor_id=args.actor_id,
        out=args.out, smoke=args.smoke,
        match_workers=args.match_workers)), flush=True)


if __name__ == "__main__":
    main()
