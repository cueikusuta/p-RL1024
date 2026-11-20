"""Fail-closed streaming preparation for one immutable R-NaD shared generation.

This prototype borrows the overlapping actor/learner *transport* idea from
high-throughput RL, NOT IMPALA's asynchronous policy updates. It may ingest
completed 64-Match actor shards in any order while other actors are running,
but does NOT release a Learner or optimizer before all 20 unique shards
have passed the authoritative strict contract. Live training is unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from scripts.rnad.shared_league_contract import (
    ACTORS, MATCHES_PER_ACTOR, MATCHES_PER_GENERATION, canonical_hash,
    verify_actor, verify_manifest,
)
from scripts.rnad.train import current_git_sha, frozen_engine_sha

SCHEMA = "rnad-i5-streaming-stager-v1"
READY_SCHEMA = "rnad-i5-streaming-generation-ready-v1"


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path, body):
    tmp = path.with_suffix(path.suffix+".tmp")
    if tmp.exists():
        raise ValueError("Concurrent or interrupted actor-stager writer")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, sort_keys=True, separators=(",", ":"),
                  allow_nan=False)
    os.replace(tmp, path)


class StrictStreamingStager:
    def __init__(self, bootstrap, staging):
        self.bootstrap = Path(bootstrap)
        self.staging = Path(staging)
        self.manifest = json.loads(
            (self.bootstrap/"generation-manifest.json").read_text())
        verify_manifest(self.manifest)
        if (self.manifest["source_sha"] != current_git_sha() or
                self.manifest["engine_sha"] != frozen_engine_sha()):
            raise ValueError("Streaming input violates frozen source/engine SHA")
        self.manifest_sha = canonical_hash(self.manifest)
        self.staging.mkdir(parents=True, exist_ok=True)
        self.index = self.staging/"stager-index.json"
        if self.index.exists():
            state = json.loads(self.index.read_text())
            if (state.get("schema") != SCHEMA or
                    state.get("manifest_sha256") != self.manifest_sha or
                    state.get("source_sha") != self.manifest["source_sha"] or
                    not isinstance(state.get("actors"), dict)):
                raise ValueError("Stale or forged generation staging index")
        else:
            state = {
                "schema": SCHEMA,
                "manifest_sha256": self.manifest_sha,
                "source_sha": self.manifest["source_sha"],
                "actors": {},
            }
            _atomic_json(self.index, state)
        self.state = state

    def ingest(self, actor_id, artifact):
        if type(actor_id) is not int or not 0 <= actor_id < ACTORS:
            raise ValueError("Actor shard ID outside sealed generation")
        source = Path(artifact)
        sha = _digest(source)
        key = str(actor_id)
        old = self.state["actors"].get(key)
        if old is not None:
            if old.get("sha256") != sha:
                raise ValueError("Immutable actor artifact changed")
            self._verify_staged(actor_id, old)
            return old
        with gzip.open(source, "rt") as f:
            payload = json.load(f)
        if payload.get("actor_id") != actor_id:
            raise ValueError("Wrong uploaded actor artifact ID")
        match_ids = verify_actor(self.manifest, payload)
        if len(match_ids) != MATCHES_PER_ACTOR:
            raise ValueError("Actor shard lacks 64 strictly accepted Matches")
        dst = self.staging/f"actor-{actor_id}"/f"actor-{actor_id}.json.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise ValueError("Unindexed prior actor artifact on staging disk")
        temp = dst.with_suffix(".partial")
        if temp.exists():
            raise ValueError("Interrupted previous actor staging copy")
        shutil.copyfile(source, temp)
        os.replace(temp, dst)
        if _digest(dst) != sha:
            raise ValueError("Staging copied different authoritative bytes")
        stored = {
            "actor_id": actor_id,
            "role": payload["role"],
            "sha256": sha,
            "accepted_matches": len(match_ids),
            "policy_id": payload["policy_id"],
            "source_sha": payload["source_sha"],
            "manifest_sha256": payload["manifest_sha256"],
        }
        self.state["actors"][key] = stored
        _atomic_json(self.index, self.state)
        return stored

    def _verify_staged(self, actor_id, record):
        path = (self.staging/f"actor-{actor_id}"/
                f"actor-{actor_id}.json.gz")
        if _digest(path) != record["sha256"]:
            raise ValueError("Staged immutable actor artifact damaged")

    def release(self):
        """Never unblock the Learner with fewer than 20 fully audited shards."""
        actual = self.state["actors"]
        expected = {str(i) for i in range(ACTORS)}
        if set(actual) != expected:
            missing = sorted(expected-set(actual), key=int)
            raise ValueError(f"Incomplete 20/20 actor streaming barrier: {missing}")
        roles = Counter()
        for i in range(ACTORS):
            row = actual[str(i)]
            self._verify_staged(i, row)
            if (row["actor_id"] != i or
                    row["accepted_matches"] != MATCHES_PER_ACTOR or
                    row["policy_id"] != self.manifest["policy_id"] or
                    row["source_sha"] != self.manifest["source_sha"] or
                    row["manifest_sha256"] != self.manifest_sha):
                raise ValueError("Staged Actor is not from this research generation")
            roles[row["role"]] += 1
        expected_roles = {"self": 8, "historical": 6,
                          "fixed": 4, "exploiter": 2}
        if not self.manifest["opponents"]["exploiter"]:
            expected_roles["historical"] += expected_roles.pop("exploiter")
        if dict(roles) != expected_roles:
            raise ValueError("Changed frozen opponent role quota")
        total = sum(a["accepted_matches"] for a in actual.values())
        if total != MATCHES_PER_GENERATION:
            raise ValueError("Incomplete authoritative streaming generation")
        ready = {
            "schema": READY_SCHEMA,
            "generation": self.manifest["generation"],
            "manifest_sha256": self.manifest_sha,
            "actors": [actual[str(i)] for i in range(ACTORS)],
            "accepted_complete_matches": total,
            "optimizer_steps": self.manifest["optimizer_steps"],
            "one_optimizer": True,
            "frozen_actor_policy_during_generation": True,
            "champion_promoted": False,
            "optimizer_updates_started": False,
        }
        _atomic_json(self.staging/"ready.json", ready)
        return ready


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", required=True)
    p.add_argument("--staging", required=True)
    op = p.add_mutually_exclusive_group(required=True)
    op.add_argument("--ingest", type=int)
    op.add_argument("--release", action="store_true")
    p.add_argument("--artifact")
    a = p.parse_args()
    pipe = StrictStreamingStager(a.bootstrap, a.staging)
    if a.release:
        if a.artifact:
            p.error("--artifact prohibited during release")
        print(json.dumps(pipe.release(), sort_keys=True), flush=True)
    else:
        if not a.artifact:
            p.error("--artifact required for actor ingestion")
        print(json.dumps(pipe.ingest(a.ingest, a.artifact),
                         sort_keys=True), flush=True)


if __name__=="__main__":
    main()
