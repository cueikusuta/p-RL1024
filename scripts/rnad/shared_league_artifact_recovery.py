"""Fail-closed planning for GitHub Artifact v4 actor-shard recovery.

Planning reads ONLY the Artifact manifest/listing. A missing logical actor
may be replayed with the SAME code SHA / generation / seed+seat. Strict
acceptance of the actual Match contents remains in shared_league_learner.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

ACTORS = 20
# B+C 100h approved continuation has fixed waves through G598.
# Keep the bound explicit: generation 599 and beyond fail closed.
MAX_RECOVERY_GENERATION = 598
PREFIX = "rnad-i5-cont-g"
FULL_NAME = re.compile(
    r"rnad-i5-cont-g(\d+)-(recovered-)?actor-(\d+)\Z"
)


def artifact_name(generation: int, actor_id: int, recovered=False) -> str:
    if (type(generation) is not int or not 0 <= generation <= MAX_RECOVERY_GENERATION or
        type(actor_id) is not int or not 0 <= actor_id < ACTORS):
        raise ValueError("Invalid shared league generation or logical actor")
    return (f"{PREFIX}{generation}-" +
            ("recovered-" if recovered else "") +
            f"actor-{actor_id}")


def _names(generation: int, artifacts):
    if not isinstance(artifacts, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("name"), str)
        or type(row.get("expired", False)) is not bool
        for row in artifacts
    ):
        raise ValueError("Invalid GitHub artifact listing")
    names = [row["name"] for row in artifacts if not row.get("expired", False)]
    if any(row["name"].startswith(f"{PREFIX}{generation}-") and
           row.get("expired", False) for row in artifacts):
        raise ValueError("Expired current-generation Actor result")
    relevant = [name for name in names
                if name.startswith(f"{PREFIX}{generation}-") and
                ("actor-" in name)]
    if any((m := FULL_NAME.fullmatch(name)) is None or
           int(m.group(1)) != generation or int(m.group(3)) >= ACTORS
           for name in relevant):
        raise ValueError("Unexpected generation Actor artifact identity")
    if any(count != 1 for count in Counter(relevant).values()):
        raise ValueError("Duplicate artifact name in GitHub listing")
    return set(relevant)


def missing_actor_ids(generation: int, artifacts):
    """Idempotent on rerun: keep a previously rescued immutable shard."""
    available = _names(generation, artifacts)
    missing = []
    for i in range(ACTORS):
        original = artifact_name(generation, i)
        recovered = artifact_name(generation, i, recovered=True)
        if original in available and recovered in available:
            raise ValueError(f"Ambiguous original/recovered Actor {i} artifacts")
        if original not in available and recovered not in available:
            missing.append(i)
    return missing


def accepted_artifact_names(generation: int, artifacts):
    """Verify exactly one original or rescued artifact for every actor."""
    available = _names(generation, artifacts)
    resolved = []
    for i in range(ACTORS):
        candidates = (artifact_name(generation, i),
                      artifact_name(generation, i, recovered=True))
        seen = [candidate for candidate in candidates if candidate in available]
        if len(seen) != 1:
            raise ValueError(
                f"Logical actor {i} has {len(seen)} immutable sources; "
                "cannot train a single-writer Main")
        resolved.append(seen[0])
    if len(available) != ACTORS or set(resolved) != available:
        raise ValueError("Unexpected or mixed actor shards")
    return resolved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["inspect", "resolve"])
    p.add_argument("--generation", required=True, type=int)
    p.add_argument("--artifact-json", required=True)
    a = p.parse_args()
    with open(a.artifact_json, encoding="utf8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("artifacts")
    if a.phase == "inspect":
        missing = missing_actor_ids(a.generation, data)
        print(json.dumps({"missing": missing, "missing_count": len(missing)},
                         separators=(",", ":"), sort_keys=True))
    else:
        print(json.dumps({"source_artifacts": accepted_artifact_names(
            a.generation, data)},
            separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
