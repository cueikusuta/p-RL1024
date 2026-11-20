"""Bit-exact JSON numeric integrity checks for OFFLINE accepted G0 replay.

This validates existing immutable G0 Matches after staging, outside measured
Learner preprocessing time, and does not simulate or train any Matches.
"""
from __future__ import annotations

import gzip
import json
import math
import struct
import time
from pathlib import Path

from scripts.rnad.shared_league_contract import ACTORS, MATCHES_PER_ACTOR


def assert_exact_json(expected, observed, path="$"):
    """Compare the complete JSON tree, including IEEE754 sign and every bit."""
    if type(expected) is not type(observed):
        raise ValueError(f"JSON type drift at {path}: {type(expected).__name__} != {type(observed).__name__}")
    if isinstance(expected, dict):
        if expected.keys() != observed.keys():
            raise ValueError(f"JSON key drift at {path}")
        for key in expected:
            assert_exact_json(expected[key], observed[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(observed):
            raise ValueError(f"JSON array length drift at {path}")
        for i, (a, b) in enumerate(zip(expected, observed)):
            assert_exact_json(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not math.isfinite(expected) or not math.isfinite(observed):
            raise ValueError(f"Nonfinite numeric payload at {path}")
        if struct.pack("!d", expected) != struct.pack("!d", observed):
            raise ValueError(f"IEEE754 float drift at {path}")
    elif expected != observed:
        raise ValueError(f"JSON value drift at {path}")


def verify_historical_stage(actors, staged, *, actor_count=ACTORS):
    """Check every actual historic Match, not merely synthetic fixture rows."""
    if type(actor_count) is not int or not 1 <= actor_count <= ACTORS:
        raise ValueError("Invalid bounded historical Actor parity budget")
    actors, staged = Path(actors), Path(staged)
    started = time.perf_counter()
    count = 0
    for actor_id in range(actor_count):
        source = actors / f"actor-{actor_id}" / f"actor-{actor_id}.json.gz"
        with gzip.open(source, "rt", encoding="utf-8") as f:
            original = json.load(f)
        matches = original.get("matches")
        if not isinstance(matches, list) or len(matches) != MATCHES_PER_ACTOR:
            raise ValueError("Invalid G0 original Match count during parity")
        for row in matches:
            name = row["match_id"]
            with (staged / (name + ".json")).open("r", encoding="utf-8") as f:
                roundtripped = json.load(f)
            assert_exact_json(row, roundtripped, name)
            count += 1
        del original
    return {"actual_match_count": count, "ieee754_bitwise_float_parity": True,
            "full_json_tree_parity": True,
            "wall_seconds": round(time.perf_counter() - started, 3)}
