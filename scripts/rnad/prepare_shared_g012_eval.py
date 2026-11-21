"""Pin existing G0/G1/G2 post-learner actor weights for held-out evaluation.

Only the actor bundle source_sha is resealed to this evaluation checkout.
The four trained head JSON files are byte-for-byte preserved. Never loads the
checkpoint into an optimizer and never mutates training or Champion artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ORIGIN_RUN = 44325967962
TRAIN_SOURCE = "af1c0b4c216a1846b44a0c80d6aefea94d504cd2"
SCHEMA = "rnad-integrated-policy-actor-bundle-v1"
NAMES = ("draft", "selection", "battle", "values")
ARTIFACTS = (
    "rnad-i5-cont-bootstrap-g1",
    "rnad-i5-cont-next-bootstrap-after-g1",
    "rnad-i5-cont-next-bootstrap-after-g2",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def prepare(generation, source, output):
    if type(generation) is not int or generation not in (0, 1, 2):
        raise ValueError("Only complete training generations G0, G1, G2")
    src, dst = Path(source), Path(output)
    if dst.exists():
        raise ValueError("Never overwrite an evaluation actor")
    training = read(src / "generation-manifest.json")
    original = read(src / "actor" / "manifest.json")
    engine = read("provenance/manifest.json")["sourceCommit"]
    if not (training.get("generation") == generation + 1 and
            original.get("generation") == generation + 1 and
            training.get("source_sha") == TRAIN_SOURCE and
            original.get("source_sha") == TRAIN_SOURCE and
            training.get("engine_sha") == engine and
            original.get("engine_sha") == engine and
            original.get("schema") == SCHEMA and
            training.get("main_sha256") == sha(src / "actor" / "manifest.json") and
            set(original.get("files", {})) == set(NAMES)):
        raise ValueError("Wrong source run, generation, source code, or frozen actor")
    for name in NAMES:
        expected = original["files"][name]
        if (type(expected) is not str or len(expected) != 64 or
                sha(src / "actor" / (name + ".json")) != expected):
            raise ValueError("Corrupt source head " + name)
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    if len(current) != 40 or not all(c in "0123456789abcdef" for c in current):
        raise ValueError("Unpinned evaluation HEAD")
    dst.mkdir(parents=True)
    actor = dst / "actor"
    actor.mkdir()
    for name in NAMES:
        shutil.copyfile(src / "actor" / (name + ".json"),
                        actor / (name + ".json"))
        if sha(actor / (name + ".json")) != original["files"][name]:
            raise ValueError("Weights modified during evaluation reseal")
    copied = dict(original)
    copied["source_sha"] = current
    (actor / "manifest.json").write_text(
        json.dumps(copied, sort_keys=True, separators=(",", ":")) + "\n")
    if copied["files"] != original["files"]:
        raise ValueError("Trained weights changed")
    report = {
        "schema": "rnad-i5-g012-frozen-evaluation-provenance-v1",
        "training_generation": generation,
        "checkpoint_generation": generation + 1,
        "origin_run_id": ORIGIN_RUN,
        "source_artifact_name": ARTIFACTS[generation],
        "source_code_sha": TRAIN_SOURCE,
        "evaluation_code_sha": current,
        "engine_sha": engine,
        "source_actor_manifest_sha256": training["main_sha256"],
        "evaluation_actor_manifest_sha256": sha(actor / "manifest.json"),
        "unchanged_head_file_sha256": dict(original["files"]),
        "champion_promoted": False,
    }
    (dst / "provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generation", type=int, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.generation, a.source, a.out), sort_keys=True))
