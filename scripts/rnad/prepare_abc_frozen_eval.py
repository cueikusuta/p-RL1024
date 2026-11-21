"""Strict read-only A/B/C frozen actor-bundle re-sealing for paired evaluation.

Comparison is fixed before a single evaluation Match:
  A = original G195-trained actor, from next-bootstrap G196;
  B = B G198-trained actor, from next-bootstrap G199;
  C = C G198-trained actor, from next-bootstrap G199.
This compares frozen checkpoint *outcomes*, NOT causal training effects.
All four expert-head JSON bytes must survive the source-identity reseal.
No checkpoint, optimizer state, source artifact or Champion is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

HEADS = ("battle", "draft", "selection", "values")
SOURCE = {
    "A": {"generation": 196,
          "source_sha": "4ac6ad652365318aa4f8b9736dffdd00cf696915",
          "run_id": 44860357036, "artifact_id": 14702954785,
          "artifact_name": "rnad-i5-cont-next-bootstrap-after-g195",
          "battle_schema": "rnad-three-phase-features-v5",
          "expected_main_sha256": None, "opponent_profile": None},
    "B": {"generation": 199,
          "source_sha": "1b3bfb8273a8d01925d5f3e8a3e6fb7fc977299a",
          "run_id": 44899166361, "artifact_id": 14706787396,
          "artifact_name": "rnad-i5-cont-next-bootstrap-after-g198",
          "battle_schema": "rnad-three-phase-features-v5",
          "expected_main_sha256":
             "773e1ffc36f538704e8f9d230d2055075a4703f08413669fb4f4a20c2e55c36c",
          "opponent_profile": "rnad-i5-opponents-b-v1"},
    "C": {"generation": 199,
          "source_sha": "a98bd7cf73b7d3fc2e7d4a9b81764833843653a7",
          "run_id": 44897382403, "artifact_id": 14706377232,
          "artifact_name": "rnad-i5-cont-next-bootstrap-after-g198",
          "battle_schema": "rnad-three-phase-features-v6",
          "expected_main_sha256":
             "67b019669c8c902e672ef372c6d2c1e9d52141594672297b3574dcbe160343fb",
          "opponent_profile": None},
}
SEED_START = 120000001
SEED_BLOCKS = 2048
ALPHA = 0.05
ENGINE_MARKER = "rnad-integrated-policy-actor-bundle-v1"


def require(condition, message):
    if not condition:
        raise ValueError("A/B/C frozen evaluation: " + message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def verify_file_pins(root):
    root = Path(root)
    pins = root / "file-pins.sha256"
    require(pins.is_file(), "missing source bootstrap file pins")
    observed = set()
    for entry in pins.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", entry)
        require(match is not None, "invalid source pin line")
        expected, name = match.groups()
        p = (root / name).resolve()
        require(root.resolve() in p.parents and p.is_file(),
                "source pin escaped artifact directory: " + name)
        require(name not in observed and digest(p) == expected,
                "unverifiable source artifact byte: " + name)
        observed.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p != pins}
    require(observed == actual, "incomplete source file-pins inventory")


def prepare(sources, out):
    out = Path(out)
    require(not out.exists(), "evaluation directory must be new")
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    require(re.fullmatch(r"[a-f0-9]{40}", source_sha),
            "unverifiable evaluation source SHA")
    temp = out.with_name(out.name + ".incomplete")
    require(not temp.exists(), "refusing dirty evaluation staging")
    temp.mkdir(parents=True)
    sealed = {}
    audit = {}
    engine = None
    try:
        for label in ("A", "B", "C"):
            origin = SOURCE[label]
            root = Path(sources[label])
            verify_file_pins(root)
            generation = read_json(root / "generation-manifest.json")
            actor_dir = root / "actor"
            model = read_json(actor_dir / "manifest.json")
            require(generation.get("generation") == origin["generation"] and
                    model.get("generation") == origin["generation"] and
                    generation.get("source_sha") == origin["source_sha"] and
                    model.get("source_sha") == origin["source_sha"] and
                    generation.get("policy_id") == model.get("policy_id") and
                    generation.get("feature_schema") == "rnad-integrated-three-phase-v1",
                    "wrong generation/source/policy for " + label)
            require(generation.get("league_opponent_profile") ==
                    origin["opponent_profile"] and
                    generation.get("battle_feature_schema",
                        "rnad-three-phase-features-v5") == origin["battle_schema"],
                    "wrong opponent or Battle feature arm for " + label)
            if label == "B":
                require({x.get("id") for x in generation["opponents"]["fixed"]} ==
                    {"fixed-aqbase001", "fixed-coretrioaq001",
                     "fixed-base256", "fixed-c005"},
                    "B does not contain the approved fixed opponent set")
            main_sha = digest(actor_dir / "manifest.json")
            require(main_sha == generation.get("main_sha256") ==
                    generation.get("policy_manifest_sha256") and
                    (origin["expected_main_sha256"] is None or
                     main_sha == origin["expected_main_sha256"]),
                    "frozen actor does not match registered Main SHA for " + label)
            require(model.get("schema") == ENGINE_MARKER and
                    set(model.get("files", {})) == set(HEADS),
                    "incompatible actor manifest for " + label)
            if engine is None:
                engine = generation.get("engine_sha")
            require(engine and generation.get("engine_sha") == engine and
                    model.get("engine_sha") == engine,
                    "game engine changed across comparison arms")
            index = read_json(root / "checkpoint-index.json")
            cp = root / "checkpoint-g0.pt"
            require(cp.is_file() and digest(cp) == index.get("sha256") and
                    index.get("generation") == origin["generation"],
                    "unverified full checkpoint for " + label)
            dest = temp / label
            dest.mkdir()
            saved_heads = {}
            for head in HEADS:
                f = actor_dir / (head + ".json")
                pinned = model["files"][head]
                require(re.fullmatch(r"[a-f0-9]{64}", pinned) and
                        digest(f) == pinned,
                        "tampered frozen " + head + " head for " + label)
                shutil.copyfile(f, dest / f.name)
                require(digest(dest / f.name) == pinned,
                        "model head bytes drifted during copy: " + label)
                saved_heads[head] = pinned
            battle = read_json(dest / "battle.json")
            width = (384, 160) if label == "C" else (96, 48)
            require(battle.get("feature_schema") == origin["battle_schema"] and
                    (battle.get("state_features"),
                     battle.get("action_features")) == width,
                    "wrong frozen Battle dimensions for " + label)
            resealed = dict(model, source_sha=source_sha)
            save_json(dest / "manifest.json", resealed)
            after = read_json(dest / "manifest.json")
            require(after["files"] == saved_heads and
                    after["source_sha"] == source_sha and
                    after["engine_sha"] == engine,
                    "reseal changed weights or source identity")
            sealed[label] = digest(dest / "manifest.json")
            audit[label] = {
                "origin": origin, "old_main_manifest_sha256": main_sha,
                "resealed_manifest_sha256": sealed[label],
                "full_checkpoint_sha256": index["sha256"],
                "head_sha256": saved_heads,
                "evaluation_code_source_sha": source_sha,
                "generation_run_matches_in_common": label != "A",
            }
        plan = {
            "schema": "rnad-i5-prospective-paired-league-plan-v1",
            "stage": "final", "evaluation_type": "shared-opponent",
            "source_sha": source_sha, "engine_sha": engine,
            "opponent_id": "AQ-Base-001",
            "seed_start": SEED_START, "seed_blocks": SEED_BLOCKS,
            "familywise_alpha": ALPHA, "confidence_level": 1 - ALPHA,
            "method": "paired-seed-block-student-t-bonferroni",
            "fixed_sample_size": True, "model_selection_before_holdout": True,
            "independent_holdout_precommitted": True,
            "automatic_champion_promotion": False,
            "minimum_practical_gain": 0,
            "model_sha256_by_id": sealed,
            "comparisons": [
                {"candidate_model_id": "B", "baseline_model_id": "A"},
                {"candidate_model_id": "C", "baseline_model_id": "A"},
                {"candidate_model_id": "B", "baseline_model_id": "C"}
            ],
            "excluded_seed_ranges": [
                {"start": 9900001, "count": 150000},
                {"start": 9500001, "count": 20000},
                {"start": 1970001, "count": 20000}
            ]
        }
        save_json(temp / "plan.json", plan)
        save_json(temp / "audit.json", {
            "schema": "rnad-i5-abc-frozen-input-audit-v1",
            "comparison": "AQ-Base-001 common frozen opponent",
            "training_seed_count_as_evidence": 0,
            "new_training": False, "automated_champion_promotion": False,
            "warning": "A was not retrained; B/C have three more training generations.",
            "models": audit
        })
        digest_table = {p.relative_to(temp).as_posix():digest(p)
                        for p in temp.rglob("*") if p.is_file()}
        save_json(temp / "sha256-index.json", digest_table)
        temp.rename(out)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"source_sha": source_sha, "seed_start": SEED_START,
            "seed_blocks": SEED_BLOCKS, "models": sealed,
            "opponent": "AQ-Base-001", "comparisons": 3}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("a", "b", "c", "out"):
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(
        {"A": args.a, "B": args.b, "C": args.c}, args.out),sort_keys=True))
