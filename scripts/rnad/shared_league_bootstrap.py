"""I5 one-main three-expert initial checkpoint and SHA-pinned actor exports.

No legacy checkpoint/weight is modified; all outputs go into an empty research
directory and are immutably pinned before a single actor can be launched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import torch

from scripts.rnad.integrated_policy_v1 import (
    IntegratedPolicyV1, restore_experts, TEACHER_MANIFEST_SCHEMA,
)
from scripts.rnad.neural import (
    RNADNetwork, export_json, migrate_v2_export_to_v3,
    THREE_PHASE_FEATURE_SCHEMA_V3, THREE_PHASE_FEATURE_SCHEMA_V5,
    STATE_FEATURES_V3, ACTION_FEATURES_V3,
)
from scripts.rnad.shared_league_contract import (
    SCHEMA, POLICY_SCHEMA, verify_manifest, canonical_hash,
)
from scripts.rnad.train import current_git_sha, frozen_engine_sha

MODEL_SCHEMA = "rnad-integrated-policy-actor-bundle-v1"
CRITIC_SCHEMA = "rnad-integrated-value-heads-v1"
POLICY_ID = "RNAD-I5-IntegratedPolicyV1-Research"
A4096_SHA = "6efe87c1b30f08b964a17f6f1a8c585f1f8d37d73df9afdcf2c2cd908b23dc21"
DRAFT_SOURCE_SHA = "022aa9cd5b4ac37514df1d845ea00c11c3c13a5c"
A_SOURCE_SHA = "a7224b05eeef49650794b24546576119dffed951"
OLD_POOL_SOURCE = "f08a9587d6a7033546eab00ffa6479399ca71c2f"
OLD_NAMES = ("base128", "base256", "c005")
SEED_START = 9900001


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    if path.exists():
        raise ValueError("Refusing to overwrite immutable research file " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"),
                               allow_nan=False) + "\n")


def export_bundle(net, *, generation, original_draft, selection_template,
                  engine_sha, code_sha, out):
    """Export all three *current trainable* heads to the read-only JS actor."""
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=False)
    source_draft = json.loads(Path(original_draft).read_text())
    if (source_draft.get("schema") != "rnad-draft-deepsets-weights-v1" or
        source_draft.get("architecture") != "deepsets-soft" or
        source_draft.get("source_sha") != DRAFT_SOURCE_SHA):
        raise ValueError("Wrong frozen draft provenance")
    source_draft["training_details"] = {
        "research_shared_main_generation": generation,
        "original_teacher_source_sha": DRAFT_SOURCE_SHA,
        "not_champion": True,
    }
    source_draft["weights"] = {
        k: {"shape": list(v.shape), "values": v.detach().cpu().flatten().tolist()}
        for k, v in net.draft.state_dict().items()
    }
    write_json(directory / "draft.json", source_draft)

    source_a = json.loads(Path(selection_template).read_text())
    if (source_a.get("schema") != "rnad-i5-A-public-actor-js-weights-v1" or
        source_a.get("source_model_sha256") != A4096_SHA or
        source_a.get("source_sha") != A_SOURCE_SHA or
        source_a.get("labels_as_inference_inputs") is not False or
        source_a.get("champion_promoted") is not False):
        raise ValueError("Wrong A4096 frozen selection/actor template")
    if net.selection.embed.species.num_embeddings != source_a[
        "public_actor_vocabulary"]["sizes"]["species"]:
        raise ValueError("Selection vocab embedding differs")
    source_a["tensors"] = {
        k: {"shape": list(v.shape), "values": v.detach().cpu().flatten().tolist()}
        for k, v in net.selection.state_dict().items()
    }
    # Only the public actor fixture is reused; teacher outputs never enter actor.
    from scripts.rnad.selection_ab import pack
    a = source_a["parity_fixture"]["actor"]
    own = {
        "own_cat": torch.tensor([a["own_cats"]], dtype=torch.long),
        "foe_cat": torch.tensor([a["foe_cats"]], dtype=torch.long),
        "own_num": torch.tensor([a["own_numeric"]], dtype=torch.float32),
        "foe_num": torch.tensor([a["foe_numeric"]], dtype=torch.float32),
    }
    with torch.no_grad():
        logits, _, _ = net.selection(own)
    source_a["parity_fixture"]["logits"] = logits[0].tolist()
    source_a["research_policy_id"] = POLICY_ID
    source_a["research_generation"] = generation
    write_json(directory / "selection.json", source_a)

    export_json(net.battle, directory / "battle.json",
                feature_schema=THREE_PHASE_FEATURE_SCHEMA_V5)
    critics = {
        "schema": CRITIC_SCHEMA,
        "generation": generation,
        "policy_id": POLICY_ID,
        "tensors": {
            k: {"shape": list(v.shape),
                "values": v.detach().cpu().flatten().tolist()}
            for k, v in net.state_dict().items()
            if k.startswith(("draft_value.", "selection_value."))
        },
    }
    write_json(directory / "values.json", critics)
    files = {name: digest(directory / (name + ".json"))
             for name in ("draft", "selection", "battle", "values")}
    actor_manifest = {
        "schema": MODEL_SCHEMA, "generation": generation,
        "policy_id": POLICY_ID, "source_sha": code_sha,
        "engine_sha": engine_sha, "files": files,
    }
    write_json(directory / "manifest.json", actor_manifest)
    return digest(directory / "manifest.json")


def initialize(*, draft, draft_report, selection, selection_study,
               selection_js, battle, battle_report, old_pool,
               out, generation=0):
    if generation != 0:
        raise ValueError("Initialize exactly one shared generation zero")
    torch.set_num_threads(1)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    dr = json.loads(Path(draft_report).read_text())
    sr = json.loads(Path(selection_study).read_text())
    br = json.loads(Path(battle_report).read_text())
    legacy = json.loads((Path(old_pool) / "manifest.json").read_text())
    frozen = {
        "draft": digest(draft),
        "selection": digest(selection),
        "selection_study": digest(selection_study),
        "battle": digest(battle),
        "battle_report": digest(battle_report),
    }
    if (dr.get("model_file_sha256") != frozen["draft"] or
        dr.get("source_sha") != DRAFT_SOURCE_SHA or
        sr.get("preselected_model") != Path(selection).name or
        sr["models"][sr["preselected_model"]]["sha256"] != A4096_SHA or
        frozen["selection"] != A4096_SHA or
        sr.get("teacher_source_sha") != A_SOURCE_SHA or
        br.get("engine_sha") != frozen_engine_sha() or
        legacy.get("schema") != "rnad-i5-queued-evaluation-models-v1" or
        legacy.get("source_sha") != OLD_POOL_SOURCE):
        raise ValueError("Wrong immutable expert/fixed-source archive provenance")
    for name in OLD_NAMES:
        if legacy["models"][name] != digest(Path(old_pool) / (name + ".json")):
            raise ValueError("Old frozen opponent differs from approved source")
    torch.manual_seed(521881)
    random.seed(521881)
    model = IntegratedPolicyV1(sr["vocab"])
    teacher_manifest = {
        "schema": TEACHER_MANIFEST_SCHEMA, "policy_id": POLICY_ID,
        "selection_vocab": sr["vocab"], "sources": frozen,
    }
    provenance = restore_experts(
        model, manifest=teacher_manifest, draft_path=draft,
        selection_path=selection, selection_study_path=selection_study,
        battle_path=battle, battle_report_path=battle_report)
    # Freeze initial source identity rather than promoting an unverified teacher.
    frozen_src = copy.deepcopy(model).eval()
    with torch.no_grad():
        for value in frozen_src.parameters():
            value.requires_grad_(False)
    torch.manual_seed(521881)
    random.seed(521881)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    code_sha, engine_sha = current_git_sha(), frozen_engine_sha()
    main_sha = export_bundle(model, generation=0, original_draft=draft,
                             selection_template=selection_js,
                             engine_sha=engine_sha, code_sha=code_sha,
                             out=root / "actor")
    fixed_rows = []
    fixed_dir = root / "fixed"
    fixed_dir.mkdir()
    for name in OLD_NAMES:
        old = RNADNetwork(state_features=STATE_FEATURES_V3,
                          action_features=ACTION_FEATURES_V3)
        original_sha = migrate_v2_export_to_v3(
            old, Path(old_pool) / (name + ".json"))
        if original_sha != legacy["models"][name]:
            raise ValueError("Fixed opponent source model SHA changed")
        path = fixed_dir / (name + "-v3.json")
        export_json(old, path, feature_schema=THREE_PHASE_FEATURE_SCHEMA_V3)
        fixed_rows.append({"id": "fixed-" + name, "sha256": digest(path)})
    generation_manifest = {
        "schema": SCHEMA,
        "generation": 0,
        "policy_id": POLICY_ID,
        "policy_manifest_sha256": main_sha,
        "main_sha256": main_sha, "magnet_sha256": main_sha,
        "source_sha": code_sha,
        "engine_sha": engine_sha,
        "seed_start": SEED_START,
        "feature_schema": POLICY_SCHEMA,
        "actor_count": 20, "seeds_per_actor": 32,
        "optimizer_batch_matches": 32, "optimizer_steps": 40,
        "opponents": {
            "historical": [{"id": "integrated-initial-snapshot-v0",
                            "sha256": main_sha}],
            "fixed": fixed_rows, "exploiter": [],
        },
    }
    verify_manifest(generation_manifest)
    write_json(root / "generation-manifest.json", generation_manifest)
    checkpoint = {
        "schema": "rnad-shared-learner-full-state-v1",
        "policy_id": POLICY_ID, "generation": 0, "step": 0,
        "model": model.state_dict(),
        "target": frozen_src.state_dict(),
        "magnet": frozen_src.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "source_sha": code_sha,
        "engine_sha": engine_sha,
        "source_expert_sha256": frozen,
        "training_match_ids": [],
        "main_sha256": main_sha,
        "manifest_sha256": canonical_hash(generation_manifest),
        "is_champion": False,
    }
    save = root / "checkpoint-g0.pt"
    torch.save(checkpoint, save)
    write_json(root / "checkpoint-index.json", {
        "schema": "rnad-shared-learner-index-v1",
        "checkpoint_file": save.name,
        "sha256": digest(save),
        "generation": 0, "applied_matches": 0, "completed_steps": 0,
        "is_champion": False,
    })
    write_json(root / "teacher-provenance.json", provenance)
    return generation_manifest


def main():
    p = argparse.ArgumentParser()
    for name in ("draft","draft-report","selection","selection-study",
                 "selection-js","battle","battle-report","old-pool","out"):
        p.add_argument("--" + name, required=True)
    args = vars(p.parse_args())
    data = initialize(
        draft=args["draft"],draft_report=args["draft_report"],
        selection=args["selection"],selection_study=args["selection_study"],
        selection_js=args["selection_js"],battle=args["battle"],
        battle_report=args["battle_report"],old_pool=args["old_pool"],
        out=args["out"])
    print(json.dumps({"policy_id": data["policy_id"],
                      "generation": data["generation"],
                      "frozen_teacher_weights_verified": True,
                      "actor_count": data["actor_count"],
                      "planned_accepted_matches": 1280,
                      "actual_training_started": False}),flush=True)


if __name__ == "__main__":
    main()
