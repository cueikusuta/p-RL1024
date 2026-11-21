"""Prepare a new versioned C-v6 league bootstrap, without altering prior runs.

The input MUST already be an authenticated, source-pinned next-generation v5
bootstrap at this repository HEAD. This script does NOT bridge source SHAs or
reconstruct incomplete original artifacts. It only expands the Battle feature
layers and the corresponding Adam moments after strict complete-state checks.

Out is a new output directory. Original v5 bootstrap, model and metadata are
read-only. The 20-Actor training/evaluation launch is separately gated.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch

from scripts.rnad.integrated_policy_v1 import IntegratedPolicyV1
from scripts.rnad.neural import THREE_PHASE_FEATURE_SCHEMA_V5, THREE_PHASE_FEATURE_SCHEMA_V6
from scripts.rnad.shared_league_bootstrap import (
    export_bundle, digest, write_json, POLICY_ID,
)
from scripts.rnad.shared_league_contract import (
    verify_manifest, canonical_hash,
)
from scripts.rnad.train import current_git_sha
from scripts.rnad.upgrade_battle_v6 import upgrade_checkpoint


def read_json(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prepare(src, out):
    src, out = Path(src), Path(out)
    require(src.is_dir() and not out.exists(), "Expected immutable input and new output")
    old = read_json(src / "generation-manifest.json")
    verify_manifest(old)
    require(old.get("battle_feature_schema", THREE_PHASE_FEATURE_SCHEMA_V5)
            == THREE_PHASE_FEATURE_SCHEMA_V5 and
            old.get("league_opponent_profile") is None,
            "C changes only Battle features; opponent mix must stay original")
    source_sha = current_git_sha()
    require(old["source_sha"] == source_sha,
            "Prior bridge must pin source to this exact C commit")
    require(old["policy_id"] == POLICY_ID,
            "Do not migrate a different research policy")
    current_actor = src / "actor"
    current_magnet = src / "magnet"
    require(current_actor.is_dir() and current_magnet.is_dir(),
            "Missing complete generation Main/Magnet bundles")
    for directory, key in ((current_actor, "main_sha256"),
                           (current_magnet, "magnet_sha256")):
        require(digest(directory / "manifest.json") == old[key],
                "Frozen actor bundle SHA identity mismatch")
        payload = read_json(directory / "manifest.json")
        require(payload.get("source_sha") == source_sha and
                payload.get("engine_sha") == old["engine_sha"],
                "Frozen actor/engine source mismatch")
        for name, sha in payload.get("files", {}).items():
            require(digest(directory / f"{name}.json") == sha,
                    "Frozen head file SHA mismatch " + name)
        battle = read_json(directory / "battle.json")
        require(battle.get("feature_schema") == THREE_PHASE_FEATURE_SCHEMA_V5 and
                battle.get("state_features") == 96 and
                battle.get("action_features") == 48,
                "Refusing to migrate non-v5 Battle head")
    index = read_json(src / "checkpoint-index.json")
    index_path = src / "checkpoint-g0.pt"
    require(index_path.is_file() and index.get("sha256") == digest(index_path),
            "Untrusted input full-state checkpoint")
    ckpt = torch.load(index_path, weights_only=True, map_location="cpu")
    require(ckpt.get("schema") == "rnad-shared-learner-full-state-v1" and
            ckpt.get("source_sha") == source_sha and
            ckpt.get("generation") == old["generation"] and
            ckpt.get("step") == 0 and
            ckpt.get("is_champion") is False and
            ckpt.get("manifest_sha256") == canonical_hash(old) and
            ckpt.get("main_sha256") == old["main_sha256"] and
            ckpt.get("magnet_sha256") == old["magnet_sha256"],
            "Current checkpoint is not a complete source-pinned next generation")
    selection = read_json(current_actor / "selection.json")
    vocab = selection.get("public_actor_vocabulary")
    require(isinstance(vocab, dict),
            "Missing frozen public vocabulary for three-phase model")
    old_net = IntegratedPolicyV1(vocab, battle_schema=THREE_PHASE_FEATURE_SCHEMA_V5)
    new_net = IntegratedPolicyV1(vocab, battle_schema=THREE_PHASE_FEATURE_SCHEMA_V6)
    old_net.load_state_dict(ckpt["model"], strict=True)
    converted = upgrade_checkpoint(ckpt, old_net, new_net)
    # Verify all model heads before writing; never infer a changed tensor shape.
    new_net.load_state_dict(converted["model"], strict=True)
    new_magnet = IntegratedPolicyV1(vocab, battle_schema=THREE_PHASE_FEATURE_SCHEMA_V6)
    new_magnet.load_state_dict(converted["magnet"], strict=True)
    dst_tmp = out.with_name(out.name + ".incomplete")
    require(not dst_tmp.exists(), "Stale incomplete output must not be overwritten")
    shutil.copytree(src, dst_tmp)
    try:
        shutil.rmtree(dst_tmp / "actor")
        shutil.rmtree(dst_tmp / "magnet")
        source_draft = dst_tmp / "original-draft.json"
        source_selection = dst_tmp / "original-selection-js.json"
        require(source_draft.is_file() and source_selection.is_file(),
                "Missing frozen original expert source for valid export")
        actor_sha = export_bundle(
            new_net, generation=old["generation"],
            original_draft=source_draft, selection_template=source_selection,
            engine_sha=old["engine_sha"], code_sha=source_sha,
            out=dst_tmp / "actor",
        )
        magnet_sha = export_bundle(
            new_magnet, generation=old["generation"],
            original_draft=source_draft, selection_template=source_selection,
            engine_sha=old["engine_sha"], code_sha=source_sha,
            out=dst_tmp / "magnet",
        )
        new_manifest = copy.deepcopy(old)
        new_manifest["battle_feature_schema"] = THREE_PHASE_FEATURE_SCHEMA_V6
        new_manifest["main_sha256"] = actor_sha
        new_manifest["policy_manifest_sha256"] = actor_sha
        new_manifest["magnet_sha256"] = magnet_sha
        manifest_sha = verify_manifest(new_manifest)
        (dst_tmp / "generation-manifest.json").unlink()
        write_json(dst_tmp / "generation-manifest.json", new_manifest)
        converted["source_sha"] = source_sha
        converted["main_sha256"] = actor_sha
        converted["magnet_sha256"] = magnet_sha
        converted["manifest_sha256"] = manifest_sha
        converted["battle_feature_schema"] = THREE_PHASE_FEATURE_SCHEMA_V6
        (dst_tmp / "checkpoint-g0.pt").unlink()
        torch.save(converted, dst_tmp / "checkpoint-g0.pt")
        index["sha256"] = digest(dst_tmp / "checkpoint-g0.pt")
        (dst_tmp / "checkpoint-index.json").unlink()
        write_json(dst_tmp / "checkpoint-index.json", index)
        report_file = dst_tmp / "handoff-report.json"
        if report_file.exists():
            report = read_json(report_file)
            report["new_generation_manifest_sha256"] = manifest_sha
            report["new_full_checkpoint_sha256"] = index["sha256"]
            report["c_v6_feature_upgrade"] = {
                "source_sha":source_sha,
                "previous_battle_feature_schema":THREE_PHASE_FEATURE_SCHEMA_V5,
                "new_battle_feature_schema":THREE_PHASE_FEATURE_SCHEMA_V6,
                "state_width":384,"action_width":160,
                "original_checkpoint_sha256":digest(index_path),
            }
            report_file.unlink()
            write_json(report_file, report)
        pins = dst_tmp / "file-pins.sha256"
        if pins.exists():
            pins.unlink()
        rows = sorted(
            f"{digest(p)}  {p.relative_to(dst_tmp).as_posix()}"
            for p in dst_tmp.rglob("*") if p.is_file()
        )
        pins.write_text("\n".join(rows) + "\n")
        # Final gate reopens all sealed output and checks the new manifest.
        require(verify_manifest(read_json(dst_tmp / "generation-manifest.json"))
                == manifest_sha and
                digest(dst_tmp / "actor" / "manifest.json") == actor_sha and
                digest(dst_tmp / "magnet" / "manifest.json") == magnet_sha and
                digest(dst_tmp / "checkpoint-g0.pt") == index["sha256"],
                "Sealed v6 bootstrap content did not verify")
        dst_tmp.rename(out)
    except BaseException:
        shutil.rmtree(dst_tmp, ignore_errors=True)
        raise
    return {
        "generation":old["generation"],"source_sha":source_sha,
        "feature_schema":THREE_PHASE_FEATURE_SCHEMA_V6,
        "main_sha256":actor_sha,"magnet_sha256":magnet_sha,
        "manifest_sha256":manifest_sha,
        "full_checkpoint_sha256":index["sha256"],
        "previous_checkpoint_sha256":digest(index_path),
        "research_only":True,"champion_promoted":False,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--src",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.src,args.out),sort_keys=True))
