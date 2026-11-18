"""RNAD-I5 IntegratedPolicyV1 research model (3 independent heads, one writer).

This is a *trainable* shared policy object, NOT a runnable league or Champion.
Training uses one optimizer over its three heads, avoiding shape-incompatible
state-dict averaging. Input tensors must be extracted from seat-scoped accepted
observations. Live JS adapter and frozen-source restoration are separate gates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from scripts.rnad.train_draft_deepsets import DeepSetsDraft, FEATURE_WIDTH
from scripts.rnad.selection_ab import DecomposedTeacher, TRIPLES
from scripts.rnad.neural import (
    RNADNetwork, THREE_PHASE_FEATURE_SCHEMA_V5,
    STATE_FEATURES_V5, ACTION_FEATURES_V5,
    load_exported_json,
)

SCHEMA = "rnad-integrated-policy-v1"
TEACHER_MANIFEST_SCHEMA = "rnad-integrated-teacher-restore-v1"
SOURCE_SCHEMA_A4096 = "rnad-i5-selection-A-4096-supervised-study-v1"
BATTLE_REPORT_PROFILE = "battle-focus"
FLOAT = torch.float32
PHASES = ("draft", "selection", "battle")


def _fail(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified(path, sha256):
    _fail(isinstance(sha256, str) and len(sha256) == 64 and
          all(c in "0123456789abcdef" for c in sha256),
          "Missing immutable teacher bytes SHA256")
    _fail(Path(path).is_file() and _digest(path) == sha256,
          "Frozen teacher bytes do not match manifest SHA256")


def _no_privileged_keys(features, allowed):
    _fail(type(features) is dict and set(features) == set(allowed),
          "Unexpected field/privileged teacher or opponent input")


class IntegratedPolicyV1(nn.Module):
    """Three heads + two additional critics, within ONE trainable policy.

    The draft and selection specialists are feed-forward and do not consume
    their score targets at inference time. Battle has one recurrent memory per
    seat *managed by the caller*. A Match runner must reset all such memories
    between Matches/rounds as appropriate and only advance after acceptance.
    """
    def __init__(self, public_vocab):
        super().__init__()
        _fail(isinstance(public_vocab, dict) and
              isinstance(public_vocab.get("sizes"), dict) and
              all(type(public_vocab["sizes"].get(k)) is int and
                  public_vocab["sizes"][k] >= 2
                  for k in ("species", "names", "abilities", "items",
                            "moves", "types")),
              "Missing registered public selection vocabulary")
        self.policy_schema = SCHEMA
        self.draft = DeepSetsDraft()
        self.selection = DecomposedTeacher(public_vocab)
        self.battle = RNADNetwork(state_features=STATE_FEATURES_V5,
                                  action_features=ACTION_FEATURES_V5)
        self.draft_value = nn.Sequential(
            nn.Linear(48 + 4, 64), nn.ReLU(), nn.Linear(64, 1),
            nn.Tanh())
        self.selection_value = nn.Sequential(
            nn.Linear(2 * 64, 64), nn.ReLU(), nn.Linear(64, 1),
            nn.Tanh())

    def initial_memory(self):
        return self.battle.initial_memory()

    def forward(self, phase, seat_scoped, legal_actions, per_seat_memory=None):
        _fail(phase in PHASES, "Unknown phase")
        if phase == "draft":
            _no_privileged_keys(seat_scoped, ("pool", "extra"))
            pool, extra = seat_scoped["pool"], seat_scoped["extra"]
            codes = legal_actions
            _fail(isinstance(pool, torch.Tensor) and
                  pool.ndim == 2 and pool.shape[1] == FEATURE_WIDTH and
                  isinstance(extra, torch.Tensor) and extra.ndim == 2 and
                  extra.shape[1] == 4 and
                  isinstance(codes, torch.Tensor) and
                  codes.ndim == 2 and codes.shape[1] == 6 and
                  codes.shape[0] == extra.shape[0] and
                  codes.dtype == torch.long and codes.numel() > 0 and
                  int(codes.min()) >= 0 and int(codes.max()) < len(pool) and
                  all(len(set(row)) == 6 for row in codes.tolist()),
                  "Invalid authoritative six-Pokemon draft candidate tensor")
            logits = self.draft(pool, codes, extra)
            pooled = self.draft.phi(pool).mean(dim=0)
            critic_input = torch.cat((pooled, extra.mean(dim=0)), dim=0)
            value = self.draft_value(critic_input).reshape(())
            next_memory = per_seat_memory
        elif phase == "selection":
            _no_privileged_keys(seat_scoped,
                                ("own_cat", "foe_cat", "own_num", "foe_num"))
            _fail(legal_actions == TRIPLES,
                  "Selection must use canonical authoritative 60 actions")
            _fail(all(isinstance(seat_scoped[k], torch.Tensor)
                      for k in ("own_cat", "foe_cat", "own_num", "foe_num")),
                  "Invalid public/own selection actor")
            _fail(tuple(seat_scoped["own_cat"].shape[1:]) == (6, 9) and
                  tuple(seat_scoped["foe_cat"].shape[1:]) == (6, 3) and
                  tuple(seat_scoped["own_num"].shape[1:]) == (6, 16) and
                  tuple(seat_scoped["foe_num"].shape[1:]) == (6, 6) and
                  seat_scoped["own_cat"].shape[0] == 1 and
                  seat_scoped["foe_cat"].shape[0] == 1 and
                  seat_scoped["own_num"].shape[0] == 1 and
                  seat_scoped["foe_num"].shape[0] == 1,
                  "Selection inputs must be a single public/self seat")
            logits, _, _ = self.selection(seat_scoped)
            own, opponent_public = self.selection.embed(seat_scoped)
            critic_input = torch.cat(
                (own.mean(dim=1), opponent_public.mean(dim=1)), dim=-1)
            value = self.selection_value(critic_input).reshape(())
            logits = logits[0]
            next_memory = per_seat_memory
        else:
            _no_privileged_keys(seat_scoped, ("state",))
            state = seat_scoped["state"]
            candidates = legal_actions
            _fail(isinstance(state, torch.Tensor) and
                  state.ndim == 2 and tuple(state.shape) == (1, STATE_FEATURES_V5) and
                  isinstance(candidates, torch.Tensor) and
                  candidates.ndim == 3 and candidates.shape[0] == 1 and
                  candidates.shape[1] > 0 and
                  candidates.shape[2] == ACTION_FEATURES_V5 and
                  isinstance(per_seat_memory, torch.Tensor) and
                  tuple(per_seat_memory.shape) == (1, self.battle.hidden_size),
                  "Invalid seat-local battle tensor or recurrent memory")
            logits, value, next_memory = self.battle(
                state, candidates, per_seat_memory)
            logits, value = logits[0], value.reshape(())
        _fail(logits.ndim == 1 and logits.numel() > 0 and
              bool(torch.isfinite(logits).all()) and
              bool(torch.isfinite(value)),
              "Nonfinite or empty integrated policy prediction")
        return logits, value, next_memory

    def immutable_actor_copy(self):
        frozen = copy.deepcopy(self).eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        return frozen


def restore_experts(model, *, manifest, draft_path, selection_path,
                    selection_study_path, battle_path, battle_report_path):
    """Load frozen expert *weights* into independent trainable heads.

    Caller obtains the files from verified workflow artifacts. Their exact
    SHA256 must be supplied in a manifest registered before restoration.
    Checkpoint source weights are NOT modified or rewritten.
    """
    _fail(isinstance(model, IntegratedPolicyV1) and
          isinstance(manifest, dict) and
          manifest.get("schema") == TEACHER_MANIFEST_SCHEMA and
          isinstance(manifest.get("policy_id"), str) and
          bool(manifest["policy_id"]),
          "Invalid three-head teacher restore contract")
    sources = manifest.get("sources")
    _fail(isinstance(sources, dict) and
          set(sources) == {"draft", "selection", "selection_study",
                           "battle", "battle_report"},
          "Exactly five frozen expert sources required")
    for name, path in (
        ("draft", draft_path), ("selection", selection_path),
        ("selection_study", selection_study_path),
        ("battle", battle_path), ("battle_report", battle_report_path),
    ):
        _verified(path, sources[name])
    from scripts.rnad.draft_battle_learning_512_recovery import (
        load_model_draft, SOURCE,
    )
    restored_draft = load_model_draft(draft_path, sources["draft"])
    model.draft.load_state_dict(restored_draft.state_dict(), strict=True)
    study = json.loads(Path(selection_study_path).read_text())
    chosen = study.get("preselected_model")
    row = study.get("models", {}).get(chosen, {})
    _fail(study.get("schema") == SOURCE_SCHEMA_A4096 and
          row.get("filename") == chosen and
          row.get("sha256") == sources["selection"] and
          isinstance(study.get("vocab"), dict) and
          study["vocab"] == manifest.get("selection_vocab") and
          study.get("champion_promoted") is False and
          study.get("league_training_started") is False,
          "Selection A4096 frozen study/vocabulary provenance mismatch")
    restored = torch.load(selection_path, map_location="cpu",
                          weights_only=True)
    model.selection.load_state_dict(restored, strict=True)
    from scripts.rnad.draft_battle_learning_512 import restore_battle
    battle, report = restore_battle(
        battle_path, battle_report_path, "IntegratedPolicyV1")
    _fail(report.get("feature_schema") == THREE_PHASE_FEATURE_SCHEMA_V5 and
          report.get("phase_profile") == BATTLE_REPORT_PROFILE and
          report.get("engine_sha") == study.get("engine_sha") and
          report.get("source_sha") == SOURCE,
          "Battle expert source, engine or feature mismatch")
    model.battle.load_state_dict(battle.state_dict(), strict=True)
    # The two critic heads have no pretrained weights; explicitly report
    # the fact rather than implying the expert training supplied values.
    return {
        "policy_id": manifest["policy_id"],
        "source_sha256": dict(sources),
        "draft_value_initialized_new": True,
        "selection_value_initialized_new": True,
        "battle_value_restored": True,
        "training_started": False,
    }
