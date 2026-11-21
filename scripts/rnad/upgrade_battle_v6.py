"""Read-only v5->v6 Battle expansion for the C research arm.

This transforms model, target, magnet and Adam moments in memory. The caller
must seal a *new* source-pinned bootstrap/manifest; never overwrite the v5
checkpoint or pass a converted dict to the original immutable workflow.
"""
from __future__ import annotations

import copy

import torch

from scripts.rnad.neural import (
    STATE_FEATURES_V5, ACTION_FEATURES_V5,
    STATE_FEATURES_V6, ACTION_FEATURES_V6, HIDDEN,
)

SHAPES = {
    "battle.state_proj.weight": ((HIDDEN, STATE_FEATURES_V5),
                                 (HIDDEN, STATE_FEATURES_V6)),
    "battle.policy_hidden.weight": ((HIDDEN, HIDDEN+ACTION_FEATURES_V5),
                                    (HIDDEN, HIDDEN+ACTION_FEATURES_V6)),
}


def _expand(name, value):
    old_shape, new_shape = SHAPES[name]
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != old_shape:
        raise ValueError("Mismatched v5 Battle tensor: " + name)
    result = value.new_zeros(new_shape)
    result[:, :old_shape[1]] = value
    return result


def _upgrade_weights(weights, new_template):
    if not isinstance(weights, dict) or set(weights) != set(new_template):
        raise ValueError("Missing or changed pretrained model parameter keys")
    migrated = {}
    for name, target in new_template.items():
        value = weights[name]
        if name in SHAPES:
            expanded = _expand(name, value)
            if tuple(expanded.shape) != tuple(target.shape):
                raise ValueError("Wrong new Battle shape: " + name)
            migrated[name] = expanded
        else:
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                raise ValueError("Unexpected old network shape: " + name)
            migrated[name] = value.clone()
    return migrated


def _upgrade_optimizer(optimizer, old_model, new_model):
    old_named = list(old_model.named_parameters())
    new_named = list(new_model.named_parameters())
    if [x[0] for x in old_named] != [x[0] for x in new_named]:
        raise ValueError("Optimizer/model parameter identity or order changed")
    result = copy.deepcopy(optimizer)
    if not isinstance(result, dict) or len(result.get("param_groups", [])) != 1:
        raise ValueError("Expected one Adam optimizer parameter group")
    ids = result["param_groups"][0].get("params")
    if not isinstance(ids, list) or len(ids) != len(old_named) or len(set(ids)) != len(ids):
        raise ValueError("Unexpected optimizer identity mapping")
    for pid, (name, param), (_, updated) in zip(ids, old_named, new_named):
        moments = result.get("state", {}).get(pid)
        if moments is None:
            raise ValueError("Uninitialized Adam state for " + name)
        for key, value in moments.items():
            if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in SHAPES:
                    moments[key] = _expand(name, value)
                elif not isinstance(value, torch.Tensor) or value.shape != param.shape:
                    raise ValueError("Changed optimizer moment shape: " + name)
            elif key == "step":
                if not isinstance(value, (int, float, torch.Tensor)):
                    raise ValueError("Unexpected Adam step type")
            else:
                raise ValueError("Unknown optimizer slot: " + key)
        if any(tuple(moments[k].shape) != tuple(updated.shape)
               for k in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Optimizer migration changed parameter dimensions")
    if set(result.get("state", {})) != set(ids):
        raise ValueError("Unrecognized Adam state ID")
    return result


def upgrade_checkpoint(checkpoint, old_model, new_model):
    """Return a detached v6 checkpoint; caller must validate provenance/seal.

    The source/checkpoint SHA and manifest are intentionally NOT rewritten here:
    this helper is not a bypass around the approved full-state transfer gates.
    """
    if old_model.battle.state_features != STATE_FEATURES_V5 or (
        old_model.battle.action_features != ACTION_FEATURES_V5 or
        new_model.battle.state_features != STATE_FEATURES_V6 or
        new_model.battle.action_features != ACTION_FEATURES_V6):
        raise ValueError("v5->v6 Battle migration only")
    if not isinstance(checkpoint, dict) or checkpoint.get("is_champion") is not False:
        raise ValueError("Only frozen non-Champion research checkpoint supported")
    required = ("model", "target", "magnet", "optimizer", "scheduler",
                "torch_rng", "python_rng")
    if any(key not in checkpoint for key in required):
        raise ValueError("Incomplete full-state checkpoint")
    result = copy.deepcopy(checkpoint)
    template = new_model.state_dict()
    for name in ("model", "target", "magnet"):
        result[name] = _upgrade_weights(result[name], template)
    result["optimizer"] = _upgrade_optimizer(result["optimizer"],old_model,new_model)
    result["battle_feature_schema"] = "rnad-three-phase-features-v6"
    return result
