"""RNAD-001 small CPU policy/value network; separate from I975/I976.
The network consumes only versioned, seat-scoped public/own features.
No opponent hidden features or privileged critic input are allowed.
"""
from __future__ import annotations
import copy
import json
import hashlib
import math
from pathlib import Path
import torch
from torch import nn

STATE_FEATURES = 32
ACTION_FEATURES = 16
HIDDEN = 128
MODEL_SCHEMA = "rnad-small-gru-v1"
BATTLE_FEATURE_SCHEMA = "rnad-battle-features-v1"
THREE_PHASE_FEATURE_SCHEMA = "rnad-three-phase-features-v1"
THREE_PHASE_FEATURE_SCHEMA_V2 = "rnad-three-phase-features-v2"
THREE_PHASE_FEATURE_SCHEMA_V3 = "rnad-three-phase-features-v3"
THREE_PHASE_FEATURE_SCHEMA_V4 = "rnad-three-phase-features-v4"
STATE_FEATURES_V3 = 48
ACTION_FEATURES_V3 = 32


class RNADNetwork(nn.Module):
    def __init__(self, hidden: int = HIDDEN, *,
                 state_features: int = STATE_FEATURES,
                 action_features: int = ACTION_FEATURES):
        super().__init__()
        if (state_features,action_features) not in (
            (STATE_FEATURES,ACTION_FEATURES),
            (STATE_FEATURES_V3,ACTION_FEATURES_V3)):
            raise ValueError("Unsupported versioned RNAD feature widths")
        self.hidden_size = hidden
        self.state_features = state_features
        self.action_features = action_features
        self.state_proj = nn.Linear(state_features, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.policy_hidden = nn.Linear(hidden + action_features, hidden)
        self.policy_out = nn.Linear(hidden, 1)
        self.value_hidden = nn.Linear(hidden, hidden)
        self.value_out = nn.Linear(hidden, 1)

    def forward(self, state, candidates, memory):
        if state.ndim != 2 or state.shape[-1] != self.state_features:
            raise ValueError("Invalid actor state width")
        if candidates.ndim != 3 or candidates.shape[0] != state.shape[0] or candidates.shape[-1] != self.action_features:
            raise ValueError("Invalid legal-action width")
        if candidates.shape[1] < 1:
            raise ValueError("No legal actions in decision")
        encoded = torch.relu(self.state_proj(state))
        next_memory = self.gru(encoded, memory)
        repeated = next_memory.unsqueeze(1).expand(-1, candidates.size(1), -1)
        logits = self.policy_out(torch.relu(
            self.policy_hidden(torch.cat([repeated, candidates], dim=-1))
        )).squeeze(-1)
        value = torch.tanh(self.value_out(torch.relu(
            self.value_hidden(next_memory)
        ))).squeeze(-1)
        return logits, value, next_memory

    def initial_memory(self, batch=1):
        return torch.zeros(batch, self.hidden_size)


def freeze_magnet(model):
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference


def export_json(model, path, *, feature_schema=BATTLE_FEATURE_SCHEMA):
    if feature_schema not in (BATTLE_FEATURE_SCHEMA, THREE_PHASE_FEATURE_SCHEMA,
                              THREE_PHASE_FEATURE_SCHEMA_V2,
                              THREE_PHASE_FEATURE_SCHEMA_V3,
                              THREE_PHASE_FEATURE_SCHEMA_V4):
        raise ValueError("Invalid RNAD export feature schema")
    if ((feature_schema in (THREE_PHASE_FEATURE_SCHEMA_V3,
                             THREE_PHASE_FEATURE_SCHEMA_V4)) !=
        (model.state_features == STATE_FEATURES_V3)):
        raise ValueError("Model/feature schema mismatch on export")
    # All model parameters have exact names and dimensions; no Python eval.
    state = {key: {"shape": list(tensor.shape),
                   "values": tensor.detach().cpu().flatten().tolist()}
             for key, tensor in model.state_dict().items()}
    payload = {"schema": MODEL_SCHEMA, "state_features": model.state_features,
               "action_features": model.action_features, "hidden": model.hidden_size,
               "weights": state}
    # Preserve byte-for-byte legacy battle-only exports. New three-phase
    # weights MUST identify the distinct observation/action encoding.
    if feature_schema != BATTLE_FEATURE_SCHEMA:
        payload["feature_schema"] = feature_schema
    Path(path).write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    return payload


def export_onnx(model, path, opset_version=17):
    model = model.eval()
    state = torch.zeros(1, model.state_features)
    actions = torch.zeros(1, 2, model.action_features)
    memory = model.initial_memory()
    torch.onnx.export(model, (state, actions, memory), path,
                      input_names=["state", "actions", "memory"],
                      output_names=["logits", "value", "next_memory"],
                      dynamic_axes={"state": {0: "batch"},
                                    "actions": {0: "batch", 1: "legal_actions"},
                                    "memory": {0: "batch"},
                                    "logits": {0: "batch", 1: "legal_actions"},
                                    "value": {0: "batch"},
                                    "next_memory": {0: "batch"}},
                      opset_version=opset_version, dynamo=False)
    return str(path)


def compare_onnx(model, onnx_path):
    import numpy as np
    import onnxruntime as ort
    torch.manual_seed(314)
    state = torch.randn(2, model.state_features)
    actions = torch.randn(2, 7, model.action_features)
    memory = torch.randn(2, model.hidden_size)
    with torch.no_grad():
        expected = model.eval()(state, actions, memory)
    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    actual = session.run(None, {"state": state.numpy(),
                                "actions": actions.numpy(),
                                "memory": memory.numpy()})
    maxima = [float(np.max(np.abs(x.detach().numpy() - y)))
              for x, y in zip(expected, actual)]
    if not all(x <= 1e-4 for x in maxima):
        raise AssertionError("PyTorch/ONNX parity exceeds 1e-4: " + str(maxima))
    return maxima


def load_exported_json(model, path, *, expected_feature_schema=BATTLE_FEATURE_SCHEMA):
    """Import a STRICT versioned finite model from same-job trusted JSON.

    Returns the SHA256 of JSON bytes to bind all later checkpoints to the
    optional initialization. No pickle execution or silent shape coercion.
    """
    raw=Path(path).read_bytes()
    payload=json.loads(raw)
    if (payload.get("schema")!=MODEL_SCHEMA or
        payload.get("hidden")!=model.hidden_size or
        payload.get("state_features")!=model.state_features or
        payload.get("action_features")!=model.action_features or
        not isinstance(payload.get("weights"),dict) or
        payload.get("feature_schema",BATTLE_FEATURE_SCHEMA) != expected_feature_schema):
        raise ValueError("Invalid warm-start model schema/dimensions")
    reference=model.state_dict()
    if set(payload["weights"])!=set(reference):
        raise ValueError("Missing or unexpected warm-start tensors")
    loaded={}
    for name,prior in reference.items():
        item=payload["weights"][name]
        if (not isinstance(item,dict) or
            item.get("shape")!=list(prior.shape) or
            not isinstance(item.get("values"),list) or
            len(item["values"])!=prior.numel() or
            any(type(x) not in (float,int) or not math.isfinite(x)
                for x in item["values"])):
            raise ValueError("Invalid warm-start tensor: "+name)
        loaded[name]=torch.tensor(item["values"],dtype=prior.dtype).reshape(prior.shape)
    model.load_state_dict(loaded,strict=True)
    return hashlib.sha256(raw).hexdigest()

def migrate_v2_export_to_v3(model, path):
    """Explicit v2->v3 warm start: old logits exactly preserved for zero
    extra features; additional 16 state/action channels start at weight zero.
    The returned origin SHA MUST stay in checkpoint provenance. v2 unchanged.
    """
    if (model.state_features,model.action_features)!=(
        STATE_FEATURES_V3,ACTION_FEATURES_V3):
        raise ValueError("Target is not a v3 architecture")
    source=RNADNetwork()
    origin=load_exported_json(source,path,
        expected_feature_schema=THREE_PHASE_FEATURE_SCHEMA_V2)
    old=source.state_dict()
    new=model.state_dict()
    for name,p in old.items():
        if name=='state_proj.weight':
            new[name].zero_()
            new[name][:,:STATE_FEATURES].copy_(p)
        elif name=='policy_hidden.weight':
            new[name].zero_()
            new[name][:,:HIDDEN].copy_(p[:,:HIDDEN])
            new[name][:,HIDDEN:HIDDEN+ACTION_FEATURES].copy_(p[:,HIDDEN:])
        else:
            new[name].copy_(p)
    model.load_state_dict(new,strict=True)
    return origin
