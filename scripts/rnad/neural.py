"""RNAD-001 small CPU policy/value network; separate from I975/I976.
The network consumes only versioned, seat-scoped public/own features.
No opponent hidden features or privileged critic input are allowed.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from torch import nn

STATE_FEATURES = 32
ACTION_FEATURES = 16
HIDDEN = 128
MODEL_SCHEMA = "rnad-small-gru-v1"


class RNADNetwork(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.hidden_size = hidden
        self.state_proj = nn.Linear(STATE_FEATURES, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.policy_hidden = nn.Linear(hidden + ACTION_FEATURES, hidden)
        self.policy_out = nn.Linear(hidden, 1)
        self.value_hidden = nn.Linear(hidden, hidden)
        self.value_out = nn.Linear(hidden, 1)

    def forward(self, state, candidates, memory):
        if state.ndim != 2 or state.shape[-1] != STATE_FEATURES:
            raise ValueError("Invalid actor state width")
        if candidates.ndim != 3 or candidates.shape[0] != state.shape[0] or candidates.shape[-1] != ACTION_FEATURES:
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


def export_json(model, path):
    # All model parameters have exact names and dimensions; no Python eval.
    state = {key: {"shape": list(tensor.shape),
                   "values": tensor.detach().cpu().flatten().tolist()}
             for key, tensor in model.state_dict().items()}
    payload = {"schema": MODEL_SCHEMA, "state_features": STATE_FEATURES,
               "action_features": ACTION_FEATURES, "hidden": model.hidden_size,
               "weights": state}
    Path(path).write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    return payload


def export_onnx(model, path, opset_version=17):
    model = model.eval()
    state = torch.zeros(1, STATE_FEATURES)
    actions = torch.zeros(1, 2, ACTION_FEATURES)
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
    state = torch.randn(2, STATE_FEATURES)
    actions = torch.randn(2, 7, ACTION_FEATURES)
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
