"""Research-only NumPy CPU kernel for detached two-player R-NaD targets.

Keeps the original reverse chronological two-player V-trace recurrences and
all-action NeuRD Q targets. This is an OPT-IN alternative to torch's reference
kernel: no gradient is taken through target construction. Every Match is still
strictly receipt-verified before the learner runs, and this function performs
its own finite/normalization/action checks as well.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def rnad_targets_numpy(steps, eta=0.2, rho_bar=1.0, c_bar=1.0):
    if not steps:
        raise ValueError("Empty complete Match trajectory")
    if eta < 0 or rho_bar <= 0 or c_bar <= 0:
        raise ValueError("Invalid RNAD correction parameters")
    vhat = [0.0, 0.0]
    vnext = [0.0, 0.0]
    rhat = [0.0, 0.0]
    xi = [1.0, 1.0]
    output = [None] * len(steps)
    for t in range(len(steps)-1, -1, -1):
        row = steps[t]
        seat = row["actor"]
        if seat not in (0, 1):
            raise ValueError("Invalid acting seat")
        pi_tensor, mag_tensor = row["pi"].detach(), row["magnet"].detach()
        if pi_tensor.device.type != "cpu" or mag_tensor.device.type != "cpu":
            raise ValueError("Opt-in NumPy RNAD target path requires hosted CPU")
        # Conversion is bit-exact for the original float32/float64 inputs.
        pi = pi_tensor.flatten().numpy().astype(np.float64, copy=False)
        magnet = mag_tensor.flatten().numpy().astype(np.float64, copy=False)
        action = row["action_index"]
        mu = float(row["behavior_prob"])
        if (pi.size < 1 or magnet.shape != pi.shape or
            type(action) is not int or not 0 <= action < pi.size or
            mu <= 0 or mu > 1 or
            not np.isfinite(pi).all() or not np.isfinite(magnet).all() or
            np.any(pi <= 0) or np.any(magnet <= 0) or
            abs(float(pi.sum())-1) > 1e-5 or
            abs(float(magnet.sum())-1) > 1e-5):
            raise ValueError("Illegal/missing policy or behavior probabilities")
        original_reward = row["reward"]
        if len(original_reward) != 2 or abs(sum(original_reward)) > 1e-6:
            raise ValueError("Expected two-player zero-sum rewards")
        ratio = float(pi[action]) / mu
        logratio = np.log(pi / magnet)
        taken = float(logratio[action])
        reg = [float(original_reward[0]), float(original_reward[1])]
        reg[seat] -= eta*taken
        reg[1-seat] += eta*taken
        if abs(sum(reg)) > 1e-8:
            raise AssertionError("Regularization broke zero-sum signs")
        other = 1-seat
        rhat[other] = reg[other] + ratio*rhat[other]
        xi[other] *= ratio
        v = float(row["value"])
        if not math.isfinite(v) or abs(v) >= 1e6:
            raise ValueError("Invalid target-network value")
        prev_vhat = vhat[seat]
        corrected_ratio = ratio * xi[seat]
        rho = min(rho_bar, corrected_ratio)
        c = min(c_bar, corrected_ratio)
        delta = rho*(reg[seat] + ratio*rhat[seat] + vnext[seat] - v)
        value_target = v + delta + c*(prev_vhat-vnext[seat])
        q = -eta*logratio + v
        correction = (reg[seat] + eta*taken +
                      ratio*(rhat[seat]+prev_vhat) - v) / mu
        q[action] += correction
        if not np.isfinite(q).all() or not math.isfinite(value_target):
            raise ValueError("Nonfinite two-player R-NaD target")
        output[t] = {
            "value_target": float(value_target),
            "q_targets": torch.from_numpy(q.astype(np.float32)),
            "rho": rho, "c": c, "ratio": ratio,
            "actor": seat, "regularized_reward": tuple(reg),
        }
        vhat[seat] = value_target
        vnext[seat] = v
        rhat[seat] = 0.0
        xi[seat] = 1.0
    return output
