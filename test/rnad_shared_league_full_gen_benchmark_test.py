"""Negative fixtures for whole-generation state verification (no gameplay)."""
from __future__ import annotations

import copy
import random
import unittest

import torch

from scripts.rnad.shared_league_full_gen_benchmark import _audit_full_state


def _fixture():
    torch.manual_seed(20261120)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    adam = optimizer.state_dict()
    for entry in adam["state"].values():
        entry["step"].fill_(40)
    return {
        "mean_loss_per_step": [0.2] * 40,
        "maximum_ratio_per_step": [1.0] * 40,
        "full_state": {
            "step": 40,
            "training_match_ids": [f"accepted-{i}" for i in range(1280)],
            "model": {"weight": torch.tensor([1., 2.])},
            "target": {"weight": torch.tensor([1., 3.])},
            "magnet": {"weight": torch.tensor([4., 5.])},
            "optimizer": adam,
            "scheduler": {"last_epoch":40,"base_lrs":[3e-5]},
            "torch_rng": torch.get_rng_state().clone(),
            "python_rng": random.getstate(),
        },
    }


class FullGenerationParityNegativeTest(unittest.TestCase):
    def test_valid_identical_complete_checkpoint(self):
        baseline = _fixture()
        candidate = copy.deepcopy(baseline)
        result = _audit_full_state(baseline, candidate)
        self.assertTrue(result["full_state_parity_passed"])
        self.assertEqual(result["max_40_step_mean_loss_abs_diff"], 0)
        self.assertGreaterEqual(result["adam_moment_tensors_compared"], 2)

    def test_missing_or_reordered_accepted_matches_fails(self):
        baseline = _fixture()
        altered = copy.deepcopy(baseline)
        altered["full_state"]["training_match_ids"][:2] = (
            altered["full_state"]["training_match_ids"][1::-1])
        with self.assertRaisesRegex(AssertionError, "Match ordering"):
            _audit_full_state(baseline, altered)

    def test_checkpoint_model_target_magnet_and_rng_drift_fails(self):
        original = _fixture()
        for component in ("model", "target", "magnet"):
            altered = copy.deepcopy(original)
            altered["full_state"][component]["weight"][0] += 0.5
            with self.subTest(component=component), self.assertRaises(AssertionError):
                _audit_full_state(original, altered)
        changed = copy.deepcopy(original)
        changed["full_state"]["torch_rng"][0] ^= 1
        with self.assertRaisesRegex(AssertionError, "RNG"):
            _audit_full_state(original, changed)
        changed = copy.deepcopy(original)
        changed["full_state"]["scheduler"]["last_epoch"] += 1
        with self.assertRaisesRegex(AssertionError, "Scheduler"):
            _audit_full_state(original, changed)

    def test_adam_step_moment_and_group_drift_fails(self):
        baseline = _fixture()
        state_id = next(iter(baseline["full_state"]["optimizer"]["state"]))
        for field in ("step", "exp_avg", "exp_avg_sq"):
            altered = copy.deepcopy(baseline)
            moment = altered["full_state"]["optimizer"]["state"][state_id][field]
            if field == "step":
                moment.add_(1)
            elif field == "exp_avg":
                moment.add_(0.1)
            else:
                moment.add_(0.01)
            with self.subTest(field=field), self.assertRaises(AssertionError):
                _audit_full_state(baseline, altered)
        changed = copy.deepcopy(baseline)
        changed["full_state"]["optimizer"]["param_groups"][0]["lr"] *= 2
        with self.assertRaisesRegex(AssertionError, "Adam optimizer groups"):
            _audit_full_state(baseline, changed)

    def test_loss_or_ratio_drift_fails(self):
        baseline = _fixture()
        changed = copy.deepcopy(baseline)
        changed["mean_loss_per_step"][12] += 0.1
        with self.assertRaisesRegex(AssertionError, "Loss or V-trace"):
            _audit_full_state(baseline, changed)
        changed = copy.deepcopy(baseline)
        changed["maximum_ratio_per_step"][31] += 0.001
        with self.assertRaisesRegex(AssertionError, "Loss or V-trace"):
            _audit_full_state(baseline, changed)


if __name__ == "__main__":
    unittest.main()
