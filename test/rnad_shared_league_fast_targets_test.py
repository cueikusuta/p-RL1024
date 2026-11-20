"""Numerical and fail-closed reference parity for CPU NumPy R-NaD target candidate."""
import math
import unittest

import torch

from scripts.rnad.two_player import rnad_targets
from scripts.rnad.shared_league_fast_targets import rnad_targets_numpy


class RnadFastTargetsParity(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def _accepted_like_steps(self, n):
        torch.manual_seed(20261120)
        steps = []
        for t in range(n):
            width = (3, 60, 500, 6)[t % 4]
            logits = torch.randn(width) / 3
            magnet_logits = torch.randn(width) / 3
            pi = logits.softmax(0)
            mag = magnet_logits.softmax(0)
            action = (t*17 + 3) % width
            steps.append({
                "actor": t % 2, "reward": [1.,-1.] if t == n-1 else [0.,0.],
                "pi": pi, "magnet": mag,
                "behavior_prob": float(pi[action]),
                "value": math.tanh(t/21-1),
                "action_index": action,
            })
        return steps

    def test_detached_two_player_targets_and_all_action_q_match(self):
        for n in (1, 2, 11, 64, 128):
            with self.subTest(events=n):
                steps = self._accepted_like_steps(n)
                ref = rnad_targets(steps, eta=.2, rho_bar=1., c_bar=1.)
                fast = rnad_targets_numpy(steps, eta=.2, rho_bar=1., c_bar=1.)
                self.assertEqual(len(ref), len(fast))
                for a, b in zip(ref, fast):
                    for field in ("actor", "rho", "c", "ratio", "value_target"):
                        self.assertAlmostEqual(a[field], b[field], delta=2e-8)
                    for x, y in zip(a["regularized_reward"],
                                    b["regularized_reward"]):
                        self.assertAlmostEqual(x, y, delta=2e-8)
                    self.assertEqual(a["q_targets"].dtype, torch.float32)
                    self.assertFalse(b["q_targets"].requires_grad)
                    torch.testing.assert_close(a["q_targets"], b["q_targets"],
                                               rtol=1e-5, atol=1e-6)

    def test_illegal_inputs_remain_fail_closed(self):
        valid = self._accepted_like_steps(2)
        cases = [
            ([], "Empty"),
            ([dict(valid[0], actor=3)], "seat"),
            ([dict(valid[0], behavior_prob=0)], "probabilities"),
            ([dict(valid[0], action_index=999)], "probabilities"),
            ([dict(valid[0], value=float("nan"))], "value"),
            ([dict(valid[0], pi=torch.tensor([0.0, 1.0]))], "probabilities"),
            ([dict(valid[0], magnet=torch.full((3,), float("nan")))], "probabilities"),
            ([dict(valid[0], reward=[1.,1.])], "zero-sum"),
        ]
        for example, message in cases:
            with self.subTest(reason=message):
                with self.assertRaises((ValueError, AssertionError)):
                    rnad_targets_numpy(example)
        with self.assertRaisesRegex(ValueError, "parameters"):
            rnad_targets_numpy(valid, rho_bar=0)


if __name__ == "__main__":
    unittest.main()
