"""Opt-in NeuRD fused-surrogate forward, gradient and clipping parity."""
import unittest
import torch

from scripts.rnad.two_player import neurd_policy_loss
from scripts.rnad.shared_league_linearized_neurd import (
    linearized_neurd_loss_batch,
)


class LinearizedNeuRDParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_ragged_candidate_gradients_and_values(self):
        for beta in (0., 2.):
            with self.subTest(beta=beta):
                torch.manual_seed(20261120)
                samples = []
                for i, n in enumerate((2, 60, 1, 30, 512, 8)):
                    logits = torch.randn(n, dtype=torch.float32)
                    if i == 1:
                        logits[0] = 5.0  # exterior NeuRD clipping gate
                    q = torch.randn(n, dtype=torch.float32)
                    samples.append((logits, q, (0., .5, 1.)[i % 3]))
                reference = [l.detach().clone().requires_grad_(True)
                             for l, _, _ in samples]
                candidate = [l.detach().clone().requires_grad_(True)
                             for l, _, _ in samples]
                expected = sum(
                    neurd_policy_loss(l, q, beta=beta, is_c=c)
                    for l, (_, q, c) in zip(reference, samples)
                ) / len(samples)
                actual = linearized_neurd_loss_batch([
                    (l, q, c) for l, (_, q, c) in zip(candidate, samples)],
                    beta=beta)
                torch.testing.assert_close(actual, expected,
                                           rtol=1e-5, atol=1e-6)
                expected.backward()
                actual.backward()
                for a, b in zip(reference, candidate):
                    torch.testing.assert_close(a.grad, b.grad,
                                               rtol=1e-4, atol=1e-6)

    def test_invalid_actions_and_parameters_fail_closed(self):
        good = (torch.zeros(3), torch.ones(3)/3, 1.0)
        with self.assertRaisesRegex(ValueError, "No legal"):
            linearized_neurd_loss_batch([])
        with self.assertRaisesRegex(ValueError, "parameters"):
            linearized_neurd_loss_batch([good], beta=-1)
        for invalid in [
            (torch.zeros(0), torch.zeros(0), 1.0),
            (torch.zeros(3), torch.ones(4), 1.0),
            (torch.zeros(3), torch.ones(3), 1.1),
            (torch.full((3,), float("nan")), torch.ones(3), 1.0),
            (torch.zeros(3), torch.full((3,), float("inf")), 1.0),
        ]:
            with self.subTest(invalid=str(invalid[2])):
                with self.assertRaises(ValueError):
                    linearized_neurd_loss_batch([invalid])


if __name__ == "__main__":
    unittest.main()
