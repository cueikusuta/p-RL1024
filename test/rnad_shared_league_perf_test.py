"""CPU fast-path regression: no authoritative Match or strength claim."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.rnad.integrated_policy_v1 import IntegratedPolicyV1
from scripts.rnad.selection_ab import TRIPLES
from scripts.rnad.shared_league_learner import _load, _selection_batch


class SharedLeagueThroughputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _model(self):
        vocab = {"sizes": {key: 7 for key in
                 ("species", "names", "abilities", "items", "moves", "types")}}
        torch.manual_seed(20261119)
        return IntegratedPolicyV1(vocab)

    def _observations(self):
        torch.manual_seed(404)
        return {
            "own_cat": torch.randint(0, 7, (4, 6, 9), dtype=torch.long),
            "foe_cat": torch.randint(0, 7, (4, 6, 3), dtype=torch.long),
            "own_num": torch.randn(4, 6, 16),
            "foe_num": torch.randn(4, 6, 6),
        }

    def test_selection_batch_matches_unbatched_predictions_and_gradients(self):
        model = self._model()
        obs = self._observations()
        batched_logits, batched_values = model.selection_forward_batch(obs)
        one_logits, one_values = [], []
        for i in range(4):
            seat_obs = {k: v[i:i+1] for k, v in obs.items()}
            logits, value, _ = model("selection", seat_obs, TRIPLES)
            one_logits.append(logits)
            one_values.append(value)
        serial_logits, serial_values = torch.stack(one_logits), torch.stack(one_values)
        torch.testing.assert_close(batched_logits, serial_logits,
                                   rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(batched_values, serial_values,
                                   rtol=1e-5, atol=1e-5)
        model.zero_grad(set_to_none=True)
        (batched_logits.square().mean()+batched_values.square().mean()).backward()
        batch_grads = [p.grad.detach().clone() if p.grad is not None else None
                       for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        (serial_logits.square().mean()+serial_values.square().mean()).backward()
        serial_grads = [p.grad for p in model.parameters()]
        for left, right in zip(batch_grads, serial_grads):
            if left is None or right is None:
                self.assertIsNone(left)
                self.assertIsNone(right)
            else:
                torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)

    def test_selection_embedding_reused_once_per_batch(self):
        model = self._model()
        obs = self._observations()
        with patch.object(model.selection.embed, "forward",
                          wraps=model.selection.embed.forward) as mocked:
            logits, values = model.selection_forward_batch(obs)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(tuple(logits.shape), (4, len(TRIPLES)))
        self.assertEqual(tuple(values.shape), (4,))

    def test_selection_batch_only_trains_own_legitimate_seats(self):
        import copy
        model = self._model()
        target = copy.deepcopy(model).eval()
        magnet = copy.deepcopy(model).eval()
        obs = self._observations()
        events = []
        for i in range(4):
            p = {k: v[i].tolist() for k, v in obs.items()}
            events.append({"actor": i % 2, "phase": "selection",
                           "phase_input": p})
        batched = _selection_batch(model, target, magnet, events, (0,))
        self.assertEqual(set(batched), {0, 2})
        for i, (logits, value, t, tv, mag) in batched.items():
            single = {k: v[i:i+1] for k, v in obs.items()}
            expected_logits, expected_value, _ = model(
                "selection", single, TRIPLES)
            torch.testing.assert_close(logits, expected_logits,
                                       rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(value, expected_value,
                                       rtol=1e-5, atol=1e-5)
            self.assertFalse(t.requires_grad)
            self.assertFalse(mag.requires_grad)
            self.assertFalse(tv.requires_grad)

    def test_strict_verified_plain_json_and_original_actor_gzip_load_same(self):
        row = {"match_id": "unit-only-not-authority", "events": [
            {"phase": "selection", "action_index": 3}], "winner": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, zipped = root / "match.json", root / "match.json.gz"
            raw.write_text(json.dumps(row), encoding="utf-8")
            with gzip.open(zipped, "wt") as stream:
                json.dump(row, stream)
            self.assertEqual(_load(raw), row)
            self.assertEqual(_load(zipped), row)


if __name__ == "__main__":
    unittest.main()
