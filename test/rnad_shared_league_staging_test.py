"""Staging equivalence and rejection tests; synthetic data is NEVER training evidence."""
import gzip
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from rnad_shared_league_contract_test import manifest, payload
from scripts.rnad.shared_league_numeric_parity import assert_exact_json
from scripts.rnad.shared_league_staging import (
    stage_verified_actors, staging_config,
)


class SharedLearnerStagingTest(unittest.TestCase):
    def _actors(self, root):
        m = manifest()
        paths = []
        for i in range(20):
            directory = root / f"actor-{i}"
            directory.mkdir()
            file = directory / f"actor-{i}.json.gz"
            with gzip.open(file, "wt") as sink:
                json.dump(payload(m, i), sink, separators=(",", ":"),
                          allow_nan=False)
            paths.append(file)
        return m, paths

    def test_all_1280_outputs_identical_under_parallel_fast_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m, paths = self._actors(root)
            a = root / "reference"
            b = root / "fast-two"
            c = root / "fast-four"
            first = stage_verified_actors(
                m, paths, a, workers=1, fast_json=False,
                return_metrics=True)
            for workers, dest in ((2, b), (4, c)):
                candidate = stage_verified_actors(
                    m, paths, dest, workers=workers, fast_json=True,
                    return_metrics=True)
                self.assertEqual(first[0], candidate[0])
                self.assertEqual(first[1], candidate[1])
                self.assertEqual(first[2], candidate[2])
                self.assertEqual(candidate[3]["accepted_matches"], 1280)
                self.assertEqual(candidate[3]["workers"], workers)
                for row in first[1]:
                    name = row["match_id"] + ".json"
                    self.assertEqual(
                        hashlib.sha256((a / name).read_bytes()).digest(),
                        hashlib.sha256((dest / name).read_bytes()).digest(),
                        name)

    def test_corrupt_and_missing_shards_fail_before_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m, paths = self._actors(root)
            with self.assertRaisesRegex(ValueError, "20/20"):
                stage_verified_actors(m, paths[:-1], root / "missing")
            with gzip.open(paths[0], "wt") as target:
                json.dump(payload(m, 1), target, separators=(",", ":"))
            for workers in (1, 2):
                with self.assertRaisesRegex(ValueError, "Actor artifact ID"):
                    stage_verified_actors(
                        m, paths, root / f"bad-{workers}",
                        workers=workers, fast_json=True)

    def test_orjson_strict_round_trip_preserves_all_1280_matches(self):
        import orjson
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m, paths = self._actors(root)
            # Use a signed-zero float and nontrivial precision to catch
            # serializer numeric drift, not merely approximate loss parity.
            with gzip.open(paths[0], "rt") as f:
                first_actor = json.load(f)
            first_actor["matches"][0]["events"][0]["phase_input"]["pool"][0][0] = -0.0
            first_actor["matches"][0]["events"][0]["phase_input"]["pool"][0][1] = 0.12345678901234566
            with gzip.open(paths[0], "wt") as f:
                json.dump(first_actor, f, separators=(",", ":"), allow_nan=False)
            std_dir, ort_dir = root / "json", root / "orjson"
            legacy = stage_verified_actors(
                m, paths, std_dir, workers=1, fast_json=True,
                codec="stdlib", return_metrics=True)
            candidate = stage_verified_actors(
                m, paths, ort_dir, workers=4, fast_json=True,
                codec="orjson", return_metrics=True,
                monitor_memory=True)
            self.assertEqual(legacy[:3], candidate[:3])
            self.assertEqual(candidate[3]["codec"], "orjson")
            self.assertGreater(candidate[3]["concurrent_memory"]["sample_count"], 0)
            self.assertGreater(candidate[3]["concurrent_memory"]["peak_parent_and_children_rss_kib"], 0)
            self.assertGreater(candidate[3]["concurrent_memory"]["peak_children_concurrent_rss_kib"], 0)
            self.assertEqual(len(legacy[1]), 1280)
            for choice in legacy[1]:
                name = choice["match_id"] + ".json"
                original = json.loads((std_dir / name).read_bytes())
                converted = json.loads((ort_dir / name).read_bytes())
                assert_exact_json(original, converted, name)
            mid = legacy[1][0]["match_id"] + ".json"
            example = json.loads((ort_dir / mid).read_bytes())
            self.assertEqual(struct.pack("!d", example["events"][0]["phase_input"]["pool"][0][0]),
                             struct.pack("!d", -0.0))
            # Fast codec MUST reject non-RFC nonfinite rather than silently
            # turning NaN/Infinity into null when encoding.
            self.assertRaises(orjson.JSONDecodeError, orjson.loads, b'{"x":NaN}')

    def test_ieee754_numeric_comparison_rejects_signed_zero_and_ulp_drift(self):
        self.assertEqual(struct.pack("!d", -0.0), struct.pack("!d", -0.0))
        assert_exact_json([-0.0, 0.12345678901234566, 4, False],
                          [-0.0, 0.12345678901234566, 4, False])
        with self.assertRaisesRegex(ValueError, "IEEE754"):
            assert_exact_json([-0.0], [0.0])
        with self.assertRaisesRegex(ValueError, "IEEE754"):
            assert_exact_json([0.1], [math.nextafter(0.1, 1.0)])
        with self.assertRaisesRegex(ValueError, "JSON type"):
            assert_exact_json([1], [1.0])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            assert_exact_json([float("nan")], [float("nan")])

    def test_all_nonfinite_tokens_fail_closed_in_parallel_staging(self):
        import orjson
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m, paths = self._actors(root)
            for label, value in (("nan", float("nan")),
                                 ("positive_inf", float("inf")),
                                 ("negative_inf", float("-inf"))):
                actor = payload(m, 0)
                actor["matches"][0]["events"][0]["phase_input"]["pool"][0][0] = value
                with gzip.open(paths[0], "wt") as dest:
                    json.dump(actor, dest, separators=(",", ":"), allow_nan=True)
                with self.assertRaisesRegex(ValueError, "Invalid strict JSON Actor"):
                    stage_verified_actors(
                        m, paths, root / f"reject-{label}",
                        workers=4, fast_json=True, codec="orjson")
                # No acceptance report or first optimizer update is emitted.

    def test_invalid_staging_configuration_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "RNAD_PREP_WORKERS"):
            stage_verified_actors(manifest(), [], Path("."), workers=3)
        from unittest.mock import patch
        with patch.dict("os.environ", {"RNAD_PREP_WORKERS": "20"}):
            with self.assertRaisesRegex(ValueError, "Invalid bounded"):
                staging_config()
        with patch.dict("os.environ", {"RNAD_PREP_FAST_JSON": "bad"}):
            with self.assertRaisesRegex(ValueError, "Invalid bounded"):
                staging_config()
        with patch.dict("os.environ", {"RNAD_PREP_CODEC": "orjson",
                                       "RNAD_PREP_FAST_JSON": "0"}):
            with self.assertRaisesRegex(ValueError, "codec"):
                staging_config()
        with self.assertRaisesRegex(ValueError, "codec"):
            stage_verified_actors(manifest(), [], Path("."), codec="orjson",
                                  fast_json=False)


if __name__ == "__main__":
    unittest.main()
