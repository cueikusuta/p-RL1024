"""Boundary and fail-closed checks for approved B+C 100h recovery range."""
import unittest
from scripts.rnad.shared_league_artifact_recovery import (
    artifact_name, accepted_artifact_names, missing_actor_ids,
)

class RecoveryBoundsTest(unittest.TestCase):
    def test_valid_generations(self):
        for generation in (240, 241, 598):
            with self.subTest(generation=generation):
                self.assertEqual(artifact_name(generation, 19),
                                 f"rnad-i5-cont-g{generation}-actor-19")
                rows = [{"name": artifact_name(generation, i), "expired": False}
                        for i in range(20)]
                self.assertEqual(missing_actor_ids(generation, rows), [])
                self.assertEqual(accepted_artifact_names(generation, rows),
                                 [artifact_name(generation, i) for i in range(20)])

    def test_out_of_range(self):
        for generation in (-1, 599, 600):
            with self.subTest(generation=generation):
                with self.assertRaises(ValueError):
                    artifact_name(generation, 0)

    def test_missing_actor_and_ambiguous_recovery(self):
        rows = [{"name": artifact_name(241, i)} for i in range(19)]
        self.assertEqual(missing_actor_ids(241, rows), [19])
        rows.append({"name": artifact_name(241, 0, recovered=True)})
        with self.assertRaises(ValueError):
            accepted_artifact_names(241, rows)

if __name__ == "__main__":
    unittest.main()
