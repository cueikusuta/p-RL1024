"""Negative tests for 20-way Actor Artifact recovery; no training artifacts."""
import unittest

from scripts.rnad.shared_league_artifact_recovery import (
    ACTORS, artifact_name, missing_actor_ids, accepted_artifact_names
)


def rows(names):
    return [{"name": name, "expired": False} for name in names]


class SharedArtifactRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.generation = 3
        self.original = [artifact_name(self.generation, i)
                         for i in range(ACTORS)]

    def test_generation_zero_supported_and_fail_closed(self):
        # G0 uses bootstrap/actor as both Main and Magnet, but its
        # artifact namespace and strict recovery still require 20 IDs.
        names = [artifact_name(0, i) for i in range(ACTORS)]
        self.assertEqual(missing_actor_ids(0, rows(names)), [])
        self.assertEqual(accepted_artifact_names(0, rows(names)), names)
        partial = [name for i, name in enumerate(names) if i != 19]
        self.assertEqual(missing_actor_ids(0, rows(partial)), [19])
        rescued = artifact_name(0, 19, recovered=True)
        self.assertEqual(
            accepted_artifact_names(0, rows(partial + [rescued]))[19],
            rescued)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            missing_actor_ids(0, rows(names + [rescued]))
        for bad in (-1, 62):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                artifact_name(bad, 0)

    def test_g49_to_g61_strict_recovery_and_resolution(self):
        # G48 handoff starts G49; research workflow may reach G61.
        for generation in (48, 49, 50, 60, 61):
            with self.subTest(generation=generation):
                full = [artifact_name(generation, i) for i in range(ACTORS)]
                self.assertEqual(missing_actor_ids(generation, rows(full)), [])
                self.assertEqual(accepted_artifact_names(generation, rows(full)), full)
                missing = full[:19]
                self.assertEqual(missing_actor_ids(generation, rows(missing)), [19])
                rescued = artifact_name(generation, 19, recovered=True)
                self.assertEqual(
                    accepted_artifact_names(generation, rows(missing+[rescued]))[-1],
                    rescued)
                with self.assertRaisesRegex(ValueError, "Ambiguous"):
                    missing_actor_ids(generation, rows(full+[rescued]))

    def test_all_twenty_success_no_replay(self):
        self.assertEqual(
            missing_actor_ids(self.generation, rows(self.original)), [])
        self.assertEqual(
            accepted_artifact_names(self.generation, rows(self.original)),
            self.original)

    def test_one_missing_replays_only_same_logical_actor(self):
        entries = [item for i, item in enumerate(self.original) if i != 16]
        self.assertEqual(missing_actor_ids(self.generation, rows(entries)),
                         [16])
        with self.assertRaisesRegex(ValueError, "Logical actor 16"):
            accepted_artifact_names(self.generation, rows(entries))
        recovered = artifact_name(self.generation, 16, recovered=True)
        resolved = accepted_artifact_names(
            self.generation, rows(entries + [recovered]))
        self.assertEqual(resolved[16], recovered)
        self.assertEqual([resolved[i] for i in range(ACTORS) if i != 16],
                         [self.original[i] for i in range(ACTORS) if i != 16])

    def test_many_missing_and_exact_rescue(self):
        entries = self.original[:4]
        expected = list(range(4, 20))
        self.assertEqual(missing_actor_ids(self.generation, rows(entries)),
                         expected)
        rescued = [artifact_name(self.generation, i, recovered=True)
                   for i in expected]
        self.assertEqual(
            accepted_artifact_names(self.generation, rows(entries + rescued)),
            entries + rescued)

    def test_duplicate_original_and_recovered_refused(self):
        with self.assertRaisesRegex(ValueError, "Logical actor 16"):
            accepted_artifact_names(
                self.generation, rows(
                    self.original + [artifact_name(
                        self.generation, 16, recovered=True)]))

    def test_inspection_is_idempotent_after_one_successful_rescue(self):
        old = [item for i,item in enumerate(self.original) if i != 16]
        repaired = artifact_name(self.generation, 16, recovered=True)
        self.assertEqual(
            missing_actor_ids(self.generation, rows(old + [repaired])), [])
        self.assertEqual(
            accepted_artifact_names(self.generation, rows(old + [repaired]))[16],
            repaired)

    def test_conflicting_original_and_rescue_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            missing_actor_ids(
                self.generation, rows(self.original + [
                    artifact_name(self.generation, 16, recovered=True)]))

    def test_expired_or_duplicated_artifact_refused(self):
        entries = rows(self.original)
        entries[4]["expired"] = True
        with self.assertRaisesRegex(ValueError, "Expired"):
            accepted_artifact_names(self.generation, entries)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            accepted_artifact_names(
                self.generation, rows(self.original + [self.original[4]]))

    def test_old_generation_does_not_satisfy_missing_actor(self):
        entries = [entry for i, entry in enumerate(self.original) if i != 1]
        entries += [artifact_name(2, 1)]
        self.assertEqual(missing_actor_ids(self.generation, rows(entries)),
                         [1])
        with self.assertRaisesRegex(ValueError, "Logical actor 1"):
            accepted_artifact_names(self.generation, rows(entries))

    def test_invalid_actors_and_unexpected_same_gen_results(self):
        with self.assertRaisesRegex(ValueError, "Invalid"):
            artifact_name(self.generation, 20)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            accepted_artifact_names(
                self.generation, rows(self.original + [
                    "rnad-i5-cont-g3-actor-20"]))
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            accepted_artifact_names(
                self.generation, rows(self.original + [
                    "rnad-i5-cont-g3-recovered-actor-99"]))

    def test_invalid_manifest_structure_fails_closed(self):
        for bad in (None, {}, ["bad"], [{"name": 7}],
                    [{"name": self.original[0], "expired": "false"}]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    missing_actor_ids(self.generation, bad)


if __name__ == "__main__":
    unittest.main()
