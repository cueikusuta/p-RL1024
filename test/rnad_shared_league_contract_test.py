"""Contract tests only: synthetic rows are NEVER authority training traces."""
import copy
import math
import unittest

from scripts.rnad.shared_league_contract import (
    ACTORS, MATCHES_PER_GENERATION, OPTIMIZER_STEPS, SEEDS_PER_ACTOR,
    ACTOR_SCHEMA, MATCH_SCHEMA, POLICY_SCHEMA, SCHEMA, _opponent,
    _matrix, _valid_draft_codes, _SELECTION_TRIPLES,
    actor_role, canonical_hash, verify_actor, verify_generation,
    verify_manifest,
)


def manifest(*, exploiter=False):
    return {
        "schema": SCHEMA,
        "generation": 1,
        "policy_id": "RNAD-I5-IntegratedPolicyV1",
        "policy_manifest_sha256": "a" * 64,
        "main_sha256": "b" * 64,
        "magnet_sha256": "c" * 64,
        "source_sha": "d" * 40,
        "engine_sha": "e" * 40,
        "feature_schema": POLICY_SCHEMA,
        "seed_start": 7100001,
        "actor_count": 20,
        "seeds_per_actor": 32,
        "optimizer_batch_matches": 32,
        "optimizer_steps": 40,
        "opponents": {
            "historical": [{"id": "past-v0", "sha256": "f" * 64}],
            "fixed": [{"id": "base128", "sha256": "1" * 64},
                      {"id": "base256", "sha256": "2" * 64}],
            "exploiter": [{"id": "approved-exploiter", "sha256": "3" * 64}]
                         if exploiter else [],
        },
    }


def event(seat, phase, terminal=False):
    if phase == "draft":
        data = {
            "pool": [[.1] * 44 for _ in range(6)],
            "codes": [[0,1,2,3,4,5]], "extra": [[.2,.3,0,0]],
        }
        n = 1
    elif phase == "selection":
        from scripts.rnad.train_selection_specialist import TRIPLES
        data = {
            "own_cat": [[0]*9 for _ in range(6)],
            "foe_cat": [[0]*3 for _ in range(6)],
            "own_num": [[.1]*16 for _ in range(6)],
            "foe_num": [[.1]*6 for _ in range(6)],
            "legal": [list(t) for t in TRIPLES],
        }
        n = 60
    else:
        data = {"state": [.1]*96,
                "actions": [[.1]*48,[.2]*48]}
        n = 2
    return {
        "actor": seat, "phase": phase, "phase_input": data,
        "action_index": 0,
        "behavior_prob": 1/n,
        "magnet_log_prob": -__import__("math").log(n),
        "policy_pi": [1/n]*n, "magnet_pi": [1/n]*n,
        "predicted_value": 0,
        "reward": [1,-1] if terminal else [0,0],
    }


def payload(m, actor_id):
    role, opponent = _opponent(m, actor_id)
    rows = []
    for seed in range(m["seed_start"] + actor_id * SEEDS_PER_ACTOR,
                      m["seed_start"] + (actor_id + 1) * SEEDS_PER_ACTOR):
        for seat in (0, 1):
            events = [
                event(0, "draft"), event(1, "draft"),
                event(0, "selection"), event(1, "selection"),
                event(0, "battle", terminal=True),
            ]
            rows.append({
                "schema": MATCH_SCHEMA,
                "match_id": f"g{m['generation']}-a{actor_id}-s{seed}-seat{seat}",
                "policy_id": m["policy_id"],
                "main_sha256": m["main_sha256"],
                "magnet_sha256": m["magnet_sha256"],
                "opponent_sha256": opponent["sha256"],
                "source_sha": m["source_sha"],
                "engine_sha": m["engine_sha"],
                "seed": seed, "learner_seat": seat,
                "completed": True, "score": [4, 2], "winner": 0,
                "rounds": 1, "action_counts": [1, 0],
                "events": events,
            })
    return {
        "schema": ACTOR_SCHEMA,
        "generation": m["generation"], "policy_id": m["policy_id"],
        "manifest_sha256": canonical_hash(m), "actor_id": actor_id,
        "main_sha256": m["main_sha256"],
        "magnet_sha256": m["magnet_sha256"],
        "source_sha": m["source_sha"], "engine_sha": m["engine_sha"],
        "opponent_id": opponent["id"],
        "opponent_sha256": opponent["sha256"], "role": role,
        "matches": rows,
    }


class SharedLeagueContractTest(unittest.TestCase):
    def setUp(self):
        self.manifest = manifest()
        self.payloads = [payload(self.manifest, i) for i in range(ACTORS)]

    def test_draft_candidate_single_pass_rejects_all_invalid_shapes(self):
        self.assertEqual(len(_SELECTION_TRIPLES), 60)
        self.assertEqual(_SELECTION_TRIPLES[0], [0, 1, 2])
        self.assertTrue(_valid_draft_codes([[0, 1, 2, 3, 4, 5]], 6))
        self.assertTrue(_valid_draft_codes([[5, 4, 3, 2, 1, 0]], 6))
        for values in ([], None, (), "not-a-list", [[0, 1, 2, 3, 4]],
                       [[0, 1, 2, 3, 4, 4]],
                       [[0, 1, 2, 3, 4, 6]],
                       [[0, 1, 2, 3, 4, -1]],
                       [[0, 1, 2, 3, 4, 5.0]],
                       [[0, 1, 2, 3, 4, True]],
                       [[0, 1, 2, 3, 4, float("nan")]],
                       [[0, 1, 2, 3, 4, []]]):
            self.assertFalse(_valid_draft_codes(values, 6), repr(values))

    def test_numeric_matrix_fastpath_preserves_strict_boundary(self):
        self.assertTrue(_matrix([[0, 1], [-2, 3]], 2, 2, integer=True))
        self.assertTrue(_matrix([[0.5, -1], [1.0, 2.0]], 2, 2))
        for bad in (True, False, 1.0, float("nan"), float("inf"),
                    -float("inf"), "1", None, [], {}):
            self.assertFalse(_matrix([[0, bad]], 1, 2, integer=True))
        for bad in (True, False, float("nan"), float("inf"),
                    -float("inf"), "1", None, [], {}):
            self.assertFalse(_matrix([[0, bad]], 1, 2))
        for bad_rows in (None, (), [[1, 2]], [[1], [1, 2]],
                         [[1, 2], (3, 4)]):
            self.assertFalse(_matrix(bad_rows, 2, 2))
        self.assertFalse(_matrix([[0, 1.0]], 1, 2, integer=True))
        self.assertTrue(_matrix([], 0, 2))

    def test_exact_1280_match_one_shared_learner_contract(self):
        report = verify_generation(self.manifest, self.payloads)
        self.assertEqual(report["accepted_complete_four_win_matches"],
                         MATCHES_PER_GENERATION)
        self.assertEqual(report["independent_seed_clusters"], 640)
        self.assertEqual(report["optimizer_steps_one_epoch"], OPTIMIZER_STEPS)
        self.assertEqual(report["opponent_roles"],
                         {"self": 8, "historical": 8, "fixed": 4})

    def test_actual_exploiter_role_only_if_immutable_registry_populated(self):
        m = manifest(exploiter=True)
        report = verify_generation(m, [payload(m, i) for i in range(ACTORS)])
        self.assertEqual(report["opponent_roles"],
                         {"self": 8, "historical": 6, "fixed": 4,
                          "exploiter": 2})

    def test_missing_or_duplicate_actor_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "20"):
            verify_generation(self.manifest, self.payloads[:-1])
        double = self.payloads[:]
        double[19] = self.payloads[18]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_generation(self.manifest, double)

    def test_missing_paired_seat_and_repeated_match_rejected(self):
        corrupted = copy.deepcopy(self.payloads[0])
        corrupted["matches"][-1] = copy.deepcopy(corrupted["matches"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_actor(self.manifest, corrupted)

    def test_mismatched_main_and_magnet_are_not_aggregated(self):
        for key in ("main_sha256", "magnet_sha256", "source_sha"):
            corrupted = copy.deepcopy(self.payloads[0])
            corrupted[key] = "0" * len(corrupted[key])
            with self.assertRaisesRegex(ValueError, "provenance"):
                verify_actor(self.manifest, corrupted)

    def test_wrong_accepted_trajectory_and_privileged_labels_rejected(self):
        corrupted = copy.deepcopy(self.payloads[0])
        corrupted["matches"][0]["events"][0]["teacher_utility"] = 10.
        with self.assertRaisesRegex(ValueError, "privileged"):
            verify_actor(self.manifest, corrupted)
        corrupted = copy.deepcopy(self.payloads[0])
        corrupted["matches"][0]["events"][0]["behavior_prob"] = 0.
        with self.assertRaisesRegex(ValueError, "probability"):
            verify_actor(self.manifest, corrupted)
        corrupted = copy.deepcopy(self.payloads[0])
        corrupted["matches"][0]["completed"] = False
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            verify_actor(self.manifest, corrupted)

    def test_stale_generation_and_seed_drift_rejected(self):
        corrupted = copy.deepcopy(self.payloads[0])
        corrupted["matches"][0]["seed"] += 1000000
        with self.assertRaisesRegex(ValueError, "seed"):
            verify_actor(self.manifest, corrupted)
        corrupted = copy.deepcopy(self.payloads[0])
        corrupted["generation"] += 1
        with self.assertRaisesRegex(ValueError, "provenance"):
            verify_actor(self.manifest, corrupted)

    def test_model_dimensions_not_silently_accepted_as_integration(self):
        m = copy.deepcopy(self.manifest)
        m["feature_schema"] = "rnad-three-phase-features-v5"
        with self.assertRaisesRegex(ValueError, "integrated"):
            verify_manifest(m)
        m = copy.deepcopy(self.manifest)
        m["optimizer_steps"] = 1280
        with self.assertRaisesRegex(ValueError, "40"):
            verify_manifest(m)


if __name__ == "__main__":
    unittest.main()
