"""Research-only league plan tests. Zero training is performed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.rnad.night_league_2026 import (
    SCHEMA, SHARDS, MAX_LEAGUE_SECONDS_PER_WAVE, FIRST_SEED,
    verify_budget_and_seeds, strategy, run_one_wave,
)


class NightLeaguePlanTests(unittest.TestCase):
    def test_20_disjoint_lanes_and_frozen_pool_diversity(self):
        lanes = verify_budget_and_seeds()
        self.assertEqual(len(lanes), 20)
        self.assertEqual({x["shard"] for x in lanes}, set(range(20)))
        self.assertEqual({x["origin"] for x in lanes if x["mode"] == "league"},
                         {"base128", "base256", "c005"})
        self.assertEqual({x["origin"] for x in lanes if x["mode"] == "selfplay"},
                         {"battle-focus256-v5"})
        self.assertEqual({x["updates_per_match"] for x in lanes}, {1, 4})
        self.assertEqual({x["mode"] for x in lanes}, {"league", "selfplay"})
        self.assertEqual(len([x for x in lanes if x["mode"] == "league"]), 15)
        self.assertEqual(len([x for x in lanes if x["mode"] == "selfplay"]), 5)
        self.assertEqual(lanes[0]["seed"], FIRST_SEED)
        self.assertEqual(lanes[19]["seed"] - lanes[0]["seed"], 19 * 50000)

    def test_no_same_shard_or_extra_shard(self):
        for shard in (-1, 20, 21, 1.5, "2"):
            with self.subTest(shard=shard):
                with self.assertRaises(ValueError):
                    strategy(shard)

    def test_invalid_wave_or_time_rejected_before_any_model_access(self):
        with self.assertRaisesRegex(ValueError, "bounded"):
            run_one_wave(shard=0, wave=3, models="missing", output="missing", seconds=2)
        with self.assertRaisesRegex(ValueError, "bounded"):
            run_one_wave(shard=0, wave=1, models="missing", output="missing",
                         seconds=MAX_LEAGUE_SECONDS_PER_WAVE + 1)
        with self.assertRaisesRegex(ValueError, "bounded"):
            run_one_wave(shard=0, wave=1, models="missing", output="missing",
                         seconds=0)

    def test_study_is_exploratory_not_champion(self):
        src = Path("scripts/rnad/night_league_2026.py").read_text()
        self.assertIn('"league_eligibility_gate_passed": False', src)
        self.assertIn('"unresolved_independent_review_P1": True', src)
        self.assertIn('"champion_promoted": False', src)
        self.assertIn('"main_merged": False', src)
        self.assertIn("restore", Path("scripts/rnad/train.py").read_text())


if __name__ == "__main__":
    unittest.main()
