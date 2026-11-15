"""Fast fail-closed contract tests for RNAD population research."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.rnad.population_probe import (
    FEATURE_SCHEMA, variants, mutate, verified_weights, validate_report, digest,
)


def fixture():
    return {'schema': 'rnad-small-gru-v1',
            'feature_schema': FEATURE_SCHEMA,
            'state_features': 32, 'action_features': 16, 'hidden': 128,
            'weights': {
                'policy_hidden.weight': {'shape': [128, 144],
                                         'values': [.1]*18432},
                'policy_out.weight': {'shape': [1, 128],
                                      'values': [.2]*128},
                'state_proj.weight': {'shape': [128,32],
                                      'values': [.3]*4096}}}


class PopulationContractTest(unittest.TestCase):
    def test_exactly_100_reproducible_diverse_variants(self):
        a = variants()
        self.assertEqual(a, variants())
        self.assertEqual(len(a), 100)
        self.assertEqual(len({x['id'] for x in a}), 100)
        self.assertEqual(a[0]['family'], 'unchanged-control')
        self.assertEqual(sum(x['family']=='guided-action' for x in a), 32)
        self.assertEqual(sum(x['family']=='guided-state' for x in a), 32)
        self.assertEqual(sum(x['family']=='random-policy' for x in a), 35)
        with self.assertRaises(ValueError):
            variants(99)

    def test_all_mutations_finite_reproducible_and_v2_only(self):
        base = fixture()
        verified_weights(base)
        self.assertEqual(mutate(base, variants()[0]), base)
        for spec in variants()[1:]:
            a = mutate(base, spec)
            self.assertNotEqual(a, base)
            self.assertEqual(a, mutate(base, spec))
            self.assertEqual(a['feature_schema'], FEATURE_SCHEMA)
            verified_weights(a)
        self.assertEqual(base, fixture())
        broken = copy.deepcopy(base)
        broken['feature_schema'] = 'rnad-three-phase-features-v1'
        with self.assertRaises(ValueError):
            mutate(broken, variants()[1])
        broken = copy.deepcopy(base)
        broken['weights']['policy_out.weight']['values'][0] = float('nan')
        with self.assertRaises(ValueError):
            mutate(broken, variants()[1])

    def test_paired_report_rejects_duplicate_seat_and_forged_score(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.json'
            path.write_text('candidate-model')
            rows = [
                {'seed': 1301001, 'candidateSeat': 0, 'won': True,
                 'score':[4,2], 'rounds': 6},
                {'seed': 1301001, 'candidateSeat': 1, 'won': False,
                 'score':[4,2], 'rounds': 6},
            ]
            report = {
                'schema': 'rnad-three-phase-vs-basic-paired-v1',
                'source_sha': 'a'*40, 'engine_sha': 'b'*40,
                'inference_selection_mode': 'greedy',
                'model_sha256': digest(path), 'opponent_id': 'basic-all-phases',
                'pairs': 1, 'matches': 2, 'wins': 1,
                'rows': rows, 'champion_promoted':False,
            }
            with patch('scripts.rnad.population_probe.source_sha',return_value='a'*40), \
                 patch('scripts.rnad.population_probe.engine_sha',return_value='b'*40):
                validate_report(report,model=path,seed=1301001,pairs=1)
                bad = copy.deepcopy(report)
                bad['rows'][1]['candidateSeat'] = 0
                with self.assertRaises(ValueError):
                    validate_report(bad,model=path,seed=1301001,pairs=1)
                bad = copy.deepcopy(report)
                bad['rows'][0]['score'] = [3,2]
                with self.assertRaises(ValueError):
                    validate_report(bad,model=path,seed=1301001,pairs=1)
                bad = copy.deepcopy(report)
                bad['source_sha'] = 'c'*40
                with self.assertRaises(ValueError):
                    validate_report(bad,model=path,seed=1301001,pairs=1)
                bad = copy.deepcopy(report)
                bad['champion_promoted'] = True
                with self.assertRaises(ValueError):
                    validate_report(bad,model=path,seed=1301001,pairs=1)


if __name__ == '__main__':
    unittest.main()
