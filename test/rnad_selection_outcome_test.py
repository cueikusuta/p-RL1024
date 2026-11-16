import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.rnad.learn_selection_outcome import load_examples, train_outcome_rank


def sample(seed, pairs=2, ranks=3):
    rows = []
    for n in range(pairs):
        for seat in (0, 1):
            scenarios = []
            for rank in range(ranks):
                won = (n + seat + rank) % 2 == 0
                scenarios.append({
                    'rank':rank,'won':won,
                    'selected':json.dumps(['lead-'+str(rank),'a','b']),
                    'signature':'same-seed-seat-'+str(seed+n)+'-'+str(seat),
                    'features':[float(rank)/ranks]*48,
                    'state_features':[float(n+seat)/10]*96,
                    'legal_count':60,
                    'score':[4,2] if won and seat==0 or not won and seat==1 else [1,4]
                })
            rows.append({'seed':seed+n,'seat':seat,'scenarios':scenarios})
    return {'schema':'rnad-i5-v3-first-selection-outcome-probe-v1',
            'source_sha':'a'*40,'engine_sha':'b'*40,
            'v3_model_sha256':'c'*64,'aq_base_model_sha256':'d'*64,
            'seed':seed,'pairs':pairs,'ranks':ranks,'complete_matches':pairs*2*ranks,
            'rows':rows,'exploration_only':True,
            'strength_established':False,'champion_promoted':False,
            'baseline_wins':sum(row['scenarios'][0]['won'] for row in rows)}


class OutcomeStudyTest(unittest.TestCase):
    def test_realistic_schema_disjoint_training_and_heldout_remain_nonchampion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            t,h=root/'train.json',root/'heldout.json'
            t.write_text(json.dumps(sample(10000)))
            h.write_text(json.dumps(sample(20000)))
            result=train_outcome_rank(t,h,root/'model',epochs=3)
            self.assertFalse(result['strength_established'])
            self.assertFalse(result['selection_policy_deployed'])
            self.assertFalse(result['champion_promoted'])
            self.assertEqual(result['heldout_pairs'],2)
            self.assertEqual(len(result['choices']),4)
            self.assertEqual(len(result['model_file_sha256']),64)

    def test_mismatched_seed_or_forged_result_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            t,h=root/'train.json',root/'holdout.json'
            t.write_text(json.dumps(sample(10000)))
            h.write_text(json.dumps(sample(10001)))
            with self.assertRaises(ValueError):
                train_outcome_rank(t,h,root/'bad')
            corrupted=sample(30000)
            corrupted['rows'][0]['scenarios'][0]['features']=[float('nan')]*48
            h.write_text(json.dumps(corrupted))
            with self.assertRaises(ValueError):
                load_examples(h)
            malformed=sample(30000)
            malformed['rows'][0]['scenarios'][1]['signature']='mismatched-state'
            h.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):
                load_examples(h)
            malformed=sample(30000)
            malformed['rows'][0]['scenarios'][1]['selected']=malformed['rows'][0]['scenarios'][0]['selected']
            h.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):
                load_examples(h)


if __name__=='__main__':
    unittest.main()
