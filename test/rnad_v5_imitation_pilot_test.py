import copy
import unittest
from scripts.rnad.compare_v5_imitation_pilot import compare_pilot


def make(schema):
    return {'schema':'rnad-aq-imitator-strategy-selection-eval-v1',
            'source_sha':'a'*40,'engine_sha':'b'*40,
            'teacher_model_sha256':'c'*64,
            'teacher_id':'rnad-aq-public-strategy-selection-v1',
            'student_model_sha256':('d' if schema.endswith('v4') else 'e')*64,
            'feature_schema':schema,
            'utility_labels':'aq-public-phase-rank-v1',
            'seed':2311001,'matches':4,'is_champion':False,
            'phases':{p:{'examples':42,'top1_accuracy':.2,
                         'top10_accuracy':.5,
                         'mean_normalized_cross_entropy':.9,
                         'mean_normalized_utility_regret':.25}
                      for p in ('draft','selection','battle')}}


class PilotReportTest(unittest.TestCase):
    def test_exploration_not_formal_gate(self):
        r=compare_pilot(make('rnad-three-phase-features-v4'),
                        make('rnad-three-phase-features-v5'))
        self.assertFalse(r['strength_established'])
        self.assertFalse(r['champion_promoted'])
        self.assertTrue(r['formal_imitation_goal_gate_not_applied'])
        self.assertEqual(r['phases']['selection']['examples'],42)

    def test_teacher_provenance_and_same_source_required(self):
        a=make('rnad-three-phase-features-v4')
        for change in ('seed','source_sha','teacher_id','utility_labels'):
            b=make('rnad-three-phase-features-v5')
            b[change]='wrong'
            with self.subTest(change=change),self.assertRaises(ValueError):
                compare_pilot(a,b)
        b=make('rnad-three-phase-features-v5')
        b['phases']['selection']['examples']=5
        with self.assertRaises(ValueError):
            compare_pilot(a,b)


if __name__=='__main__':
    unittest.main()
