"""Tests for the phase-level intermediate evidence gate; no fake wins."""
import copy
import unittest
from scripts.rnad.phase_imitation_goals import compare_heldout_imitation

def report():
    return {
        'schema':'rnad-aq-imitator-strategy-selection-eval-v1',
        'teacher_id':'rnad-aq-public-strategy-selection-v1',
        'source_sha':'a'*40,'engine_sha':'b'*40,
        'teacher_model_sha256':'c'*64,'student_model_sha256':'d'*64,
        'feature_schema':'rnad-three-phase-features-v2',
        'seed':8990001,'matches':32,'is_champion':False,
        'phases':{
            'draft':{'examples':352,'top1_accuracy':.04,
                     'top10_accuracy':.18,'mean_normalized_cross_entropy':.86},
            'selection':{'examples':352,'top1_accuracy':.05,
                         'top10_accuracy':.48,'mean_normalized_cross_entropy':.90},
        }
    }


class PhaseGoalsTest(unittest.TestCase):
    def test_paired_candidate_goals_remain_intermediate(self):
        base=report()
        new=copy.deepcopy(base)
        new['student_model_sha256']='e'*64
        new['phases']['draft'].update(top1_accuracy=.10,top10_accuracy=.35,
                                      mean_normalized_cross_entropy=.76)
        new['phases']['selection'].update(top1_accuracy=.17,top10_accuracy=.70,
                                          mean_normalized_cross_entropy=.73)
        result=compare_heldout_imitation(base,new)
        self.assertTrue(all(x['goal_screen_pass']
                            for x in result['phases'].values()))
        self.assertFalse(result['strength_established'])
        self.assertFalse(result['is_champion'])

    def test_fail_closed_if_source_teacher_seed_or_samples_mismatch(self):
        base=report()
        for key,value in (
            ('teacher_id','rnad-aq-public-belief-v2'),
            ('seed',100),('source_sha','f'*40),
            ('teacher_model_sha256','e'*64),
            ('student_model_sha256','d'*64),
        ):
            with self.subTest(key=key):
                new=copy.deepcopy(base)
                new[key]=value
                with self.assertRaises(ValueError):
                    compare_heldout_imitation(base,new)
        new=copy.deepcopy(base)
        new['student_model_sha256']='e'*64
        new['phases']['selection']['examples']=8
        with self.assertRaises(ValueError):
            compare_heldout_imitation(base,new)

    def test_thresholds_are_not_strength_and_need_improvement(self):
        base=report()
        new=copy.deepcopy(base)
        new['student_model_sha256']='e'*64
        screen=compare_heldout_imitation(base,new)
        self.assertFalse(screen['phases']['draft']['goal_screen_pass'])
        self.assertFalse(screen['phases']['selection']['goal_screen_pass'])


if __name__=='__main__':
    unittest.main()
