"""Fast deterministic AQ teacher score and v4 alias diagnostics."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.rnad.audit_teacher_utility import (
    analyze_events,audit_trace_files,EXPECTED_SHA,EXPECTED_TEACHER
)

def event(phase,scores,alias=False):
    actions=[[float(i)]+[0.]*31 for i in range(len(scores))]
    if alias and len(actions)>1:
        actions[1]=actions[0].copy()
    return {'phase':phase,'actions':actions,
            'candidate_utilities':scores,'choice':0}

class SignalAuditTest(unittest.TestCase):
    def test_topk_ties_and_distinct_teacher_scores_at_same_features(self):
        draft=event('draft',[10.]+[9.]*12,alias=True)
        selection=event('selection',[10.]+[9.]*10,alias=True)
        metrics=analyze_events([draft,selection])
        for phase in ('draft','selection'):
            row=metrics[phase]
            self.assertEqual(row['decisions'],1)
            self.assertEqual(row['topk_boundary_tie_fraction'],1.)
            self.assertEqual(row['feature_alias_decision_fraction'],1.)
            self.assertEqual(row['aliased_teacher_choice_fraction'],1.)
            self.assertEqual(row['maximum_utility_difference_at_equal_feature'],1.)
            self.assertEqual(row['top1_tied_fraction'],0.)

    def test_reject_invalid_utility_or_phase_provenance(self):
        for bad in [
            event('draft',[2.,float('nan')]),
            event('draft',[2.,1.])|{'choice':1},
            event('selection',[1.,0.])|{'actions':[[1.]*32]},
            event('unknown',[1.,0.]),
        ]:
            with self.subTest(event=bad),self.assertRaises(ValueError):
                analyze_events([bad,event('selection',[1.,0.])])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i in (0,1):
                (root/f'teacher-{i}.json').write_text(json.dumps({
                    'teacher':EXPECTED_TEACHER,
                    'teacher_model_sha256':EXPECTED_SHA,
                    'utility_labels':'aq-public-phase-rank-v1',
                    'feature_schema':'rnad-three-phase-features-v4',
                    'source_sha':'a'*40,'engine_sha':'b'*40,
                    'completed':True,'score':[4,0],'winner':0,'seed':i+1,
                    'examples':[event('draft',[10.,9.]),
                                event('selection',[10.,9.])]
                }))
            a=audit_trace_files(root.glob('teacher-*.json'),2)
            self.assertEqual(a['accepted_matches'],2)
            self.assertFalse(a['strength_established'])
            self.assertFalse(a['champion_promoted'])
            with self.assertRaises(ValueError):
                audit_trace_files(root.glob('teacher-*.json'),3)

if __name__=='__main__':
    unittest.main()
