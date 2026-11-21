"""Bounded utility-target supervision is a diagnostic, not a win signal."""
import math
import tempfile
import unittest
from pathlib import Path
import torch
from scripts.rnad.train_teacher import (
    utility_teacher_loss,normalized_utility_regret,initialize_teacher_model)
from scripts.rnad.neural import (
    THREE_PHASE_FEATURE_SCHEMA_V4,THREE_PHASE_FEATURE_SCHEMA_V2,
    export_json,load_exported_json)


class AQUtilityTests(unittest.TestCase):
    def test_better_student_rank_has_lower_soft_ce_and_regret(self):
        scores=[10.,9.,1.,0.]
        a=utility_teacher_loss(torch.tensor([[4.,3.,-2.,-3.]]),scores,'selection')
        b=utility_teacher_loss(torch.tensor([[-2.,-3.,4.,3.]]),scores,'selection')
        self.assertTrue(bool(torch.isfinite(a)))
        self.assertLess(float(a),float(b))
        self.assertEqual(normalized_utility_regret(scores,0),0)
        self.assertEqual(normalized_utility_regret(scores,3),1)
        self.assertAlmostEqual(normalized_utility_regret(scores,1),.1)
        hard_target=torch.tensor([0])
        first=torch.tensor([[0.,0.,0.,0.]],requires_grad=True)
        soft=utility_teacher_loss(first,scores,'selection')
        hard=torch.nn.functional.cross_entropy(first,hard_target)
        mixed=.5*soft+.5*hard
        self.assertTrue(bool(torch.isfinite(mixed)))
        mixed.backward()
        self.assertTrue(bool(torch.isfinite(first.grad).all()))
        self.assertGreater(float(soft),0)


    def test_fail_closed_on_invalid_public_utility_label(self):
        for utilities in ([float('nan'),0],[1.],[1.,float('inf')],[None,2]):
            with self.subTest(utilities=utilities),self.assertRaises(ValueError):
                utility_teacher_loss(torch.zeros((1,2)),utilities,'draft')
        with self.assertRaises(ValueError):
            utility_teacher_loss(torch.zeros((1,2)),[1,2],'battle')
        with self.assertRaises(ValueError):
            normalized_utility_regret([1.,2.],-1)

    def test_version_bound_warmstart_preserves_input_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'v4.json'
            model,_=initialize_teacher_model(THREE_PHASE_FEATURE_SCHEMA_V4)
            self.assertEqual((model.state_features,model.action_features),(48,32))
            export_json(model,path,feature_schema=THREE_PHASE_FEATURE_SCHEMA_V4)
            original=path.read_bytes()
            copy,digest=initialize_teacher_model(THREE_PHASE_FEATURE_SCHEMA_V4,path)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(copy.state_features,48)
            self.assertIsInstance(digest,str)
            with self.assertRaises(ValueError):
                initialize_teacher_model(THREE_PHASE_FEATURE_SCHEMA_V2,path)


if __name__=='__main__':
    unittest.main()
