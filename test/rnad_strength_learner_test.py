"""G49 training contract checks only; never claim strength from unit tests."""
import unittest
import torch
from scripts.rnad.shared_league_learner import (
    _phase_policy_weight, _critic_loss, DRAFT_POLICY_GRADIENT_WEIGHT,
    VALUE_TERMINAL_AUXILIARY_WEIGHT,
)
from scripts.rnad.shared_league_contract import actor_role

class StrengthTest(unittest.TestCase):
    def test_scales(self):
        self.assertEqual(DRAFT_POLICY_GRADIENT_WEIGHT, 3.0)
        self.assertEqual([_phase_policy_weight(x) for x in ("draft", "selection", "battle")], [3., 1., 1.])
        with self.assertRaises(ValueError):
            _phase_policy_weight("wrong")
    def test_critic_training_label(self):
        x=torch.tensor(.2,requires_grad=True)
        v=_critic_loss(x,.3,winner=0,actor=0)
        self.assertAlmostEqual(float(v),(.2-.3)**2+VALUE_TERMINAL_AUXILIARY_WEIGHT*(.2-1)**2,places=6)
        v.backward()
        self.assertTrue(bool(torch.isfinite(x.grad)))
        self.assertGreater(float(_critic_loss(torch.tensor(.2),.3,winner=0,actor=1)),float(v))
        with self.assertRaises(ValueError):
            _critic_loss(x,.3,winner=3,actor=1)
    def test_roles(self):
        for exploiter,want in [(False,{"self":8,"historical":10,"fixed":2}),
                               (True,{"self":8,"historical":8,"fixed":2,"exploiter":2})]:
            got={}
            for i in range(20):
                role=actor_role(i,has_exploiter=exploiter)
                got[role]=got.get(role,0)+1
            self.assertEqual(got,want)
if __name__=="__main__": unittest.main()
