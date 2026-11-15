"""Executable correctness checks against DeepNash supplementary equations."""
import unittest
import torch
from scripts.rnad.neural import RNADNetwork, export_json, export_onnx, compare_onnx
from scripts.rnad.two_player import rnad_targets, neurd_policy_loss
import tempfile
from pathlib import Path
import subprocess
import json


def step(actor, reward=(0.0,0.0), pi=(.5,.5), magnet=(.5,.5),
         behavior=.5, value=0.0, action=0):
    return dict(actor=actor, reward=reward, pi=torch.tensor(pi),
                magnet=torch.tensor(magnet), behavior_prob=behavior,
                value=value, action_index=action)


class RnadMathTest(unittest.TestCase):
    def test_two_player_signed_terminal_reward_across_opponent_turn(self):
        # Seat 1 wins on the last decision. Seat 0 must receive -1
        # even though it did not act on that terminal event.
        targets=rnad_targets([step(0),step(1,(-1,1))],eta=0)
        self.assertAlmostEqual(targets[0]["value_target"],-1)
        self.assertAlmostEqual(targets[1]["value_target"],1)
        self.assertAlmostEqual(float(targets[0]["q_targets"][0]),-2)
        self.assertAlmostEqual(float(targets[1]["q_targets"][0]),2)
        self.assertAlmostEqual(targets[0]["ratio"],1)
        self.assertAlmostEqual(sum(targets[1]["regularized_reward"]),0)

    def test_reference_log_ratios_correct_both_seats(self):
        rows=[step(0,pi=(.75,.25),magnet=(.5,.5),behavior=.5),
              step(1,(-1,1),pi=(.25,.75),magnet=(.5,.5),behavior=.5)]
        output=rnad_targets(rows,eta=.2)
        self.assertAlmostEqual(output[0]["regularized_reward"][0],
                               -.2*torch.log(torch.tensor(1.5)).item(),places=5)
        self.assertAlmostEqual(output[0]["regularized_reward"][1],
                               +.2*torch.log(torch.tensor(1.5)).item(),places=5)
        self.assertAlmostEqual(output[1]["regularized_reward"][0],
                               -1+.2*torch.log(torch.tensor(.5)).item(),places=5)
        self.assertAlmostEqual(output[1]["regularized_reward"][1],
                               1-.2*torch.log(torch.tensor(.5)).item(),places=5)
        self.assertTrue(all(torch.isfinite(x["q_targets"]).all()
                            for x in output))

    def test_two_player_importance_crosses_other_seat_turn(self):
        # Three decisions: 0, 1, 0. The acting 0 at t0 sees the
        # opponent importance ratio product via xi and rhat.
        rows=[step(0,behavior=.5),step(1,pi=(.8,.2),behavior=.4),
              step(0,(1,-1),behavior=.5)]
        result=rnad_targets(rows,eta=0)
        self.assertAlmostEqual(result[0]["ratio"],1.0)
        self.assertAlmostEqual(result[0]["rho"],1.0)
        self.assertAlmostEqual(result[0]["value_target"],1.0)
        self.assertAlmostEqual(result[1]["ratio"],2.0)
        self.assertAlmostEqual(result[1]["regularized_reward"][0],0)

    def test_neurd_is_not_log_probability_ppo(self):
        logits=torch.tensor([0.1,0.2,-0.1],requires_grad=True)
        q=torch.tensor([1.,-.5,.5])
        loss=neurd_policy_loss(logits,q,beta=0,q_clip=10)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0].item(),0) # raise better action logit
        self.assertGreater(logits.grad[1].item(),0)
        # No behavior/ratio/clip PPO surrogate is used.

    def test_neurd_proposed_logit_boundary_is_respected(self):
        logits=torch.tensor([1.99,-1.99],requires_grad=True)
        advantage=torch.tensor([10.,-10.])
        loss=neurd_policy_loss(logits,advantage,beta=2.0,
                               logit_step=.1,q_clip=10.)
        loss.backward()
        self.assertTrue(torch.allclose(logits.grad,torch.zeros_like(logits)))
        logits2=torch.tensor([0.1,-0.1],requires_grad=True)
        neurd_policy_loss(logits2,advantage,beta=2.0,
                          logit_step=.01).backward()
        self.assertTrue(torch.isfinite(logits2.grad).all())
        self.assertGreater(float(logits2.grad.abs().sum()),0)

    def test_neurd_off_policy_gradient_is_scaled_by_two_player_c(self):
        def gradient(c):
            logits=torch.tensor([.2,-.1,.3],requires_grad=True)
            q=torch.tensor([1.0,-.25,.5])
            neurd_policy_loss(logits,q,beta=0,is_c=c).backward()
            return logits.grad.detach()
        full=gradient(1.0)
        self.assertGreater(float(full.abs().sum()),1e-6)
        self.assertTrue(torch.allclose(gradient(.5),full*.5,atol=1e-7))
        self.assertTrue(torch.equal(gradient(0),torch.zeros(3)))
        with self.assertRaises(ValueError):
            neurd_policy_loss(torch.tensor([0.,0.]),torch.tensor([1.,0.]),
                              is_c=1.01)

    def test_invalid_probability_and_zero_sum_fail_closed(self):
        with self.assertRaises(ValueError):
            rnad_targets([step(0,(1,1))])
        with self.assertRaises(ValueError):
            rnad_targets([step(0,behavior=0)])
        with self.assertRaises(ValueError):
            rnad_targets([])


class NeuralParityTest(unittest.TestCase):
    def test_model_backprop_and_export(self):
        torch.manual_seed(123)
        net=RNADNetwork(hidden=128)
        state=torch.randn(1,32)
        actions=torch.randn(1,4,16)
        logits,value,next_memory=net(state,actions,net.initial_memory())
        self.assertEqual(tuple(logits.shape),(1,4))
        self.assertEqual(tuple(value.shape),(1,))
        self.assertEqual(tuple(next_memory.shape),(1,128))
        (neurd_policy_loss(logits[0],torch.tensor([1.,-1.,.5,0.]))+
         value.square().mean()).backward()
        self.assertTrue(any(param.grad is not None and
                            torch.isfinite(param.grad).all()
                            for param in net.parameters()))
        with tempfile.TemporaryDirectory() as temp:
            payload=export_json(net,Path(temp)/'network.json')
            self.assertEqual(payload['schema'],'rnad-small-gru-v1')
            export_onnx(net,Path(temp)/'network.onnx')
            maxima=compare_onnx(net,Path(temp)/'network.onnx')
            self.assertLess(max(maxima),1e-4)
            fixture_state=torch.randn(1,32)
            fixture_actions=torch.randn(1,5,16)
            fixture_memory=torch.randn(1,128)
            with torch.no_grad():
                expected=net(fixture_state,fixture_actions,fixture_memory)
            fixture={'state':fixture_state[0].tolist(),
                     'actions':fixture_actions[0].tolist(),
                     'memory':fixture_memory[0].tolist()}
            (Path(temp)/'fixture.json').write_text(json.dumps(fixture))
            response=subprocess.run(['node','scripts/rnad/infer-check.mjs',
                                    str(Path(temp)/'network.json'),
                                    str(Path(temp)/'fixture.json')],
                                    check=True,capture_output=True,text=True)
            got=json.loads(response.stdout)
            self.assertLess(max(abs(a-b) for a,b in zip(
                expected[0][0].tolist(),got['logits'])),1e-4)
            self.assertLess(abs(float(expected[1][0])-got['value']),1e-4)
            self.assertLess(max(abs(a-b) for a,b in zip(
                expected[2][0].tolist(),got['memory'])),1e-4)


if __name__=='__main__':
    unittest.main()
