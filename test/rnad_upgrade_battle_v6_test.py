"""Contract tests for C-only v6 tensor/optimizer migration (no model claims)."""
import copy
import unittest

import torch
from torch import nn

from scripts.rnad.neural import RNADNetwork
from scripts.rnad.upgrade_battle_v6 import upgrade_checkpoint


class MiniModel(nn.Module):
    def __init__(self, state, action):
        super().__init__()
        self.battle = RNADNetwork(state_features=state,action_features=action)


def initialized():
    torch.manual_seed(7)
    old=MiniModel(96,48)
    new=MiniModel(384,160)
    opt=torch.optim.Adam(old.parameters(),lr=3e-5)
    state=torch.rand(1,96)
    moves=torch.rand(1,3,48)
    logits,value,_=old.battle(state,moves,old.battle.initial_memory())
    (logits.sum()+value.sum()).backward()
    opt.step()
    checkpoint={
        'model':copy.deepcopy(old.state_dict()),
        'target':copy.deepcopy(old.state_dict()),
        'magnet':copy.deepcopy(old.state_dict()),
        'optimizer':copy.deepcopy(opt.state_dict()),
        'scheduler':{'last_epoch':4},
        'torch_rng':torch.get_rng_state(),
        'python_rng':('dummy',),
        'is_champion':False,
    }
    return old,new,checkpoint


class V6IntegratedModelTest(unittest.TestCase):
    def test_full_integrated_v6_reuses_draft_selection_heads_and_backpropagates(self):
        from scripts.rnad.integrated_policy_v1 import IntegratedPolicyV1
        from scripts.rnad.neural import THREE_PHASE_FEATURE_SCHEMA_V6, export_json
        from scripts.rnad.train_selection_specialist import TRIPLES
        from scripts.rnad.train_draft_deepsets import FEATURE_WIDTH
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        vocab={'sizes': {key:3 for key in
                         ('species','names','abilities','items','moves','types')}}
        model=IntegratedPolicyV1(vocab,
              battle_schema=THREE_PHASE_FEATURE_SCHEMA_V6)
        self.assertEqual(model.battle.state_features,384)
        self.assertEqual(model.battle.action_features,160)
        draft={'pool':torch.randn(6,FEATURE_WIDTH),
               'extra':torch.randn(1,4)}
        selection={'own_cat':torch.zeros(1,6,9,dtype=torch.long),
                   'foe_cat':torch.zeros(1,6,3,dtype=torch.long),
                   'own_num':torch.randn(1,6,16),
                   'foe_num':torch.randn(1,6,6)}
        battle={'state':torch.randn(1,384)}
        seq=[
            ('draft',draft,torch.tensor([[0,1,2,3,4,5]],dtype=torch.long),None),
            ('selection',selection,TRIPLES,None),
            ('battle',battle,torch.randn(1,4,160),model.initial_memory()),
        ]
        values=[]
        for phase,view,actions,mem in seq:
            logits,critic,next_mem=model(phase,view,actions,mem)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(critic).all())
            values.append(logits.mean()+critic)
        sum(values).backward()
        for tensor in (model.battle.state_proj.weight,
                       model.battle.policy_hidden.weight,
                       model.selection_value[0].weight,
                       model.draft_value[0].weight):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        with TemporaryDirectory() as temp:
            target=Path(temp)/'battle.json'
            export_json(model.battle,target,
                        feature_schema=THREE_PHASE_FEATURE_SCHEMA_V6)
            payload=json.loads(target.read_text())
            self.assertEqual((payload['state_features'],
                              payload['action_features']),(384,160))
            self.assertEqual(payload['feature_schema'],
                             THREE_PHASE_FEATURE_SCHEMA_V6)


class UpgradeTest(unittest.TestCase):
    def test_expands_all_three_heads_and_optimizer_without_changing_prefix(self):
        old,new,original=initialized()
        upgraded=upgrade_checkpoint(original,old,new)
        self.assertEqual(upgraded['battle_feature_schema'],
                         'rnad-three-phase-features-v6')
        for key in ('model','target','magnet'):
            old_state=original[key]
            new_state=upgraded[key]
            for name,v in old_state.items():
                if name=='battle.state_proj.weight':
                    self.assertTrue(torch.equal(v,new_state[name][:,:96]))
                    self.assertEqual(torch.count_nonzero(new_state[name][:,96:]),0)
                elif name=='battle.policy_hidden.weight':
                    self.assertTrue(torch.equal(v,new_state[name][:,:176]))
                    self.assertEqual(torch.count_nonzero(new_state[name][:,176:]),0)
                else:self.assertTrue(torch.equal(v,new_state[name]))
        old_named=list(old.named_parameters())
        ids=original['optimizer']['param_groups'][0]['params']
        for pid,(name,param) in zip(ids,old_named):
            old_slots=original['optimizer']['state'][pid]
            new_slots=upgraded['optimizer']['state'][pid]
            for slot in ('exp_avg','exp_avg_sq'):
                if name in ('battle.state_proj.weight','battle.policy_hidden.weight'):
                    self.assertTrue(torch.equal(
                        old_slots[slot],new_slots[slot][:,:param.shape[1]]))
                    self.assertEqual(torch.count_nonzero(
                        new_slots[slot][:,param.shape[1]:]),0)
                else:self.assertTrue(torch.equal(old_slots[slot],new_slots[slot]))
        self.assertTrue(torch.equal(
            original['model']['battle.state_proj.weight'],
            old.state_dict()['battle.state_proj.weight']))
        new.load_state_dict(upgraded['model'],strict=True)
        old.eval();new.eval()
        x=torch.rand(1,96)
        actions=torch.rand(1,3,48)
        with torch.no_grad():
            ol,ov,_=old.battle(x,actions,old.battle.initial_memory())
            nl,nv,_=new.battle(torch.cat([x,torch.zeros(1,288)],dim=-1),
                torch.cat([actions,torch.zeros(1,3,112)],dim=-1),
                new.battle.initial_memory())
        self.assertTrue(torch.allclose(ol,nl,atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.allclose(ov,nv,atol=1e-6,rtol=1e-6))

    def test_fail_closed_on_missing_or_untrusted_state(self):
        old,new,ckpt=initialized()
        x=copy.deepcopy(ckpt);x['is_champion']=True
        with self.assertRaises(ValueError):upgrade_checkpoint(x,old,new)
        x=copy.deepcopy(ckpt);x['optimizer']['state'].clear()
        with self.assertRaises(ValueError):upgrade_checkpoint(x,old,new)
        x=copy.deepcopy(ckpt);x['model'].pop('battle.state_proj.weight')
        with self.assertRaises(ValueError):upgrade_checkpoint(x,old,new)


if __name__=='__main__':
    unittest.main()
