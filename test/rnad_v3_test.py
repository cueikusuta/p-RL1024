"""v3 is a new model contract, not an in-place relabel of frozen v2."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from scripts.rnad.neural import (
  RNADNetwork, freeze_magnet, export_json, load_exported_json,
  migrate_v2_export_to_v3, THREE_PHASE_FEATURE_SCHEMA_V2,
  THREE_PHASE_FEATURE_SCHEMA_V3)
from scripts.rnad.train import train_match


class V3RepresentationTest(unittest.TestCase):
  def test_v2_to_v3_warmstart_preserves_old_zero_extra_logits(self):
    torch.manual_seed(347)
    old=RNADNetwork()
    expanded=RNADNetwork(state_features=48,action_features=32)
    with tempfile.TemporaryDirectory() as t:
      source=Path(t)/'frozen-v2.json'
      export_json(old,source,feature_schema=THREE_PHASE_FEATURE_SCHEMA_V2)
      origin=migrate_v2_export_to_v3(expanded,source)
      self.assertEqual(len(origin),64)
      state=torch.randn(1,32)
      legal=torch.randn(1,5,16)
      mem=torch.randn(1,128)
      with torch.no_grad():
        a=old(state,legal,mem)
        b=expanded(torch.cat([state,torch.zeros(1,16)],-1),
                   torch.cat([legal,torch.zeros(1,5,16)],-1),mem)
      for x,y in zip(a,b):
        self.assertTrue(torch.allclose(x,y,atol=1e-6))
      other=Path(t)/'v3.json'
      payload=export_json(expanded,other,
                          feature_schema=THREE_PHASE_FEATURE_SCHEMA_V3)
      self.assertEqual(payload['state_features'],48)
      self.assertEqual(payload['action_features'],32)
      loaded=RNADNetwork(state_features=48,action_features=32)
      self.assertEqual(load_exported_json(loaded,other,
          expected_feature_schema=THREE_PHASE_FEATURE_SCHEMA_V3).__len__(),64)
      with self.assertRaises(ValueError):
        load_exported_json(RNADNetwork(),other,
                           expected_feature_schema=THREE_PHASE_FEATURE_SCHEMA_V2)

  def test_reusing_same_accepted_match_for_four_bounded_updates(self):
    torch.manual_seed(349)
    net=RNADNetwork(state_features=48,action_features=32)
    target=copy.deepcopy(net)
    magnet=freeze_magnet(net)
    opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=100,gamma=.98)
    events=[]
    for seat in (0,1):
      state=[.1]*48
      candidates=[[.2]*32,[.5]*32]
      with torch.no_grad():
        logits,_,_=magnet(torch.tensor([state]),
                          torch.tensor([candidates]),magnet.initial_memory())
        p=torch.softmax(logits[0],dim=-1)
      events.append({'actor':seat,'state':state,'actions':candidates,
        'action_index':0,'behavior_prob':float(.98*p[0]+.02/2),
        'magnet_log_prob':float(torch.log(p[0])),
        'reward':[0,0] if seat==0 else [-1,1]})
    first=copy.deepcopy(net.state_dict())
    for _ in range(4):
      result=train_match(net,target,magnet,opt,scheduler,
                         {'events':events},eta=.2)
      self.assertEqual(result['accepted_decisions'],2)
    self.assertEqual(scheduler.last_epoch,4)
    self.assertTrue(any(not torch.equal(first[k],p)
                        for k,p in net.state_dict().items()))

if __name__=='__main__':
  unittest.main()
