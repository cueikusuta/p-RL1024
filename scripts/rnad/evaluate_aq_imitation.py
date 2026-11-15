"""Seat/phase-stratified held-out imitation accuracy on actual AQ teacher
4-win Matches, with separate unseen seed block. Not a strength metric.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts.rnad.neural import RNADNetwork,load_exported_json,THREE_PHASE_FEATURE_SCHEMA
from scripts.rnad.train import current_git_sha,frozen_engine_sha
from scripts.rnad.train_teacher import collect,AQ_WEIGHTS_SHA256

def evaluate(model,*,seed=996301,matches=16,out):
    if not isinstance(seed,int) or not 1<=seed<=0xffffffff-32 or not 1<=matches<=32:
        raise ValueError('Invalid held-out data bounds')
    model=Path(model)
    torch.set_num_threads(1)
    net=RNADNetwork()
    model_sha=load_exported_json(net,model,
                               expected_feature_schema=THREE_PHASE_FEATURE_SCHEMA)
    net.eval()
    folder=Path(out)
    folder.mkdir(parents=True,exist_ok=False)
    traces=collect(seed,matches,folder,three_phase=True,
                   teacher='aq-public-belief')
    stat={phase:{'correct':0,'examples':0,'cross_entropy':0.0}
          for phase in ('draft','selection','battle')}
    with torch.no_grad():
        for record in traces:
            if record['teacher_model_sha256']!=AQ_WEIGHTS_SHA256:
                raise ValueError('Unknown frozen AQ teacher label')
            memory=[net.initial_memory() for _ in range(2)]
            for event in record['examples']:
                phase=event['phase']
                if phase not in stat:raise ValueError('Invalid phase')
                state=torch.tensor([event['state']],dtype=torch.float32)
                actions=torch.tensor([event['actions']],dtype=torch.float32)
                logits,_,hidden=net(state,actions,memory[event['actor']])
                memory[event['actor']]=hidden.detach()
                predicted=int(logits[0].argmax())
                chosen=event['choice']
                ce=float(F.cross_entropy(logits,torch.tensor([chosen])))
                if not math.isfinite(ce):raise ValueError('Nonfinite heldout crossentropy')
                row=stat[phase]
                row['correct']+=int(predicted==chosen)
                row['examples']+=1
                row['cross_entropy']+=ce
    for row in stat.values():
        if row['examples']==0:raise ValueError('Empty held-out phase')
        row['top1_accuracy']=row.pop('correct')/row['examples']
        row['mean_cross_entropy']=row.pop('cross_entropy')/row['examples']
    report={'schema':'rnad-aq-imitator-unseen-match-eval-v1',
            'feature_schema':THREE_PHASE_FEATURE_SCHEMA,
            'teacher_model_sha256':AQ_WEIGHTS_SHA256,
            'student_model_sha256':model_sha,
            'source_sha':current_git_sha(),'engine_sha':frozen_engine_sha(),
            'seed':seed,'matches':matches,'phases':stat,
            'is_champion':False}
    (folder/'heldout-imitation.json').write_text(
        json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--seed',type=int,default=996301)
    p.add_argument('--matches',type=int,default=16)
    p.add_argument('--out',required=True)
    opts=p.parse_args()
    print(json.dumps(evaluate(opts.model,seed=opts.seed,matches=opts.matches,
                              out=opts.out),indent=2))
