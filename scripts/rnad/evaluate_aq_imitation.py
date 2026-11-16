"""Seat/phase-stratified held-out imitation accuracy on actual AQ teacher
4-win Matches, with separate unseen seed block. Not a strength metric.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from scripts.rnad.neural import (RNADNetwork, load_exported_json,
                                 THREE_PHASE_FEATURE_SCHEMA,
                                 THREE_PHASE_FEATURE_SCHEMA_V2)
from scripts.rnad.train import current_git_sha,frozen_engine_sha
from scripts.rnad.train_teacher import collect,AQ_WEIGHTS_SHA256

def per_decision_metrics(logits, chosen):
    """Model/label-only held-out metrics, never a game strength claim.

    Log-count is the uniform-random teacher CE for this decision; a Top-1
    number without the actual legal candidate count is misleading across
    large draft candidates versus 60 selection candidates.
    """
    if (not isinstance(logits,torch.Tensor) or logits.ndim!=1 or
        logits.numel()<1 or not bool(torch.isfinite(logits).all()) or
        type(chosen) is not int or not 0<=chosen<logits.numel()):
        raise ValueError('Invalid finite teacher logits / choice')
    n=logits.numel()
    order=torch.argsort(logits,descending=True,stable=True)
    # If logits tie, choosing an arbitrary first index must not receive
    # privileged top-k credit. This is explicitly a fixed-order rank.
    pos=int(torch.nonzero(order==chosen,as_tuple=False)[0,0])
    logp=torch.log_softmax(logits,dim=0)
    probs=logp.exp()
    ce=-float(logp[chosen])
    entropy=-float(torch.sum(probs*logp))
    uniform_ce=math.log(n)
    if not all(map(math.isfinite,(ce,entropy,uniform_ce))):
        raise ValueError('Nonfinite teacher agreement diagnostics')
    return {'candidates':n,'top1':int(pos==0),
        'top3':int(pos<min(3,n)),'top10':int(pos<min(10,n)),
        'teacher_action_probability':float(probs[chosen]),
        'cross_entropy':ce,'entropy':entropy,
        'uniform_cross_entropy':uniform_ce,
        'uniform_random_top1':1/n,
        'uniform_random_top3':min(3,n)/n,
        'uniform_random_top10':min(10,n)/n,
        'log_probability_gain_vs_uniform':uniform_ce-ce,
        'normalized_cross_entropy':
            ce/uniform_ce if n>1 else 0.0}


def evaluate(model,*,seed=996301,matches=16,out,features_v2=False):
    if not isinstance(seed,int) or not 1<=seed<=0xffffffff-32 or not 1<=matches<=32:
        raise ValueError('Invalid held-out data bounds')
    model=Path(model)
    torch.set_num_threads(1)
    net=RNADNetwork()
    schema=THREE_PHASE_FEATURE_SCHEMA_V2 if features_v2 else THREE_PHASE_FEATURE_SCHEMA
    model_sha=load_exported_json(net,model,
                               expected_feature_schema=schema)
    net.eval()
    folder=Path(out)
    folder.mkdir(parents=True,exist_ok=False)
    traces=collect(seed,matches,folder,three_phase=True,
                   teacher='aq-public-belief',features_v2=features_v2)
    stat={phase:{'correct':0,'examples':0,'cross_entropy':0.0,
                 'teacher_top3':0,'teacher_top10':0,'legal_candidates':0,
                 'teacher_probability_sum':0.0,'entropy_sum':0.0,
                 'uniform_ce_sum':0.0,'uniform_top1_sum':0.0,
                 'uniform_top3_sum':0.0,'uniform_top10_sum':0.0,
                 'log_probability_gain_sum':0.0,
                 'normalized_ce_sum':0.0}
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
                chosen=event['choice']
                score=per_decision_metrics(logits[0],chosen)
                row=stat[phase]
                row['correct']+=score['top1']
                row['examples']+=1
                row['cross_entropy']+=score['cross_entropy']
                row['teacher_top3']+=score['top3']
                row['teacher_top10']+=score['top10']
                row['legal_candidates']+=score['candidates']
                row['teacher_probability_sum']+=score['teacher_action_probability']
                row['entropy_sum']+=score['entropy']
                row['uniform_ce_sum']+=score['uniform_cross_entropy']
                row['uniform_top1_sum']+=score['uniform_random_top1']
                row['uniform_top3_sum']+=score['uniform_random_top3']
                row['uniform_top10_sum']+=score['uniform_random_top10']
                row['log_probability_gain_sum']+=score['log_probability_gain_vs_uniform']
                row['normalized_ce_sum']+=score['normalized_cross_entropy']
    for row in stat.values():
        if row['examples']==0:raise ValueError('Empty held-out phase')
        n=row['examples']
        row['top1_accuracy']=row.pop('correct')/n
        row['mean_cross_entropy']=row.pop('cross_entropy')/n
        row['top3_accuracy']=row.pop('teacher_top3')/n
        row['top10_accuracy']=row.pop('teacher_top10')/n
        row['mean_candidates']=row.pop('legal_candidates')/n
        row['mean_teacher_action_probability']=row.pop('teacher_probability_sum')/n
        row['mean_policy_entropy_nats']=row.pop('entropy_sum')/n
        row['uniform_baseline_cross_entropy_nats']=row.pop('uniform_ce_sum')/n
        row['uniform_baseline_top1']=row.pop('uniform_top1_sum')/n
        row['uniform_baseline_top3']=row.pop('uniform_top3_sum')/n
        row['uniform_baseline_top10']=row.pop('uniform_top10_sum')/n
        row['mean_log_probability_gain_vs_uniform']=row.pop('log_probability_gain_sum')/n
        row['mean_normalized_cross_entropy']=row.pop('normalized_ce_sum')/n
        # Sanity checks are descriptive only; labels are not wins.
        if (row['top1_accuracy']>row['top3_accuracy']+1e-12 or
            row['top3_accuracy']>row['top10_accuracy']+1e-12 or
            not all(math.isfinite(v) for v in row.values())):
            raise ValueError('Invalid held-out phase agreement metrics')
    report={'schema':'rnad-aq-imitator-unseen-match-eval-v1',
            'feature_schema':schema,
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
    p.add_argument('--features-v2',action='store_true')
    opts=p.parse_args()
    print(json.dumps(evaluate(opts.model,seed=opts.seed,matches=opts.matches,
                              out=opts.out,features_v2=opts.features_v2),indent=2))
