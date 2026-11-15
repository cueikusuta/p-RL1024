"""Small, pre-registered two-magnet schedule experiment on hosted CPU.

Independent from earlier 256-Match teacher warmstart and evaluation seeds.
This is a bounded 64-completed-Match research probe, NOT Champion evidence.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from scripts.rnad.train import train, current_git_sha, frozen_engine_sha
from scripts.rnad.neural import load_exported_json, RNADNetwork
from scripts.rnad.train import state_hash


SEED=976201
MATCHES=64
ETA=0.05
MAGNET_PERIOD=16


def run(*,teacher,seed=SEED,out='artifacts/rnad-eta-64'):
    if seed!=SEED:
        raise ValueError('Ablation needs the pre-registered disjoint training seed')
    root=Path(out)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Refuse to mix existing ablation artifacts')
    root.mkdir(parents=True,exist_ok=True)
    reference=RNADNetwork()
    teacher_json_sha=load_exported_json(reference,teacher)
    initial_hash=state_hash(reference)
    summaries=[]
    total=0
    latest=None
    for block in range(4):
        result=train(seed=seed,matches=16,out=str(root/'selfplay'),
                     resume=(block>0),initial_model=teacher,
                     eta=ETA,magnet_period=MAGNET_PERIOD)
        if (result.get('completed_matches')!=(block+1)*16 or
            result.get('updates')!=(block+1)*16 or
            result.get('initialization_sha256')!=teacher_json_sha or
            result.get('regularization_eta')!=ETA or
            result.get('magnet_period')!=MAGNET_PERIOD or
            result.get('source_sha')!=current_git_sha() or
            len(result.get('results',[]))!=16):
            raise ValueError('Incorrect frozen training series / parameter progress')
        want=list(range(seed+block*16,seed+(block+1)*16))
        if [x.get('seed') for x in result['results']]!=want:
            raise ValueError('Non-consecutive or duplicated Match seeds')
        count=sum(x['accepted_decisions'] for x in result['results'])
        if count<=0 or any(
            not all(math.isfinite(float(x[k])) for k in (
                'loss','critic_loss','policy_loss','behavior_ratio_max'))
            for x in result['results']):
            raise ValueError('Invalid accepted actions or learning objective')
        total+=count
        summaries.append({
            'batch':block+1,'matches':result['completed_matches'],
            'accepted_decisions':count,'model_sha256':result['model_sha256'],
            'magnet_sha256':result['magnet_sha256'],
            'checkpoint_sha256':result['last_match_checkpoint_sha']
        })
        latest=result
        print(json.dumps(summaries[-1]),flush=True)
    manifest=json.loads((root/'selfplay'/'checkpoints'/'latest.json').read_text())
    md=manifest['metadata']
    if (md.get('regularization_eta')!=ETA or
        md.get('magnet_period')!=MAGNET_PERIOD or
        md.get('initial_seed')!=seed or
        md.get('completed_matches')!=MATCHES or
        md.get('updates')!=MATCHES or
        md.get('accepted_decisions')!=total or
        md.get('initialization_sha256')!=teacher_json_sha or
        len(set(md.get('completed_match_ids',[])))!=MATCHES or
        manifest['checkpoint_sha256']!=latest['last_match_checkpoint_sha'] or
        initial_hash==latest['model_sha256']):
        raise ValueError('Incomplete 64-Match ablation evidence')
    report={
        'schema':'rnad-teacher-64-eta-ablation-v1',
        'experiment_status':'bounded research, not Champion',
        'source_sha':current_git_sha(),
        'engine_sha':frozen_engine_sha(),
        'teacher_json_sha256':teacher_json_sha,
        'teacher_model_sha256':initial_hash,
        'trained_model_sha256':latest['model_sha256'],
        'regularization_eta':ETA,
        'magnet_period':MAGNET_PERIOD,
        'seed':seed,'completed_matches':MATCHES,
        'optimizer_updates':MATCHES,'accepted_decisions':total,
        'checkpoint_sha256':manifest['checkpoint_sha256'],
        'batches':summaries,'is_champion':False
    }
    (root/'ablation-report.json').write_text(json.dumps(
        report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--teacher',required=True)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--out',default='artifacts/rnad-eta-64')
    a=parser.parse_args()
    print(json.dumps(run(teacher=a.teacher,seed=a.seed,out=a.out),indent=2))
