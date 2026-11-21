"""Optional 256-Match RNAD continuation AFTER Stage0 teacher imitation.

This research AB experiment is separate from pure RNAD, source-bound, and
NOT evidence of Champion status. Teacher and RL run on SAME pinned checkout.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch

from scripts.rnad.neural import RNADNetwork, load_exported_json
from scripts.rnad.train import train, state_hash, current_git_sha
from scripts.rnad.train_teacher import train as imitate


def run(*,out='artifacts/rnad-warmstart',seed=975801):
    if seed!=975801:raise ValueError('Pinned isolated RNAD research seed required')
    root=Path(out)
    root.mkdir(parents=True,exist_ok=True)
    teacher=imitate(seed=975601,out=str(root/'teacher'))
    source=root/'teacher'/'trained-model.json'
    init=RNADNetwork()
    initialization_sha256=load_exported_json(init,source)
    initial_weight_sha=state_hash(init)
    if initial_weight_sha!=teacher['trained_model_sha256']:
        raise ValueError('Teacher init weights differ from training report')
    summaries=[]
    total=0
    final=None
    for batch in range(16):
        result=train(seed=seed,matches=16,out=str(root/'selfplay'),
                     resume=batch!=0,initial_model=str(source))
        if (result['completed_matches']!=(batch+1)*16 or
            result['updates']!=(batch+1)*16 or
            result.get('initialization_sha256')!=initialization_sha256):
            raise ValueError('Warm-start R-NaD progress/provenance mismatch')
        accepted=sum(r['accepted_decisions'] for r in result['results'])
        total+=accepted
        summary={'batch':batch+1,
          'completed_matches':result['completed_matches'],
          'accepted_decisions':accepted,
          'model_sha256':result['model_sha256'],
          'mean_loss':sum(r['loss'] for r in result['results'])/16}
        if not math.isfinite(summary['mean_loss']):
            raise ValueError('Nonfinite warm-start R-NaD loss')
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
        final=result
    metadata=json.loads((root/'selfplay'/'checkpoints'/'latest.json').read_text())['metadata']
    if (metadata.get('completed_matches')!=256 or
        metadata.get('updates')!=256 or
        metadata.get('accepted_decisions')!=total or
        len(set(metadata.get('completed_match_ids',[])))!=256 or
        metadata.get('initialization_sha256')!=initialization_sha256 or
        final['model_sha256']==initial_weight_sha):
        raise ValueError('Incorrect pretrained R-NaD Checkpoint/weight progression')
    report={'schema':'rnad-optional-warmstart-256-research-v1',
       'algorithm':'RNAD after Stage0 fixed-policy imitation',
       'source_sha':current_git_sha(),
       'teacher_seed':975601,'teacher_matches':32,'teacher_updates':128,
       'selfplay_seed':seed,'selfplay_matches':256,'rnad_updates':256,
       'accepted_actions':total,
       'teacher_initial_model_sha256':initial_weight_sha,
       'initialization_json_sha256':initialization_sha256,
       'rnad_trained_sha256':final['model_sha256'],
       'batches':summaries,'is_champion':False}
    (root/'warmstart-report.json').write_text(json.dumps(
        report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='artifacts/rnad-warmstart')
    args=parser.parse_args()
    print(json.dumps(run(out=args.out),indent=2))
