"""RNAD-002 optional Basic/Stage0 3-phase imitation -> NeuRD/V-trace research.

True Match-authority-accepted data, separate from RNAD-001 and external teachers.
This bounded experiment compares independent actor initialization, teacher
imitation, and 16 complete self-play R-NaD updates. NOT Champion promotion.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from scripts.rnad.train_teacher import train as imitate
from scripts.rnad.train import train,current_git_sha,frozen_engine_sha
from scripts.rnad.neural import THREE_PHASE_FEATURE_SCHEMA


def run(*,out='artifacts/rnad-three-phase-teacher'):
    root=Path(out)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Refuse to reuse a completed three-phase experiment')
    root.mkdir(parents=True,exist_ok=True)
    teacher=imitate(seed=975321,matches=8,epochs=2,three_phase=True,
                    out=str(root/'teacher'))
    if (teacher.get('schema')!='rnad-three-phase-imitation-report-v1' or
        teacher.get('feature_schema')!=THREE_PHASE_FEATURE_SCHEMA or
        teacher.get('teacher_completed_matches')!=8 or
        teacher.get('teacher_updates')!=16 or
        teacher.get('rnad_updates')!=0 or
        teacher.get('source_sha')!=current_git_sha() or
        teacher.get('engine_sha')!=frozen_engine_sha()):
        raise ValueError('Unexpected authority-accepted three-phase teacher')
    result=train(seed=975361,matches=16,three_phase=True,
                 initial_model=str(root/'teacher'/'trained-model.json'),
                 out=str(root/'selfplay'))
    checkpoint=json.loads(
        (root/'selfplay'/'checkpoints'/'latest.json').read_text())
    md=checkpoint['metadata']
    accepted=sum(x['accepted_decisions'] for x in result['results'])
    if (result.get('schema')!='rnad-three-phase-training-report-v1' or
        result.get('model_id')!='RNAD-002-3phase' or
        result.get('completed_matches')!=16 or
        result.get('updates')!=16 or
        result.get('feature_schema')!=THREE_PHASE_FEATURE_SCHEMA or
        len(result.get('results',[]))!=16 or
        not all(x['accepted_decisions']>0 and
            all(math.isfinite(x[m]) for m in ('loss','critic_loss','policy_loss'))
            for x in result['results']) or
        md.get('completed_matches')!=16 or md.get('updates')!=16 or
        md.get('accepted_decisions')!=accepted or
        md.get('initial_seed')!=975361 or
        md.get('model_id')!='RNAD-002-3phase' or
        md.get('observation_schema_version')!=THREE_PHASE_FEATURE_SCHEMA or
        len(set(md.get('completed_match_ids',[])))!=16 or
        result.get('last_match_checkpoint_sha')!=checkpoint.get('checkpoint_sha256') or
        result['model_sha256']==teacher['trained_model_sha256']):
        raise ValueError('Incomplete warm-started accepted R-NaD progress')
    report={
        'schema':'rnad-three-phase-basic-teacher-warmstart-study-v1',
        'source_sha':current_git_sha(),'engine_sha':frozen_engine_sha(),
        'teacher':'Basic draft/selection + Stage0 battle',
        'teacher_seed':975321,'teacher_complete_matches':8,
        'teacher_updates':16,
        'teacher_model_sha256':teacher['trained_model_sha256'],
        'rnad_selfplay_seed':975361,'rnad_completed_matches':16,
        'rnad_optimizer_updates':16,'rnad_accepted_actions':accepted,
        'rnad_model_sha256':result['model_sha256'],
        'rnad_checkpoint_sha256':checkpoint['checkpoint_sha256'],
        'is_champion':False,
        'note':'Learning success is not evidence of strategic superiority'
    }
    (root/'study-report.json').write_text(
        json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='artifacts/rnad-three-phase-teacher')
    args=parser.parse_args()
    print(json.dumps(run(out=args.out),indent=2))
