"""Small-sample v4-v5 imitation diagnostics: NEVER an adoption gate.

The formal phase_imitation_goals validator deliberately requires 24+ completed
teacher Matches / 256+ phase decisions; do not bypass it for Champion evidence.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

FIELDS = ('source_sha', 'engine_sha', 'teacher_model_sha256',
          'teacher_id', 'utility_labels', 'seed', 'matches', 'schema')
METRICS = ('top1_accuracy', 'top10_accuracy',
           'mean_normalized_cross_entropy',
           'mean_normalized_utility_regret')


def compare_pilot(base, candidate):
    if (not isinstance(base, dict) or not isinstance(candidate, dict) or
        base.get('schema') != 'rnad-aq-imitator-strategy-selection-eval-v1' or
        candidate.get('schema') != base.get('schema') or
        base.get('is_champion') is not False or
        candidate.get('is_champion') is not False or
        any(base.get(key) != candidate.get(key) for key in FIELDS) or
        base.get('teacher_id') != 'rnad-aq-public-strategy-selection-v1' or
        base.get('utility_labels') != 'aq-public-phase-rank-v1' or
        base.get('feature_schema') != 'rnad-three-phase-features-v4' or
        candidate.get('feature_schema') != 'rnad-three-phase-features-v5' or
        not isinstance(base.get('student_model_sha256'), str) or
        not isinstance(candidate.get('student_model_sha256'), str) or
        base.get('student_model_sha256') == candidate.get('student_model_sha256') or
        type(base.get('matches')) is not int or not 4 <= base['matches'] < 24):
        raise ValueError('Not identical small-budget accepted AQ v4/v5 evaluation')
    result = {'schema':'rnad-i5-v5-small-sample-imitation-pilot-v1',
              'seed':base['seed'],'matches':base['matches'],
              'teacher_id':base['teacher_id'],
              'source_sha':base['source_sha'],
              'baseline_feature_schema':base['feature_schema'],
              'candidate_feature_schema':candidate['feature_schema'],
              'formal_imitation_goal_gate_not_applied':True,
              'strength_established':False,'champion_promoted':False,
              'phases':{}}
    for phase in ('draft','selection'):
        a=base.get('phases',{}).get(phase)
        b=candidate.get('phases',{}).get(phase)
        if (not isinstance(a,dict) or not isinstance(b,dict) or
            a.get('examples') != b.get('examples') or
            type(a.get('examples')) is not int or a['examples'] < 8):
            raise ValueError('Missing matched heldout phase decisions')
        for key in METRICS:
            if (type(a.get(key)) not in (int,float) or
                type(b.get(key)) not in (int,float) or
                not math.isfinite(a[key]) or not math.isfinite(b[key])):
                raise ValueError('Missing finite imitation metric')
        result['phases'][phase] = {'examples':a['examples'],
                                  'metrics':{
                                      k:{'v4':a[k],'v5':b[k],
                                         'delta_v5_minus_v4':b[k]-a[k]}
                                      for k in METRICS}}
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    report=compare_pilot(json.loads(Path(args.baseline).read_text()),
                         json.loads(Path(args.candidate).read_text()))
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
