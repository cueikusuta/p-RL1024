"""Audit accepted public AQ phase utility labels and feature aliasing.

Read-only on frozen teacher traces; no new games, model updates or confidence
claims. Quantifies score ties and unavoidable action-feature ambiguity.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_TEACHER = 'rnad-aq-public-strategy-selection-v1'
EXPECTED_SHA = '5ccc11a3362c7f5a8874847e3afc1f19c539d4ef258bd82704f53d8a1db1946e'


def analyze_events(events):
    output = defaultdict(list)
    for event in events:
        phase = event.get('phase')
        if phase == 'battle':
            continue
        if phase not in ('draft', 'selection'):
            raise ValueError('Unexpected phase in accepted teacher trace')
        scores = event.get('candidate_utilities')
        features = event.get('actions')
        choice = event.get('choice')
        if (not isinstance(scores, list) or not scores or len(scores)>60000 or
                not isinstance(features,list) or len(features)!=len(scores) or
                not isinstance(choice,int) or isinstance(choice,bool) or
                not 0<=choice<len(scores) or
                any(type(s) not in (int,float) or not math.isfinite(s) for s in scores)):
            raise ValueError('Invalid public utility label or candidate list')
        if (any(not isinstance(a,list) or len(a)!=32 or
                any(type(x) not in (float,int) or not math.isfinite(x) for x in a)
                for a in features)):
            raise ValueError('Invalid v4 action feature vector')
        best = max(scores)
        worst = min(scores)
        if best - scores[choice] > 1e-5:
            raise ValueError('Accepted teacher choice does not maximize utility')
        ranked = sorted(scores,reverse=True)
        k = min(12 if phase=='draft' else 10,len(scores))
        scale = max(1.,abs(best),abs(worst))
        tol = scale*1e-9
        threshold = ranked[k-1]
        tied_at_boundary = (k<len(scores) and
                            abs(threshold-ranked[k])<=tol)
        top_margin = ranked[0]-ranked[1] if len(scores)>1 else 0.
        grouped = {}
        for action,score in zip(features,scores):
            key = tuple(action)
            if key not in grouped:
                grouped[key]=[1,score,score]
            else:
                row=grouped[key]
                row[0]+=1
                row[1]=min(row[1],score)
                row[2]=max(row[2],score)
        alias_count = sum(row[0] for row in grouped.values() if row[0]>1)
        alias_score_conflict = sum(row[0] for row in grouped.values()
                if row[0]>1 and row[2]-row[1]>tol)
        output[phase].append({
            'candidates':len(scores),
            'top1_tied':int(top_margin<=tol),
            'topk_boundary_tied':int(tied_at_boundary),
            'topk_boundary_tie_size':sum(abs(v-threshold)<=tol for v in scores),
            'normalized_top1_margin':(top_margin/(best-worst)
                   if best-worst>tol else 0.),
            'collapsed_candidate_count':alias_count,
            'conflicting_score_candidate_count':alias_score_conflict,
            'aliased_teacher_choice':int(grouped[tuple(features[choice])][0]>1),
            'nonzero_action_utility_range':int(any(
                row[2]-row[1]>tol for row in grouped.values())),
            'max_same_feature_utility_gap':max((
                row[2]-row[1] for row in grouped.values()),default=0.),
        })
    if any(not output.get(phase) for phase in ('draft','selection')):
        raise ValueError('Missing accepted draft or selection utility evidence')
    summary={}
    for phase in ('draft','selection'):
        rows=output[phase]
        average=lambda k:statistics.fmean(row[k] for row in rows)
        summary[phase]={
            'decisions':len(rows),
            'mean_candidates':average('candidates'),
            'top1_tied_fraction':average('top1_tied'),
            'topk_boundary_tie_fraction':average('topk_boundary_tied'),
            'mean_boundary_tie_size':average('topk_boundary_tie_size'),
            'mean_normalized_top1_margin':average('normalized_top1_margin'),
            'aliased_teacher_choice_fraction':average('aliased_teacher_choice'),
            'feature_alias_decision_fraction':average('nonzero_action_utility_range'),
            'mean_aliased_candidate_fraction':statistics.fmean(
                x['collapsed_candidate_count']/x['candidates'] for x in rows),
            'mean_different_utility_aliased_fraction':statistics.fmean(
                x['conflicting_score_candidate_count']/x['candidates'] for x in rows),
            'maximum_utility_difference_at_equal_feature':
                max(x['max_same_feature_utility_gap'] for x in rows),
        }
    return summary


def audit_trace_files(paths,expected_matches=32):
    files=list(paths)
    if len(files)!=expected_matches or not files:
        raise ValueError('Expected complete immutable accepted teacher Match count')
    source_engine=[None,None]
    seeds=set()

    def accepted_events():
        for path in files:
            # Stream Match by Match: the draft action tensors can be large.
            data=json.loads(Path(path).read_text())
            if (data.get('teacher')!=EXPECTED_TEACHER or
                data.get('teacher_model_sha256')!=EXPECTED_SHA or
                data.get('utility_labels')!='aq-public-phase-rank-v1' or
                data.get('feature_schema')!='rnad-three-phase-features-v4' or
                data.get('completed') is not True or
                data.get('score',[0,0])[data.get('winner',-1)]!=4 or
                not isinstance(data.get('examples'),list) or
                type(data.get('seed')) is not int):
                raise ValueError('Invalid frozen teacher Match receipt')
            if data['seed'] in seeds:
                raise ValueError('Duplicate teacher training seed')
            seeds.add(data['seed'])
            if source_engine[0] is None:
                source_engine[:]=[data['source_sha'],data['engine_sha']]
            elif ([data.get('source_sha'),data.get('engine_sha')] != source_engine):
                raise ValueError('Mixed source/engine frozen teacher traces')
            yield from data['examples']

    summary=analyze_events(accepted_events())
    if len(seeds)!=expected_matches:
        raise ValueError('Incomplete independent teacher seed block')
    return {
        'schema':'rnad-public-aq-utility-signal-audit-v1',
        'source_sha':source_engine[0],'engine_sha':source_engine[1],
        'teacher_model_sha256':EXPECTED_SHA,
        'accepted_matches':len(files),'independent_seeds':len(seeds),
        'phase_metrics':summary,
        'exploratory_diagnostic_only':True,
        'strength_established':False,'champion_promoted':False,
    }



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--matches',type=int,default=32)
    parser.add_argument('--out',required=True)
    a=parser.parse_args()
    if not 1<=a.matches<=256:
        parser.error('Invalid bounded teacher count')
    result=audit_trace_files(Path(a.root).rglob('teacher-*.json'),a.matches)
    Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))
