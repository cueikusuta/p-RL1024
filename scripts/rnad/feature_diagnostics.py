"""Privacy-preserving phase action-feature distinguishability diagnostic.

No game internals are reconstructed. Operates exclusively on already
authority-accepted, seat-local, versioned teacher example vectors.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from scripts.rnad.train import current_git_sha,frozen_engine_sha
from scripts.rnad.neural import THREE_PHASE_FEATURE_SCHEMA_V2
from scripts.rnad.train_teacher import AQ_WEIGHTS_SHA256

PHASES=('draft','selection','battle')
TEACHER_SCHEMA='rnad-aq-public-belief-three-phase-teacher-v1'


def analyze(traces):
    stats={phase:{'decisions':0,'legal_candidates':0,
                  'duplicate_candidate_vectors':0,
                  'chosen_has_earlier_identical_vector':0,
                  'chosen_tied_for_identical_vector':0}
           for phase in PHASES}
    if not isinstance(traces,list) or not traces:
        raise ValueError('No trusted complete teacher Matches')
    seen_seeds=set()
    for trace in traces:
        if (trace.get('schema')!=TEACHER_SCHEMA or
            trace.get('feature_schema')!=THREE_PHASE_FEATURE_SCHEMA_V2 or
            trace.get('teacher_model_sha256')!=AQ_WEIGHTS_SHA256 or
            trace.get('source_sha')!=current_git_sha() or
            trace.get('engine_sha')!=frozen_engine_sha() or
            trace.get('completed') is not True or
            trace.get('seed') in seen_seeds or
            not trace.get('examples') or
            trace.get('score',[0,0])[trace['winner']]!=4):
            raise ValueError('Untrusted or duplicate teacher trajectory')
        seen_seeds.add(trace['seed'])
        for event in trace['examples']:
            phase=event.get('phase')
            actions=event.get('actions')
            choice=event.get('choice')
            state=event.get('state')
            if (phase not in stats or event.get('actor') not in (0,1) or
                not isinstance(state,list) or len(state)!=32 or
                not isinstance(actions,list) or not actions or
                type(choice) is not int or not 0<=choice<len(actions) or
                any(not isinstance(row,list) or len(row)!=16 or
                    any(type(v) not in (int,float) or not math.isfinite(v)
                        for v in row) for row in actions) or
                any(type(v) not in (int,float) or not math.isfinite(v)
                    for v in state)):
                raise ValueError('Invalid projected actor/action feature tuple')
            vectors=[tuple(row) for row in actions]
            unique=len(set(vectors))
            selected=vectors[choice]
            s=stats[phase]
            s['decisions']+=1
            s['legal_candidates']+=len(actions)
            s['duplicate_candidate_vectors']+=len(actions)-unique
            s['chosen_tied_for_identical_vector']+=int(vectors.count(selected)>1)
            s['chosen_has_earlier_identical_vector']+=int(
                selected in vectors[:choice])
    if any(x['decisions']==0 for x in stats.values()):
        raise ValueError('Missing accepted phase in feature diagnostic')
    for x in stats.values():
        x['fraction_candidate_vectors_duplicated']=(
            x['duplicate_candidate_vectors']/x['legal_candidates'])
        x['fraction_choices_with_an_earlier_identical_vector']=(
            x['chosen_has_earlier_identical_vector']/x['decisions'])
        x['fraction_choices_tied_on_identical_vector']=(
            x['chosen_tied_for_identical_vector']/x['decisions'])
    return stats


def run(root,seed,matches,out):
    if (type(seed) is not int or type(matches) is not int or
        not 1<=matches<=256 or not 1<=seed<=0xffffffff-matches):
        raise ValueError('Invalid bounded feature diagnostic seed block')
    root=Path(root)
    traces=[]
    for n in range(seed,seed+matches):
        path=root/f'teacher-{n}.json'
        if not path.is_file():
            raise ValueError('Missing required accepted source teacher Match')
        trace=json.loads(path.read_text())
        if trace.get('seed')!=n:
            raise ValueError('Nonconsecutive accepted source Match seed')
        traces.append(trace)
    stats=analyze(traces)
    report={'schema':'rnad-003-v2-action-feature-collision-study-v1',
            'source_sha':current_git_sha(),
            'engine_sha':frozen_engine_sha(),
            'teacher_weights_sha256':AQ_WEIGHTS_SHA256,
            'feature_schema':THREE_PHASE_FEATURE_SCHEMA_V2,
            'seed':seed,'completed_matches':matches,
            'stats':stats,'model_improvement_confirmed':False,
            'is_champion':False}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--matches',type=int,required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    print(json.dumps(run(a.root,a.seed,a.matches,a.out),flush=True))
