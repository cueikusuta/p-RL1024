"""Bounded 100-variant RNAD-003 policy exploration, not a Champion gate.

Mutation is a frozen-policy screening heuristic; the four survivors receive
separate, NEW full-state 16-Match R-NaD training runs (not optimizer resume).
Only previously unseen seed blocks are used for final assessment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

ACTION_COLUMNS = (5, 6, 13, 14, 15, 0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12)
STATE_COLUMNS = (23, 24, 25, 26, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
TRAIN_SEED = 1300001
SCREEN_SEED = 1301001
SELECT_SEED = 1302001
HOLDOUT_BASIC_SEED = 1303001
HOLDOUT_DIRECT_SEED = 1304001
COUNT = 100
ORIGIN_SOURCE_SHA = 'f08a9587d6a7033546eab00ffa6479399ca71c2f'
ORIGIN_RUN = 43747882146
FEATURE_SCHEMA = 'rnad-three-phase-features-v2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variants(count=COUNT):
    if count != 100:
        raise ValueError('Population contract requires exactly 100 candidates')
    result = [{'id': 'c000', 'family': 'unchanged-control'}]
    for i in range(1, count):
        if i <= 32:
            result.append({'id': f'c{i:03d}', 'family': 'guided-action',
                           'column': ACTION_COLUMNS[(i-1)//2],
                           'factor': 1.15 if i % 2 else 0.85})
        elif i <= 64:
            result.append({'id': f'c{i:03d}', 'family': 'guided-state',
                           'column': STATE_COLUMNS[(i-33)//2],
                           'factor': 1.15 if i % 2 else 0.85})
        else:
            result.append({'id': f'c{i:03d}', 'family': 'random-policy',
                           'target': ('action', 'head', 'state')[(i-65) % 3],
                           'relative_noise': (0.02, 0.05, 0.1, 0.2)[((i-65)//3) % 4],
                           'seed': 20261116 + i})
    if len({s['id'] for s in result}) != count:
        raise AssertionError('Duplicate candidate')
    return result


def verified_weights(payload):
    if (payload.get('schema') != 'rnad-small-gru-v1' or
        payload.get('feature_schema') != FEATURE_SCHEMA or
        payload.get('state_features') != 32 or
        payload.get('action_features') != 16 or
        payload.get('hidden') != 128):
        raise ValueError('Population requires immutable v2 GRU128 model contract')
    expected = {'policy_hidden.weight': [128, 144],
                'policy_out.weight': [1, 128],
                'state_proj.weight': [128, 32]}
    for key, shape in expected.items():
        item = payload.get('weights', {}).get(key)
        if (not isinstance(item, dict) or item.get('shape') != shape or
            not isinstance(item.get('values'), list) or
            len(item['values']) != math.prod(shape) or
            not all(type(v) in (float, int) and math.isfinite(v)
                    for v in item['values'])):
            raise ValueError('Malformed v2 model parameter: ' + key)


def mutate(base, spec):
    verified_weights(base)
    model = copy.deepcopy(base)
    family = spec['family']
    if family == 'unchanged-control':
        return model
    if family in ('guided-action', 'guided-state'):
        key = 'policy_hidden.weight' if family == 'guided-action' else 'state_proj.weight'
        width = 144 if family == 'guided-action' else 32
        col = (128 + spec['column']) if family == 'guided-action' else spec['column']
        values = model['weights'][key]['values']
        for row in range(128):
            index = row*width + col
            values[index] *= spec['factor']
    elif family == 'random-policy':
        key = {'action': 'policy_hidden.weight', 'head': 'policy_out.weight',
               'state': 'state_proj.weight'}[spec['target']]
        values = model['weights'][key]['values']
        indices = ([row*144+col for row in range(128) for col in range(128,144)]
                   if spec['target'] == 'action' else
                   list(range(len(values))))
        rms = math.sqrt(sum(values[j]**2 for j in indices)/len(indices))
        if not rms > 0:
            raise ValueError('Degenerate mutation tensor')
        rng = random.Random(spec['seed'])
        for j in indices:
            values[j] += rng.gauss(0, rms*spec['relative_noise'])
    else:
        raise ValueError('Unknown mutation family')
    verified_weights(model)
    return model


def write_model(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(payload, f, separators=(',', ':'), allow_nan=False)
        f.write('\n')
    return digest(path)


def validate_report(report, *, model, seed, pairs, direct=False, opponent=None):
    if (report.get('source_sha') != source_sha() or
        report.get('engine_sha') != engine_sha() or
        report.get('inference_selection_mode') != 'greedy' or
        report.get('pairs') != pairs or report.get('matches') != 2*pairs or
        report.get('wins') != sum(int(row.get('won') is True)
                                  for row in report.get('rows', [])) or
        len(report.get('rows', [])) != 2*pairs or
        type(report.get('wins')) is not int or
        report['wins'] < 0 or report['wins'] > 2*pairs):
        raise ValueError('Invalid or stale paired Match report')
    if direct:
        if (report.get('schema') != 'rnad-direct-neural-paired-match-v1' or
            report.get('rnad_model_sha256') != digest(model) or
            report.get('teacher_model_sha256') != digest(opponent) or
            report.get('model_feature_schema') != FEATURE_SCHEMA or
            report.get('is_champion') is not False):
            raise ValueError('Invalid direct comparison provenance')
        seat_key = 'rnadSeat'
    else:
        if (report.get('schema') != 'rnad-three-phase-vs-basic-paired-v1' or
            report.get('model_sha256') != digest(model) or
            report.get('opponent_id') != 'basic-all-phases' or
            report.get('champion_promoted') is not False):
            raise ValueError('Invalid Basic comparison provenance')
        seat_key = 'candidateSeat'
    expected = {(n, seat) for n in range(seed, seed+pairs) for seat in (0,1)}
    observed = set()
    for row in report['rows']:
        key = (row.get('seed'), row.get(seat_key))
        score = row.get('score')
        if (key not in expected or key in observed or
            type(row.get('won')) is not bool or
            not isinstance(score, list) or len(score) != 2 or
            not all(type(v) is int and 0 <= v <= 4 for v in score) or
            max(score) != 4 or
            score[key[1] if row['won'] else 1-key[1]] != 4 or
            not 4 <= row.get('rounds', -1) <= 64):
            raise ValueError('Invalid, reused or inconsistent Match outcome')
        observed.add(key)
    if observed != expected:
        raise ValueError('Missing seed/seat in paired Match')


def source_sha():
    from scripts.rnad.train import current_git_sha
    return current_git_sha()


def engine_sha():
    from scripts.rnad.train import frozen_engine_sha
    return frozen_engine_sha()


def evaluate(model, seed, pairs, output, *, direct=False, opponent=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['node', 'scripts/rnad/evaluate-direct.mjs' if direct
               else 'scripts/rnad/evaluate-neural-basic.mjs']
    if direct:
        command += ['--teacher', str(opponent), '--rnad', str(model)]
    else:
        command += ['--model', str(model)]
    command += ['--seed', str(seed), '--pairs', str(pairs),
                '--selection-mode', 'greedy', '--out', str(output)]
    subprocess.run(command, check=True, timeout=1800)
    report = json.loads(output.read_text())
    validate_report(report, model=model, seed=seed, pairs=pairs,
                    direct=direct, opponent=opponent)
    return report


def verify_origin(root):
    from scripts.rnad.neural import RNADNetwork, load_exported_json
    from scripts.rnad.train import state_hash
    root = Path(root)
    progress = json.loads((root/'generations/progress.json').read_text())
    snapshots = progress.get('snapshots', [])
    model = root/'generations/model-u128.json'
    if (progress.get('schema') != 'rnad-003-v2-verified-128-generation-v1' or
        progress.get('source_sha') != ORIGIN_SOURCE_SHA or
        progress.get('origin_run_id') != 43746142851 or
        progress.get('is_champion') is not False or
        [row.get('generation') for row in snapshots] != [64,128]):
        raise ValueError('Untrusted origin research checkpoint provenance')
    checkpoint_sha = snapshots[-1]['checkpoint_sha256']
    files = list((root/'generations').glob('checkpoint-u128-*.pt'))
    if len(files) != 1 or digest(files[0]) != checkpoint_sha:
        raise ValueError('Missing or changed original full-state checkpoint')
    network = RNADNetwork()
    load_exported_json(network, model, expected_feature_schema=FEATURE_SCHEMA)
    if state_hash(network) != snapshots[-1]['model_sha256']:
        raise ValueError('Origin model and checkpoint manifest disagree')
    return model, {'origin_run': ORIGIN_RUN, 'origin_source_sha': ORIGIN_SOURCE_SHA,
                   'origin_model_file_sha256': digest(model),
                   'origin_checkpoint_sha256': checkpoint_sha}


def research(origin, output):
    from scripts.rnad.train import train
    root = Path(output)
    if root.exists():
        raise ValueError('Refuse to overwrite or mix population trials')
    root.mkdir(parents=True)
    baseline, origin_provenance = verify_origin(origin)
    baseline_model = json.loads(baseline.read_text())
    specs = variants()
    if not (TRAIN_SEED+16 < SCREEN_SEED and SCREEN_SEED+2 < SELECT_SEED and
            SELECT_SEED+8 < HOLDOUT_BASIC_SEED and
            HOLDOUT_BASIC_SEED+32 < HOLDOUT_DIRECT_SEED):
        raise AssertionError('Overlapping pre-registered seed windows')
    entries = []
    for spec in specs:
        candidate = root/'models'/f"{spec['id']}.json"
        file_sha = write_model(candidate, mutate(baseline_model, spec))
        screening = evaluate(candidate, SCREEN_SEED, 2,
                             root/'screen'/f"{spec['id']}.json")
        entries.append({'spec': spec, 'initial_model_sha256': file_sha,
                        'screen_wins_4': screening['wins']})
        candidate.unlink()  # keep generation deterministic, avoid 100 large artifacts
    ordered = sorted(entries[1:], key=lambda x:(-x['screen_wins_4'], x['spec']['id']))
    top16 = ordered[:16]
    for entry in top16:
        spec = entry['spec']
        model = root/'models'/f"{spec['id']}.json"
        if write_model(model, mutate(baseline_model, spec)) != entry['initial_model_sha256']:
            raise ValueError('Mutation regenerated different model bytes')
        report = evaluate(model, SELECT_SEED, 8,
                          root/'select'/f"{spec['id']}.json")
        entry['select_wins_16'] = report['wins']
    survivors = sorted(top16, key=lambda x:(-x['select_wins_16'],
                                            -x['screen_wins_4'],
                                            x['spec']['id']))[:4]
    baseline_basic = evaluate(baseline, HOLDOUT_BASIC_SEED, 32,
                              root/'holdout/baseline-basic.json')
    final = []
    for entry in survivors:
        spec = entry['spec']
        model = root/'models'/f"{spec['id']}.json"
        folder = root/'training'/spec['id']
        training = train(seed=TRAIN_SEED, matches=16, out=str(folder),
                         initial_model=str(model), eta=.2, magnet_period=4,
                         three_phase=True, features_v2=True)
        index = json.loads((folder/'checkpoints/latest.json').read_text())
        pt = folder/'checkpoints'/index['checkpoint_file']
        if (training.get('source_sha') != source_sha() or
            training.get('initialization_sha256') != digest(model) or
            training.get('completed_matches') != 16 or
            training.get('updates') != 16 or
            training.get('model_id') != 'RNAD-003-3phase-v2' or
            training.get('is_champion') is not False or
            len(training.get('results', [])) != 16 or
            index['metadata'].get('initial_seed') != TRAIN_SEED or
            index['metadata'].get('completed_matches') != 16 or
            index['checkpoint_sha256'] != training.get('last_match_checkpoint_sha') or
            digest(pt) != index['checkpoint_sha256']):
            raise ValueError('Invalid actual R-NaD population learner')
        trained = folder/'final-model.json'
        basic = evaluate(trained, HOLDOUT_BASIC_SEED, 32,
                         root/'holdout'/f"{spec['id']}-basic.json")
        direct = evaluate(trained, HOLDOUT_DIRECT_SEED, 32,
                          root/'holdout'/f"{spec['id']}-direct.json",
                          direct=True, opponent=baseline)
        final.append({'id':spec['id'], 'family':spec['family'],
                      'initial_model_sha256':digest(model),
                      'trained_model_sha256':digest(trained),
                      'checkpoint_sha256':index['checkpoint_sha256'],
                      'basic_wins_64':basic['wins'],
                      'baseline_basic_wins_64':baseline_basic['wins'],
                      'direct_wins_64':direct['wins'],
                      'direct_paired_ci95':direct['paired_seed_hoeffding_95pct'],
                      'uncorrected_direct_ci_above_half':
                        direct['paired_seed_hoeffding_95pct'][0] > .5,
                      'statistical_improvement_confirmed':False,
                      'new_independent_holdout_required_for_multiplicity':True})
        print(json.dumps(final[-1],allow_nan=False),flush=True)
    report = {'schema':'rnad-003-population-100-screen-v1',
              **origin_provenance, 'source_sha':source_sha(),
              'engine_sha':engine_sha(), 'population':100,
              'stage1_pairs':2, 'stage2_pairs':8,
              'trained_finalists':4, 'matches_per_finalist':16,
              'holdout_pairs':32,
              'optimization_is_exact_checkpoint_continuation':False,
              'exploration_seed_is_a_champion_test':False,
              'champion_promoted':False,
              'stages':entries,'finalists':final}
    (root/'population-report.json').write_text(
        json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--origin')
    p.add_argument('--out',default='artifacts/rnad-population')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if args.plan_only:
        print(json.dumps({'count':100,'plan':variants(),
                          'is_champion':False},indent=2))
    else:
        if not args.origin: p.error('--origin is required for real research')
        result=research(args.origin,args.out)
        print(json.dumps({'research_complete':True,
                          'finalists':result['finalists'],
                          'champion_promoted':False}))
