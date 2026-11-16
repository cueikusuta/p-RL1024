"""Outcome-supervised first-selection ranking on real accepted 4-win Matches.

Exploratory ONLY. The candidate rank intervention changes downstream gameplay,
so this is an empirical outcome-based surrogate, not an exact game-theoretic Q
oracle and never an AQ/Champion promotion gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

SCHEMA = 'rnad-i5-v3-first-selection-outcome-probe-v1'
DIM = 96 + 48


def load_examples(path):
    report = json.loads(Path(path).read_text())
    if (report.get('schema') != SCHEMA or
        report.get('exploration_only') is not True or
        report.get('strength_established') is not False or
        report.get('champion_promoted') is not False or
        type(report.get('pairs')) is not int or
        type(report.get('ranks')) is not int or
        not 1 <= report['pairs'] <= 32 or
        not 2 <= report['ranks'] <= 12 or
        type(report.get('seed')) is not int or report['seed'] < 1 or
        not all(isinstance(report.get(x), str) and len(report[x]) == n
                for x, n in (('source_sha', 40), ('engine_sha', 40),
                             ('v3_model_sha256', 64), ('aq_base_model_sha256', 64))) or
        not isinstance(report.get('rows'), list) or
        len(report['rows']) != report['pairs'] * 2 or
        report.get('complete_matches') != report['pairs'] * 2 * report['ranks']):
        raise ValueError('Wrong or incomplete authority-accepted outcome report')
    instances = []
    keys = set()
    for row in report['rows']:
        if (not isinstance(row, dict) or
            type(row.get('seed')) is not int or
            row['seed'] not in range(report['seed'], report['seed'] + report['pairs']) or
            row.get('seat') not in (0, 1) or
            (row['seed'], row['seat']) in keys or
            not isinstance(row.get('scenarios'), list) or
            len(row['scenarios']) != report['ranks']):
            raise ValueError('Duplicate, incomplete or mismatched seat/seed')
        keys.add((row['seed'], row['seat']))
        same_sig = set()
        selections = set()
        samples = []
        for index, sc in enumerate(row['scenarios']):
            if (not isinstance(sc, dict) or sc.get('rank') != index or
                type(sc.get('won')) is not bool or
                type(sc.get('legal_count')) is not int or
                sc['legal_count'] != 60 or
                not isinstance(sc.get('selected'), str) or
                not isinstance(sc.get('signature'), str) or
                not isinstance(sc.get('features'), list) or
                not isinstance(sc.get('state_features'), list) or
                len(sc['features']) != 48 or len(sc['state_features']) != 96 or
                not isinstance(sc.get('score'), list) or
                len(sc['score']) != 2 or
                sorted(sc['score'], reverse=True)[0] != 4 or
                any(type(x) not in (int, float) or not math.isfinite(x)
                    for x in sc['features'] + sc['state_features'])):
                raise ValueError('Invalid authorized outcome candidate')
            same_sig.add(sc['signature'])
            selections.add(sc['selected'])
            samples.append((sc['state_features'] + sc['features'], float(sc['won'])))
        if len(same_sig) != 1 or len(selections) != report['ranks']:
            raise ValueError('Non-matched intervention or repeated candidate')
        instances.append((row['seed'], row['seat'], samples))
    if len(keys) != report['pairs'] * 2:
        raise ValueError('Missing seat-swapped seed cluster')
    return report, instances


def train_outcome_rank(train_file, heldout_file, out, *, epochs=80):
    if not 1 <= epochs <= 160:
        raise ValueError('Invalid bounded optimization budget')
    train_meta, train = load_examples(train_file)
    test_meta, test = load_examples(heldout_file)
    if (any(train_meta[x] != test_meta[x] for x in
            ('source_sha', 'engine_sha', 'v3_model_sha256',
             'aq_base_model_sha256', 'ranks')) or
        set(r[0] for r in train) & set(r[0] for r in test)):
        raise ValueError('Evaluation has overlapping seeds or different frozen sources')
    torch.manual_seed(20261117)
    torch.set_num_threads(1)
    net = torch.nn.Linear(DIM, 1)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.01)
    x = torch.tensor([features for _, _, cases in train
                      for features, _ in cases], dtype=torch.float32)
    y = torch.tensor([label for _, _, cases in train
                      for _, label in cases], dtype=torch.float32)
    for _ in range(epochs):
        logits = net(x).flatten()
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite outcome-learning objective')
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        choices = []
        for seed, seat, cases in test:
            inputs = torch.tensor([f for f, _ in cases], dtype=torch.float32)
            scores = net(inputs).flatten()
            best = int(torch.argmax(scores).item())
            choices.append({'seed': seed, 'seat': seat, 'predicted_rank': best,
                            'baseline_win': bool(cases[0][1]),
                            'predicted_win': bool(cases[best][1])})
        count = len(choices)
        base = sum(int(r['baseline_win']) for r in choices)
        selected = sum(int(r['predicted_win']) for r in choices)
        diff = (selected - base) / count
        half = math.sqrt(2 * math.log(40) / test_meta['pairs'])
        # Frozen training seed and data identities are part of the model artifact.
        outdir = Path(out)
        outdir.mkdir(parents=True, exist_ok=False)
        statefile = outdir / 'outcome-linear-model.pt'
        torch.save({
            'state_dict':net.state_dict(),
            'feature_schema':'rnad-three-phase-features-v5',
            'training_outcome_source_sha':train_meta['source_sha'],
            'train_seed':train_meta['seed'],
            'train_pairs':train_meta['pairs']
        }, statefile)
        report = {
            'schema':'rnad-i5-selection-outcome-linear-screen-v1',
            'train_seed':train_meta['seed'], 'train_pairs':train_meta['pairs'],
            'heldout_seed':test_meta['seed'],'heldout_pairs':test_meta['pairs'],
            'frozen_v3_sha256':train_meta['v3_model_sha256'],
            'aq_base_sha256':train_meta['aq_base_model_sha256'],
            'source_sha':train_meta['source_sha'],
            'epochs':epochs,
            'model_file_sha256':hashlib.sha256(statefile.read_bytes()).hexdigest(),
            'baseline_wins':base,'predicted_wins':selected,
            'paired_delta':diff,
            'conservative_95pct_delta_bounds':[
                max(-1, diff-half),min(1, diff+half)],
            'selection_policy_deployed':False,
            'strength_established':False,
            'champion_promoted':False,
            'choices':choices
        }
        (outdir / 'report.json').write_text(json.dumps(
            report, indent=2, allow_nan=False) + '\n')
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--heldout', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(train_outcome_rank(args.train, args.heldout, args.out,
                                        epochs=args.epochs), indent=2))
