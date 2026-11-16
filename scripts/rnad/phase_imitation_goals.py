"""Intermediate draft/selection imitation screens on paired held-out reports.

This is NOT a game-strength or Champion gate. The original AQ opponent,
accepted four-win Match evaluation and independent holdout remain mandatory.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

# Provisional engineering thresholds, not proven win-rate predictors.
GOALS = {
    'draft': {'top1_accuracy': .08, 'top10_accuracy': .30,
              'mean_normalized_cross_entropy': .78},
    'selection': {'top1_accuracy': .15, 'top10_accuracy': .65,
                  'mean_normalized_cross_entropy': .75},
}
KNOWN_SCHEMAS = {
    'rnad-aq-imitator-unseen-match-eval-v1',
    'rnad-aq-imitator-strategy-selection-eval-v1',
}


def validate_report(report, min_examples=256):
    if (not isinstance(report, dict) or report.get('schema') not in KNOWN_SCHEMAS
            or report.get('is_champion') is not False
            or type(report.get('seed')) is not int
            or type(report.get('matches')) is not int
            or report['matches'] < 24
            or not isinstance(report.get('phases'), dict)
            or not isinstance(report.get('student_model_sha256'), str)
            or len(report['student_model_sha256']) != 64
            or not isinstance(report.get('teacher_model_sha256'), str)
            or len(report['teacher_model_sha256']) != 64
            or not isinstance(report.get('engine_sha'), str)
            or not isinstance(report.get('source_sha'), str)):
        raise ValueError('Invalid or too small authority-accepted imitation report')
    teacher_id = report.get('teacher_id', 'rnad-aq-public-belief-v2')
    if teacher_id not in ('rnad-aq-public-belief-v2',
                          'rnad-aq-public-strategy-selection-v1'):
        raise ValueError('Unrecognized frozen teacher version')
    for phase in GOALS:
        metrics = report['phases'].get(phase)
        if not isinstance(metrics, dict) or type(metrics.get('examples')) is not int:
            raise ValueError('Missing phase examples')
        if metrics['examples'] < min_examples:
            raise ValueError('Too few held-out phase decisions')
        for metric in GOALS[phase]:
            value = metrics.get(metric)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError('Nonfinite or missing phase metric')
        if not (0 <= metrics['top1_accuracy'] <= metrics['top10_accuracy'] <= 1
                and metrics['mean_normalized_cross_entropy'] >= 0):
            raise ValueError('Invalid phase metric range')
    return teacher_id


def compare_heldout_imitation(base, candidate, *, min_examples=256,
                              allow_feature_upgrade=False):
    old_teacher = validate_report(base, min_examples)
    new_teacher = validate_report(candidate, min_examples)
    if (old_teacher != new_teacher or
        base['schema'] != candidate['schema'] or
        (not allow_feature_upgrade and
         base.get('feature_schema') != candidate.get('feature_schema')) or
        any(base.get(key) != candidate.get(key) for key in (
            'source_sha', 'engine_sha', 'teacher_model_sha256',
            'seed', 'matches')) or
        base['student_model_sha256'] == candidate['student_model_sha256']):
        raise ValueError('Require different students, identical teacher, feature, seed and source')
    result = {'schema':'rnad-phase-imitation-screen-v1',
              'teacher_id':old_teacher,
              'seed':base['seed'], 'matches':base['matches'],
              'min_examples_per_phase':min_examples,
              'feature_upgrade':bool(base.get('feature_schema')!=candidate.get('feature_schema')),
              'baseline_feature_schema':base.get('feature_schema'),
              'candidate_feature_schema':candidate.get('feature_schema'),
              'is_champion':False,
              'strength_established':False,
              'phases':{}}
    for phase, goals in GOALS.items():
        b, c = base['phases'][phase], candidate['phases'][phase]
        if b['examples'] != c['examples']:
            raise ValueError('Unpaired held-out phase example count')
        pass_values = (
            c['top1_accuracy'] >= goals['top1_accuracy'] and
            c['top10_accuracy'] >= goals['top10_accuracy'] and
            c['mean_normalized_cross_entropy'] <= goals['mean_normalized_cross_entropy']
        )
        improved = (
            c['top1_accuracy'] > b['top1_accuracy'] and
            c['top10_accuracy'] > b['top10_accuracy'] and
            c['mean_normalized_cross_entropy'] < b['mean_normalized_cross_entropy']
        )
        result['phases'][phase] = {
            'examples':c['examples'],
            'goal_screen_pass':bool(pass_values and improved),
            'target_reached':bool(pass_values),
            'improved_all_three_metrics':bool(improved),
            'top1':c['top1_accuracy'], 'top10':c['top10_accuracy'],
            'normalized_ce':c['mean_normalized_cross_entropy'],
            'delta_top1':c['top1_accuracy']-b['top1_accuracy'],
            'delta_top10':c['top10_accuracy']-b['top10_accuracy'],
            'delta_normalized_ce':c['mean_normalized_cross_entropy']-
                b['mean_normalized_cross_entropy'],
        }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--min-examples', type=int, default=256)
    parser.add_argument('--allow-feature-upgrade', action='store_true')
    args = parser.parse_args()
    if not 16 <= args.min_examples <= 100000:
        parser.error('Invalid minimum phase sample size')
    report = compare_heldout_imitation(
        json.loads(Path(args.baseline).read_text()),
        json.loads(Path(args.candidate).read_text()),
        min_examples=args.min_examples,
        allow_feature_upgrade=args.allow_feature_upgrade)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
