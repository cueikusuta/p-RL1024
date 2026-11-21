"""Fail-closed tests for phase-focused AQ imitation research, not Champion."""
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.rnad.train_teacher import (
    collect, phase_balanced_mean_loss, train, initialize_teacher_model,
)
from scripts.rnad.neural import (
    RNADNetwork, export_json, THREE_PHASE_FEATURE_SCHEMA_V2,
)
from scripts.rnad.train import state_hash


class PhaseFocusLossTest(unittest.TestCase):
    def test_weighted_and_equal_gradient_are_distinct(self):
        draft=torch.tensor(1.0,requires_grad=True)
        selection=torch.tensor(2.0,requires_grad=True)
        battle=torch.tensor(3.0,requires_grad=True)
        rows=[('draft',draft),('selection',selection)]+[('battle',battle)]*10
        weights={'draft':2.,'selection':2.,'battle':1.}
        loss=phase_balanced_mean_loss(rows,enabled=True,weights=weights)
        self.assertAlmostEqual(float(loss),1.8)
        loss.backward()
        self.assertAlmostEqual(float(draft.grad),.4)
        self.assertAlmostEqual(float(selection.grad),.4)
        self.assertAlmostEqual(float(battle.grad),.2)
        self.assertAlmostEqual(float(phase_balanced_mean_loss(rows,enabled=True)),2.)
        self.assertAlmostEqual(float(phase_balanced_mean_loss(rows,enabled=False)),33/12)

    def test_selection_focused_objective_preserves_all_three_phase_gradients(self):
        draft=torch.tensor(1.0,requires_grad=True)
        selection=torch.tensor(2.0,requires_grad=True)
        battle=torch.tensor(5.0,requires_grad=True)
        rows=[('draft',draft),('selection',selection)]+[('battle',battle)]*10
        weights={'draft':1.,'selection':4.,'battle':1.}
        loss=phase_balanced_mean_loss(rows,enabled=True,weights=weights)
        self.assertAlmostEqual(float(loss),(1+4*2+5)/6,places=6)
        self.assertNotAlmostEqual(float(loss),
            float(phase_balanced_mean_loss(rows,enabled=True)))
        loss.backward()
        self.assertAlmostEqual(float(draft.grad),1/6)
        self.assertAlmostEqual(float(selection.grad),4/6)
        self.assertAlmostEqual(float(battle.grad),1/6)

    def test_invalid_weights_fail_closed(self):
        x=torch.tensor(.4,requires_grad=True)
        rows=[('draft',x),('selection',x),('battle',x)]
        for weights in ({'draft':1.,'selection':1.},
                        {'draft':2.,'selection':0.,'battle':1.},
                        {'draft':float('nan'),'selection':1.,'battle':1.},
                        {'draft':2.,'selection':1.,'battle':float('inf')}):
            with self.subTest(weights=weights),self.assertRaises(ValueError):
                phase_balanced_mean_loss(rows,enabled=True,weights=weights)
        with self.assertRaises(ValueError):
            phase_balanced_mean_loss(rows,enabled=False,weights={
                'draft':2.,'selection':2.,'battle':1.})
        with self.assertRaises(ValueError):
            phase_balanced_mean_loss(rows[:-1],enabled=True,weights={
                'draft':2.,'selection':2.,'battle':1.})

    def test_frozen_v2_weights_loaded_without_mutating_saved_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'frozen-v2.json'
            torch.manual_seed(20)
            frozen=RNADNetwork()
            expected=state_hash(frozen)
            export_json(frozen,original,
                        feature_schema=THREE_PHASE_FEATURE_SCHEMA_V2)
            source_bytes=original.read_bytes()
            copied,source_sha=initialize_teacher_model(
                THREE_PHASE_FEATURE_SCHEMA_V2,original)
            import hashlib
            self.assertEqual(source_sha,hashlib.sha256(source_bytes).hexdigest())
            self.assertEqual(state_hash(copied),expected)
            self.assertEqual(original.read_bytes(),source_bytes)
            for bad_schema in ('rnad-battle-features-v1',
                               'rnad-three-phase-features-v1'):
                with self.subTest(schema=bad_schema),self.assertRaises(ValueError):
                    initialize_teacher_model(bad_schema,original)
            with self.assertRaises(ValueError):
                initialize_teacher_model(THREE_PHASE_FEATURE_SCHEMA_V2,
                                         Path(tmp)/'missing-model.json')

    def test_invalid_objective_contract_fails_before_games(self):
        args={'seed':996001,'matches':16,'epochs':1,'teacher':'aq-public-belief',
              'three_phase':True,'features_v2':True,'phase_balanced':True}
        for changes in ({'initial_model':'missing-source.json',
                         'teacher':'basic-stage0'},
                        {'initial_model':'missing-source.json',
                         'features_v2':False},
                        {'phase_profile':'bad'},
                        {'phase_balanced':False,'phase_profile':'selection-focus'},
                        {'phase_balanced':False,'phase_profile':'draft-selection'},
                        {'label_smoothing':float('nan')},
                        {'label_smoothing':-.1},
                        {'label_smoothing':.2}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                train(**{**args,**changes})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'missing independent'):
                collect(996001,1,Path(tmp)/'unused',three_phase=True,
                        teacher='aq-public-belief',features_v2=True,
                        reuse_from=Path(tmp)/'no-such-directory')
            (Path(tmp)/'archive').mkdir()
            with self.assertRaisesRegex(ValueError,'missing authority-accepted'):
                collect(996001,1,Path(tmp)/'unused',three_phase=True,
                        teacher='aq-public-belief',features_v2=True,
                        reuse_from=Path(tmp)/'archive')


if __name__=='__main__':
    unittest.main()
