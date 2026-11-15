"""PyTorch checkpoint save/reload parity; isolated from real Pokemon training."""
import random
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.rnad.checkpoint import save_checkpoint, restore_checkpoint

SHA = 'ea81165993110ee9fb3a09dcb97c6708c0d7397b'
META = {
    'model_id': 'RNAD-001',
    'source_sha': SHA,
    'engine_snapshot_sha': SHA,
    'observation_schema_version': 'rnad-v1',
    'training_code_sha': SHA,
    'opponent_policy_version': 'fixed-v1',
    'completed_matches': 1,
    'accepted_decisions': 120,
    'updates': 2,
    'next_match_seed': 125,
    'last_match_id': 'm-1',
    'boundary': 'completed_match',
}


class CheckpointContractTest(unittest.TestCase):
    def setup_model(self):
        torch.manual_seed(42)
        random.seed(42)
        policy = torch.nn.Linear(3, 2)
        value = torch.nn.Linear(3, 1)
        magnet = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters()), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        x = torch.rand(3)
        loss = policy(x).square().sum() + value(x).square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return policy, value, magnet, optimizer, scheduler

    def test_immutable_and_resume(self):
        policy, value, magnet, optimizer, scheduler = self.setup_model()
        args = {'policy': policy, 'value': value, 'magnet': magnet,
                'optimizer': optimizer, 'scheduler': scheduler}
        with tempfile.TemporaryDirectory() as directory:
            first = save_checkpoint(directory, **args, metadata=META)
            expected_rng = (random.random(), torch.rand(3))
            with torch.no_grad():
                for p in policy.parameters():
                    p.add_(11)
            random.random()
            torch.rand(3)
            restored = restore_checkpoint(
                directory, **args, expected={'engine_snapshot_sha': SHA})
            self.assertEqual(restored['checkpoint_sha256'], first['checkpoint_sha256'])
            self.assertAlmostEqual(random.random(), expected_rng[0])
            self.assertTrue(torch.equal(torch.rand(3), expected_rng[1]))
            self.assertEqual(scheduler.last_epoch, 1)
            with self.assertRaisesRegex(ValueError, 'compare-and-swap'):
                save_checkpoint(directory, **args, metadata=META)
            with self.assertRaisesRegex(ValueError, 'identity'):
                restore_checkpoint(directory, **args,
                                   expected={'engine_snapshot_sha': '0' * 40})
            with (Path(directory) / first['checkpoint_file']).open('ab') as f:
                f.write(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                restore_checkpoint(directory, **args)

    def test_partial_match_rejected(self):
        policy, value, magnet, optimizer, scheduler = self.setup_model()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'partial match'):
                save_checkpoint(directory, policy=policy, value=value,
                                magnet=magnet, optimizer=optimizer,
                                scheduler=scheduler,
                                metadata={**META, 'boundary': 'pending'})


if __name__ == '__main__':
    unittest.main()
