"""Reproducibility gate: two uninterrupted completed Matches vs save/resume."""
import argparse
import json
import tempfile
from pathlib import Path

from scripts.rnad.train import train


def verify(seed=975023):
    with tempfile.TemporaryDirectory() as root:
        a=Path(root)/'continuous'
        b=Path(root)/'interrupted'
        uninterrupted=train(seed=seed,matches=2,out=str(a))
        train(seed=seed,matches=1,out=str(b))
        resumed=train(seed=seed,matches=1,out=str(b),resume=True)
        fields=('model_sha256','magnet_sha256',
                'completed_matches','updates')
        for key in fields:
            if uninterrupted[key]!=resumed[key]:
                raise AssertionError('Checkpoint resume diverged in '+key+
                    ': '+repr(uninterrupted[key])+' != '+repr(resumed[key]))
        first=json.loads((a/'rollouts'/f'accepted-seed{seed}.json').read_text())
        second=json.loads((b/'rollouts'/f'accepted-seed{seed}.json').read_text())
        if first!=second:
            raise AssertionError('Initial accepted Match changed across runs')
        seed2=(seed%0xffffffff)+1
        next1=json.loads((a/'rollouts'/f'accepted-seed{seed2}.json').read_text())
        next2=json.loads((b/'rollouts'/f'accepted-seed{seed2}.json').read_text())
        if next1!=next2:
            raise AssertionError('Resumed policy changes authoritative Match behavior')
        return {'schema':'rnad-resume-equivalence-v1','seed':seed,
                'completed_matches':2,
                'model_sha256':uninterrupted['model_sha256'],
                'magnet_sha256':uninterrupted['magnet_sha256'],
                'trajectories_identical':True,
                'model_weights_bitwise_equal':True}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=975023)
    a=p.parse_args()
    print(json.dumps(verify(a.seed),indent=2))
