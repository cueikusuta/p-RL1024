"""RNAD-001 verified checkpoint → frozen evaluation-only Actor export."""
import argparse
import copy
import json
import subprocess
from pathlib import Path

import torch
from scripts.rnad.checkpoint import restore_checkpoint
from scripts.rnad.neural import RNADNetwork, freeze_magnet, export_json


def export_checkpoint(directory, destination):
    source=subprocess.check_output(
        ['git','rev-parse','HEAD'], text=True, timeout=5).strip()
    engine=json.loads(Path('provenance/manifest.json').read_text())['sourceCommit']
    model=RNADNetwork()
    target=copy.deepcopy(model)
    magnet=freeze_magnet(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(
        optimizer,step_size=100,gamma=.98)
    record=restore_checkpoint(
        directory,policy=model,value=target,magnet=magnet,
        optimizer=optimizer,scheduler=scheduler,restore_rng=False,
        expected={
          'model_id':'RNAD-001',
          'source_sha':source,'engine_snapshot_sha':engine,
          'training_code_sha':source,
          'observation_schema_version':'rnad-battle-features-v1'})
    if record['metadata']['completed_matches']<1:
        raise ValueError('Cannot evaluate an untrained policy')
    export_json(model,destination)
    return {'schema':'rnad-evaluation-model-export-v1',
            'source_sha':source,'engine_sha':engine,
            'checkpoint_sha256':record['checkpoint_sha256'],
            'model_path':str(destination),
            'training_matches':record['metadata']['completed_matches']}


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--checkpoint',required=True)
    cli.add_argument('--out',required=True)
    args=cli.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint,args.out),indent=2))
