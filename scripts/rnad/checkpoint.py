"""RNAD-001 immutable match-boundary PyTorch checkpoint contract.

Infrastructure only; this module is NOT the R-NaD neural learner.
The caller owns the authoritative completed-match boundary.
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path

import torch

SCHEMA = 'rnad-pytorch-checkpoint-v1'
REQUIRED = ('model_id', 'source_sha', 'engine_snapshot_sha',
            'observation_schema_version', 'training_code_sha',
            'opponent_policy_version', 'completed_matches',
            'accepted_decisions', 'updates', 'next_match_seed',
            'last_match_id', 'boundary')


def _check(metadata):
    if not isinstance(metadata, dict) or not all(k in metadata for k in REQUIRED):
        raise ValueError('Missing checkpoint provenance/optimizer progress')
    if metadata['boundary'] != 'completed_match':
        raise ValueError('Checkpoint cannot represent a partial match')
    for name in ('source_sha', 'engine_snapshot_sha', 'training_code_sha'):
        if not isinstance(metadata[name], str) or not re.fullmatch('[0-9a-fA-F]{40,64}', metadata[name]):
            raise ValueError('Missing immutable source sha: ' + name)
    for name in ('model_id', 'observation_schema_version',
                 'opponent_policy_version', 'last_match_id'):
        if not isinstance(metadata[name], str) or not metadata[name] or len(metadata[name]) > 160:
            raise ValueError('Invalid checkpoint label: ' + name)
    for name in ('completed_matches', 'accepted_decisions', 'updates'):
        if type(metadata[name]) is not int or metadata[name] < 0:
            raise ValueError('Invalid training counter: ' + name)
    if metadata['completed_matches'] < 1 or type(metadata['next_match_seed']) is not int or not (1 <= metadata['next_match_seed'] <= 0xffffffff):
        raise ValueError('Invalid completed match progress / next seed')
    json.dumps(metadata, allow_nan=False)


def _atomic_bytes(directory, writer):
    fd, temporary = tempfile.mkstemp(dir=directory, prefix='.rnad-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
        return Path(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(directory, *, policy, value, magnet, optimizer, scheduler,
                    metadata, expected_prior_sha=None):
    """Save an immutable checkpoint and CAS-update only the mutable index.

    Workflows must also serialize writers: local compare-and-swap is not
    a distributed lock. A stale expected previous SHA fails closed.
    """
    _check(metadata)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    index_path = path / 'latest.json'
    current = json.loads(index_path.read_text()) if index_path.exists() else None
    prior = current['checkpoint_sha256'] if current else None
    if prior != expected_prior_sha:
        raise ValueError('Checkpoint compare-and-swap conflict')
    payload = {
        'schema': SCHEMA,
        'metadata': json.loads(json.dumps(metadata, allow_nan=False)),
        'policy': policy.state_dict(),
        'value': value.state_dict(),
        'magnet': magnet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'python_rng': random.getstate(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temp = _atomic_bytes(path, lambda f: torch.save(payload, f))
    try:
        checksum = hashlib.sha256(temp.read_bytes()).hexdigest()
        artifact = f"checkpoint-u{metadata['updates']}-{checksum}.pt"
        destination = path / artifact
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
                raise ValueError('Immutable checkpoint collision')
            temp.unlink()
        else:
            os.replace(temp, destination)
        index = {'schema': SCHEMA, 'checkpoint_file': artifact,
                 'checkpoint_sha256': checksum, 'metadata': metadata}
        encoded = (json.dumps(index, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
        tmp_index = _atomic_bytes(path, lambda f: f.write(encoded))
        os.replace(tmp_index, index_path)
        return index
    finally:
        if temp.exists():
            temp.unlink()


def restore_checkpoint(directory, *, policy, value, magnet, optimizer, scheduler,
                       expected=None, restore_rng=True):
    path = Path(directory)
    index = json.loads((path / 'latest.json').read_text())
    if index['schema'] != SCHEMA or not re.fullmatch(
            r'checkpoint-u[0-9]+-[0-9a-f]{64}\.pt', index['checkpoint_file']):
        raise ValueError('Invalid checkpoint index')
    artifact = path / index['checkpoint_file']
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if checksum != index['checkpoint_sha256'] or not artifact.name.endswith(checksum + '.pt'):
        raise ValueError('Checkpoint SHA mismatch')
    payload = torch.load(artifact, map_location='cpu', weights_only=True)
    if payload.get('schema') != SCHEMA or payload.get('metadata') != index['metadata']:
        raise ValueError('Checkpoint payload metadata mismatch')
    _check(payload['metadata'])
    if expected:
        for name, val in expected.items():
            if payload['metadata'].get(name) != val:
                raise ValueError('Checkpoint experiment identity mismatch: ' + name)
    policy.load_state_dict(payload['policy'])
    value.load_state_dict(payload['value'])
    magnet.load_state_dict(payload['magnet'])
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    if restore_rng:
        random.setstate(payload['python_rng'])
        torch.set_rng_state(payload['torch_rng'])
        if torch.cuda.is_available() and payload['cuda_rng']:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return index
