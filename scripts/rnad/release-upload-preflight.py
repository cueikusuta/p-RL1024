"""Fail-closed upload preflight for an immutable RNAD-001 release.

The frozen engine tag, source PIN, initial seed, checkpoint SHA and exact
completed-match index MUST agree before any gh release create/upload call.
For a download the shell additionally enforces the safe two-file tar manifest.
This validator never loads or executes a Torch checkpoint.
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from pathlib import Path

def verify_upload(folder: str, tag: str, expected_source_sha: str):
    root=Path(folder)
    if (not re.fullmatch(r'[0-9a-f]{40}',expected_source_sha) or
        not re.fullmatch(r'rnad-001-[0-9a-f]{16}-s[0-9]{1,10}',tag)):
        raise ValueError('Invalid pinned source SHA or immutable release tag')
    manifest=json.loads((root/'latest.json').read_text())
    md=manifest.get('metadata')
    if not isinstance(md,dict):
        raise ValueError('Missing immutable checkpoint metadata')
    name=manifest.get('checkpoint_file')
    sha=manifest.get('checkpoint_sha256')
    updates=md.get('updates')
    seed=md.get('initial_seed')
    source=md.get('source_sha')
    engine=json.loads(Path('provenance/manifest.json').read_text())['sourceCommit']
    if (manifest.get('schema')!='rnad-pytorch-checkpoint-v1'
        or type(updates) is not int or not 1<=updates<=2048
        or type(seed) is not int or not 1<=seed<=0xffffffff
        or source!=expected_source_sha
        or md.get('training_code_sha')!=source
        or md.get('engine_snapshot_sha')!=engine
        or md.get('model_id')!='RNAD-001'
        or md.get('observation_schema_version')!='rnad-battle-features-v1'
        or md.get('completed_matches')!=updates
        or md.get('boundary')!='completed_match'
        or type(md.get('accepted_decisions')) is not int
        or md['accepted_decisions']<=0
        or not isinstance(sha,str)
        or re.fullmatch(r'[0-9a-f]{64}',sha) is None
        or not isinstance(name,str)
        or re.fullmatch(r'checkpoint-u[0-9]+-[0-9a-f]{64}\.pt',name) is None
        or name!=f'checkpoint-u{updates}-{sha}.pt'
        or tag!=f'rnad-001-{source[:16]}-s{seed}'):
        raise ValueError('Checkpoint source/seed/model/schema/update/tag mismatch')
    expected_ids=['rnad-match-'+str(((seed+index-1)%0xffffffff)+1)
                  for index in range(updates)]
    if (md.get('completed_match_ids')!=expected_ids or
        md.get('last_match_id')!=expected_ids[-1] or
        md.get('next_match_seed')!=((seed+updates-1)%0xffffffff)+1):
        raise ValueError('Checkpoint completed Match sequence identity mismatch')
    path=root/name
    if (not path.is_file() or path.is_symlink() or
        hashlib.sha256(path.read_bytes()).hexdigest()!=sha):
        raise ValueError('Checkpoint file does not match the exact SHA-256')
    return {'name':name,'sha256':sha,'updates':updates,'seed':seed,
            'source_sha':source,'engine_sha':engine,'tag':tag}

if __name__=='__main__':
    if len(sys.argv)!=4:
        raise SystemExit('Usage: release-upload-preflight.py FOLDER TAG SOURCE_SHA')
    try:
        print(json.dumps(verify_upload(*sys.argv[1:])))
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as exc:
        raise SystemExit('Release preflight rejected: '+str(exc)) from exc
