"""Release-upload P1 regression: no write to gh on forged provenance."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE='a'*40
SEED=10001
TAG=f'rnad-001-{SOURCE[:16]}-s{SEED}'

class ReleaseUploadPreflightTest(unittest.TestCase):
    def setUp(self):
        if shutil.which('jq') is None:
            self.skipTest('jq is required by the upload shell integration')
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp=Path(self.temp.name)
        self.folder=self.tmp/'checkpoints'
        self.folder.mkdir()
        self.bin=self.tmp/'bin'
        self.bin.mkdir()
        self.trace=self.tmp/'gh.trace'
        self.bin.joinpath('gh').write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" "$*" >> "$GH_TRACE"\n'
            'exit 0\n')
        self.bin.joinpath('gh').chmod(0o755)
        self.payload=b'immutable validated checkpoint fixture'
        self.sha=hashlib.sha256(self.payload).hexdigest()
        self.name=f'checkpoint-u1-{self.sha}.pt'
        (self.folder/self.name).write_bytes(self.payload)
        engine=json.loads((ROOT/'provenance/manifest.json').read_text())['sourceCommit']
        self.index={
            'schema':'rnad-pytorch-checkpoint-v1',
            'checkpoint_file':self.name,'checkpoint_sha256':self.sha,
            'metadata':{
                'updates':1,'completed_matches':1,'accepted_decisions':100,
                'model_id':'RNAD-001','initial_seed':SEED,
                'source_sha':SOURCE,'training_code_sha':SOURCE,
                'engine_snapshot_sha':engine,
                'observation_schema_version':'rnad-battle-features-v1',
                'boundary':'completed_match',
                'completed_match_ids':[f'rnad-match-{SEED}'],
                'last_match_id':f'rnad-match-{SEED}',
                'next_match_seed':SEED+1
            }
        }
    def run_upload(self, index=None, tag=TAG,source=SOURCE):
        (self.folder/'latest.json').write_text(json.dumps(index or self.index))
        if self.trace.exists():
            self.trace.unlink()
        env={**os.environ,'GH_REPO':'cueikusuta/p-RL1024',
             'GH_TOKEN':'dummy','GH_TRACE':str(self.trace),
             'RNAD_SOURCE_SHA':source,
             'PATH':str(self.bin)+os.pathsep+os.environ['PATH']}
        return subprocess.run(['bash','scripts/rnad/release-checkpoint.sh',
                               'upload',tag,str(self.folder)],
            cwd=ROOT,env=env,text=True,capture_output=True,timeout=15)
    def test_valid_fake_release_still_requires_pinned_provenance(self):
        result=self.run_upload()
        self.assertEqual(result.returncode,0,result.stderr)
        commands=self.trace.read_text()
        self.assertIn('release view',commands)
        self.assertIn('release upload',commands)
    def test_forged_index_rejected_before_any_gh_write(self):
        mutations={
            'source':('source_sha','b'*40),
            'training_code':('training_code_sha','b'*40),
            'engine':('engine_snapshot_sha','c'*40),
            'model_id':('model_id','RNAD-002-3phase'),
            'feature_schema':('observation_schema_version','rnad-three-phase-features-v2'),
            'seed':('initial_seed',SEED+1),
            'completed_updates':('completed_matches',2),
            'update_count':('updates',2),
            'accepted':('accepted_decisions',0),
            'boundary':('boundary','unaccepted'),
            'match_id':('last_match_id','rnad-match-evil'),
            'next_seed':('next_match_seed',SEED+7),
            'match_list':('completed_match_ids',['rnad-match-other'])
        }
        for title,(key,value) in mutations.items():
            with self.subTest(field=title):
                bad=copy.deepcopy(self.index)
                bad['metadata'][key]=value
                r=self.run_upload(bad)
                self.assertNotEqual(r.returncode,0,title)
                self.assertFalse(self.trace.exists(),title)
    def test_wrong_pin_tag_and_checkpoint_sha_rejected(self):
        for kind in ('source-pin','tag','checksum','unsafe-path'):
            with self.subTest(kind=kind):
                bad=copy.deepcopy(self.index)
                if kind=='source-pin':
                    r=self.run_upload(bad,source='b'*40)
                elif kind=='tag':
                    r=self.run_upload(bad,tag='rnad-001-'+SOURCE[:16]+'-s10002')
                elif kind=='checksum':
                    bad['checkpoint_sha256']='d'*64
                    r=self.run_upload(bad)
                else:
                    bad['checkpoint_file']='../../unsafe.pt'
                    r=self.run_upload(bad)
                self.assertNotEqual(r.returncode,0,kind)
                self.assertFalse(self.trace.exists(),kind)

if __name__=='__main__':
    unittest.main()
