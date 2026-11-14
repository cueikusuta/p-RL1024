import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {compare} from '../scripts/verify-evaluation.mjs';

test('RL-Deep-001 frozen two-seat evaluation is repeatable',{timeout:120000},()=>{
  const root=resolve('tmp');mkdirSync(root,{recursive:true});
  const dir=mkdtempSync(join(root,'deep001-')),model='aq-runtime/models/RL-Deep-001.json';
  const sha=()=>createHash('sha256').update(readFileSync(model)).digest('hex');
  try {
    const before=sha(),outputs=['a','b'].map(x=>join(dir,x+'.json'));
    for(const out of outputs){const r=spawnSync(process.execPath,['scripts/evaluate-deep001.mjs','--model',model,'--purpose','smoke','--blocks','2','--seed','975020','--out',out],{encoding:'utf8',timeout:110000});assert.equal(r.status,0,r.stderr);}
    const [a,b]=outputs.map(p=>JSON.parse(readFileSync(p,'utf8')));
    assert.equal(a.modelSha256,'55673679fe773e4489e1dec505fb434525322175af5cb2ee1c2bf919adaf5d41');
    assert.equal(a.opponentModelSha256,'5ccc11a3362c7f5a8874847e3afc1f19c539d4ef258bd82704f53d8a1db1946e');
    assert.equal(compare(a,b).normalizedOutcomeSha256,a.outcomeDigestSha256);
    assert.equal(sha(),before);
  } finally { rmSync(dir,{recursive:true,force:true}); }
});
