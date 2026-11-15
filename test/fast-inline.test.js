import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

const cwd=fileURLToPath(new URL('../',import.meta.url));
function file(folder,name){return JSON.parse(readFileSync(join(folder,'RL-Deep-vs-aq',name),'utf8'));}
test('fast inline retains exact trained weights, checkpoints and independent frozen evaluation outputs',()=>{
  const home=mkdtempSync(join(tmpdir(),'i1-inline-parity-'));
  try{
    const reports=[];
    for(const [name,inline] of [['standard',false],['inline',true]]){
      const dir=join(home,name);
      const env={...process.env,RL_MODEL:'RL-Deep',RL_OPPONENT:'AQ-Base-001',
        RL_MATCHES:'9',RL_SEED:'975020',RL_OUTPUT_ROOT:dir,
        RL_FAST_WORKER:inline?'1':'0',RL_FAST_INLINE:inline?'1':'0'};
      const run=spawnSync(process.execPath,['scripts/run-experiment.mjs'],
        {cwd,env,encoding:'utf8',timeout:180000,maxBuffer:8*1024*1024});
      assert.equal(run.status,0,`mode=${name}\n${run.stderr??''}\n${(run.stdout??'').slice(-3000)}`);
      reports.push({train:file(dir,'training.json'),checkpoint:file(dir,'training-m8.json'),
        proof:file(dir,'comparison.json'),a:file(dir,'eval-a.json'),b:file(dir,'eval-b.json')});
    }
    const [reference,optimized]=reports;
    assert.equal(reference.train.modelSha256,optimized.train.modelSha256);
    assert.deepEqual(reference.train.history,optimized.train.history);
    assert.deepEqual(reference.checkpoint,optimized.checkpoint);
    assert.equal(reference.proof.modelSha256,optimized.proof.modelSha256);
    assert.equal(reference.proof.normalizedOutcomeSha256,optimized.proof.normalizedOutcomeSha256);
    assert.equal(reference.proof.modelFileSha256,optimized.proof.modelFileSha256);
    assert.deepEqual(reference.a,optimized.a);
    assert.deepEqual(reference.b,optimized.b);
  }finally{rmSync(home,{recursive:true,force:true});}
},{timeout:200000});
