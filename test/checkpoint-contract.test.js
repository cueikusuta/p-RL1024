import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,mkdtempSync,readFileSync,rmSync,writeFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';

for(const [id,script,extra] of [['RL-Linear','scripts/train-linear.mjs',['--all-phases','true']],['RL-Deep','scripts/train-deep.mjs',[]],['RL-Linear vs AQ','scripts/train-vs-aq.mjs',['--model','RL-Linear']],['RL-Deep vs AQ','scripts/train-vs-aq.mjs',['--model','RL-Deep']]]) {
  test(`${id} resumes a validated checkpoint without changing training`,{timeout:120000},()=>{
    const temporaryRoot=resolve('tmp');mkdirSync(temporaryRoot,{recursive:true});
    const dir=mkdtempSync(join(temporaryRoot,'rl-checkpoint-'));
    const full=join(dir,'full.json'),resumed=join(dir,'resumed.json'),checkpoint=join(dir,'full-m8.json');
    const base=[script,...extra,'--matches','9','--seed','975020','--checkpoint-every','8'];
    const run=args=>spawnSync(process.execPath,args,{encoding:'utf8',timeout:110000});
    try {
      const first=run([...base,'--out',full]);assert.equal(first.status,0,first.stderr);
      const second=run([...base,'--resume',checkpoint,'--out',resumed]);assert.equal(second.status,0,second.stderr);
      const a=JSON.parse(readFileSync(full)),b=JSON.parse(readFileSync(resumed));
      assert.equal(a.modelSha256,b.modelSha256);
      assert.deepEqual(a.history??a.matches,b.history??b.matches);
      const broken=JSON.parse(readFileSync(checkpoint));broken.seed=1;
      const bad=join(dir,'bad.json');writeFileSync(bad,JSON.stringify(broken));
      assert.notEqual(run([...base,'--resume',bad,'--out',resumed]).status,0);
    } finally { rmSync(dir,{recursive:true,force:true}); }
  });
}
