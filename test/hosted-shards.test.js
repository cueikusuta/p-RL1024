import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {planShards} from '../scripts/hosted-shard-plan.mjs';
test('256 training Matches distributed without overlap',()=>{
  const plan=planShards({mode:'training',total:256,shards:20,seed:975020});
  assert.equal(plan.include.length,20);
  assert.equal(plan.include.reduce((sum,s)=>sum+s.count,0),256);
  assert.equal(plan.include[0].offset,0);
  for(let i=0;i<plan.include.length;i++){
    const x=plan.include[i];assert.ok(x.count>=9);
    assert.equal(x.seedStart,975020+x.offset);
    assert.equal(x.seedEnd,x.seedStart+x.count-1);
    if(i>0)assert.equal(x.offset,plan.include[i-1].offset+plan.include[i-1].count);
  }
});
test('1000 evaluation Matches fit 12 shards',()=>{
  const p=planShards({total:1000,shards:12,seed:975020});
  assert.equal(p.include.reduce((s,x)=>s+x.count,0),1000);
  assert.ok(p.include.every(x=>x.count===83||x.count===84));
});
test('reject overbudget, impossible shards and seed overflows',()=>{
  assert.throws(()=>planShards({mode:'training',total:9,shards:2,seed:975020}));
  assert.throws(()=>planShards({total:1000,shards:21,seed:1}));
  assert.throws(()=>planShards({total:1000,shards:1,seed:0xffffffff}));
});

test('ordinary research pushes stay at nine Matches/one shard unless dispatched explicitly',()=>{
  const yaml=readFileSync(new URL('../.github/workflows/i975.yml',import.meta.url),'utf8');
  assert.match(yaml,/RL_MATCHES: \$\{\{ inputs\.matches \|\| '9' \}\}/);
  assert.match(yaml,/RL_SHARDS: \$\{\{ inputs\.shards \|\| '1' \}\}/);
  assert.match(yaml,/runAttempt/);
  assert.match(yaml,/GITHUB_RUN_ATTEMPT/);
});
