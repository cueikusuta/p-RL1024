import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createModel} from '../models/deep.mjs';
import {mixSeed} from '../scripts/dev/simulator/runner.js';
import {frozenDeepBatch} from '../scripts/batch-frozen-deep.mjs';

test('worker pool preserves full-match outcomes, order, and frozen model', {timeout:60000},async()=>{
  const model=createModel(975020);
  const modelBefore=JSON.stringify(model);
  const seeds=Array.from({length:8},(_,i)=>mixSeed(975020,i,0x2b10));
  const sequential=await frozenDeepBatch(model,seeds,{workers:1});
  const parallel=await frozenDeepBatch(model,seeds,{workers:4});
  const rows=results=>results.map((r,i)=>({
    seed:seeds[i],score:r.score,rounds:r.rounds,
    roundWinners:r.roundWinners,actions:r.actionCounts,
  }));
  assert.deepEqual(rows(parallel),rows(sequential));
  const digest=createHash('sha256').update(JSON.stringify(rows(sequential))).digest('hex');
  assert.equal(digest,'b7323a8441e0a518257452f2c14df61952ef7b5bca7b91838a92e0dc82a896c4');
  assert.equal(JSON.stringify(model),modelBefore);
  for(const r of parallel) {
    assert.equal(r.completed,true);
    assert.equal(r.score[r.winner],4);
    assert.ok(r.pureRlStats.every(s=>s.train===false&&s.updates===0));
  }
});
