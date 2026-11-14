import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {availableParallelism} from 'node:os';
import {createModel} from '../models/deep.mjs';
import {mixSeed} from '../scripts/dev/simulator/runner.js';
import {frozenDeepBatch,acceptWorkerRows} from '../scripts/batch-frozen-deep.mjs';

test('worker pool preserves full-match outcomes, order, and frozen model', {timeout:60000},async()=>{
  const model=createModel(975020);
  const modelBefore=JSON.stringify(model);
  const seeds=Array.from({length:8},(_,i)=>mixSeed(975020,i,0x2b10));
  const sequential=await frozenDeepBatch(model,seeds,{workers:1});
  const parallel=await frozenDeepBatch(model,seeds,{workers:Math.min(4,availableParallelism())});
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

test('worker validation rejects sparse, duplicated, malformed and missing batches',()=>{
  const tasks=[{index:0},{index:1}];
  const valid=index=>({index,result:{completed:true,winner:0,score:[4,0]}});
  const accept=rows=>{
    const ordered=new Array(2),filled=new Set();
    acceptWorkerRows(tasks,rows,ordered,filled);
    return {ordered,filled};
  };
  const good=accept([valid(1),valid(0)]);
  assert.equal(good.filled.size,2);
  assert.equal(good.ordered[0].winner,0);
  assert.throws(()=>accept([]),/incomplete/);
  assert.throws(()=>accept([valid(0)]),/incomplete/);
  assert.throws(()=>accept([valid(0),valid(0)]),/duplicate/);
  assert.throws(()=>accept([valid(0),valid(5)]),/unexpected/);
  assert.throws(()=>accept(new Array(2)),/unexpected/);
  assert.throws(()=>accept([valid(0),{index:1,result:null}]),/invalid Match/);
  const filled=new Set([0]),ordered=new Array(2);
  assert.throws(()=>acceptWorkerRows(tasks,[valid(0),valid(1)],ordered,filled),/duplicate/);
  assert.equal(0 in ordered,false);
});

test('worker count above allocated CPUs is rejected',async()=>{
  const model=createModel(975020);
  await assert.rejects(
    frozenDeepBatch(model,[1,2],{workers:availableParallelism()+1}),
    /Worker count/);
});
