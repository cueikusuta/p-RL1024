import test from 'node:test';
import assert from 'node:assert/strict';
import {compare,normalize} from '../scripts/verify-evaluation.mjs';

function report() {
  return {schema:'i975-frozen-paired-four-win-evaluation-v1',purpose:'smoke',blocks:2,completedMatches:4,formalEligible:false,modelSha256:'a'.repeat(64),seedDigestSha256:'b'.repeat(64),outcomeDigestSha256:'c'.repeat(64),records:[0,1].flatMap(block => [0,1].map(rlSeat => ({block,seed:100+block,rlSeat,winner:0,won:Number(rlSeat===0),rounds:4,actionCounts:[20,21],score:[4,1]})))};
}
test('normalizes record order without changing seat order', () => {
  const a=report(), b=report(); b.records.reverse();
  assert.equal(normalize(b)[0].rlSeat,0);
  assert.equal(compare(a,b).schema,'p-rl1024-normalized-outcome-v1');
});
test('rejects result, model and seed changes', () => {
  const a=report();
  for (const mutate of [b=>b.records[0].actionCounts[0]++, b=>b.modelSha256='other', b=>b.seedDigestSha256='other']) {
    const b=report(); mutate(b); assert.throws(()=>compare(a,b));
  }
});
test('rejects missing or duplicate match records', () => {
  const a=report(); a.records[0].rlSeat=1; assert.throws(()=>normalize(a));
  const b=report(); delete b.records[0].score; assert.throws(()=>normalize(b));
});
