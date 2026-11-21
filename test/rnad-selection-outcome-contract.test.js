import test from 'node:test';
import assert from 'node:assert/strict';
import {createSelectionRankSession,evaluateSelectionRankOutcomes}
  from '../scripts/rnad/evaluate-selection-outcome-rank.mjs';
test('outcome probe rejects unsupported model/rank and invalid seeds',()=>{
  assert.throws(()=>createSelectionRankSession({model:{},seat:0,rank:0}));
  assert.throws(()=>createSelectionRankSession({model:null,seat:0,rank:-1}));
  assert.throws(()=>evaluateSelectionRankOutcomes({
    v3Path:'missing',seed:1,pairs:0,ranks:3}));
  assert.throws(()=>evaluateSelectionRankOutcomes({
    v3Path:'missing',seed:1,pairs:3,ranks:1}));
});
