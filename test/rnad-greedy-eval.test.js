import test from 'node:test';
import assert from 'node:assert/strict';
import {greedyLegalIndex} from '../scripts/rnad/evaluate-neural-basic.mjs';
test('greedy deployment consistently selects the highest legal logit',()=>{
  assert.equal(greedyLegalIndex([0,1,-2]),1);
  assert.equal(greedyLegalIndex([-1,-1,-3]),0);
  assert.equal(greedyLegalIndex([2]),0);
  for(const invalid of [[],[Infinity],[NaN],[0,-Infinity],[1,'2']]){
    assert.throws(()=>greedyLegalIndex(invalid),/finite bounded logits/);
  }
});
