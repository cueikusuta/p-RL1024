import test from 'node:test';
import assert from 'node:assert/strict';
import {createTeacherMatch} from '../scripts/rnad/export-teacher.mjs';
test('accepted public strategy teacher supplies non-forged draft/selection utilities',
  {timeout:600000},()=>{
    const result=createTeacherMatch(975335,{
      threePhase:true,featuresV4:true,utilityLabels:true,
      teacher:'aq-public-strategy-selection'});
    assert.equal(result.completed,true);
    assert.equal(result.score[result.winner],4);
    assert.equal(result.utility_labels,'aq-public-phase-rank-v1');
    for(const e of result.examples){
      if(e.phase==='battle'){
        assert.equal(e.candidate_utilities,undefined);
        continue;
      }
      const scores=e.candidate_utilities;
      assert.equal(scores.length,e.actions.length);
      assert.ok(scores.every(Number.isFinite));
      const best=Math.max(...scores);
      assert.ok(best-scores[e.choice]<1e-5);
    }
    assert.ok(result.examples.filter(e=>e.phase==='draft').length>=8);
    assert.ok(result.examples.filter(e=>e.phase==='selection')
      .every(e=>e.candidate_utilities.length===60));
  });
test('utilities are opt-in and invalid for legacy teacher schema',()=>{
  assert.throws(()=>createTeacherMatch(975335,{
    threePhase:true,featuresV2:true,utilityLabels:true,
    teacher:'aq-public-belief'}));
});
