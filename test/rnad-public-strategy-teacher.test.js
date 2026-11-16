import test from 'node:test';
import assert from 'node:assert/strict';
import {createTeacherMatch} from '../scripts/rnad/export-teacher.mjs';
import {AQ_WEIGHTS_SHA256,AQ_STRATEGY_SELECTION_TEACHER_ID}
  from '../scripts/rnad/aq-teacher-policy.mjs';

test('new public AQ strategy-selection teacher completes a real authority Match',
  {timeout:600000},()=>{
    const trace=createTeacherMatch(975329,{
      threePhase:true,featuresV2:true,teacher:'aq-public-strategy-selection'});
    assert.equal(trace.schema,
      'rnad-aq-public-strategy-selection-three-phase-teacher-v1');
    assert.equal(trace.teacher,AQ_STRATEGY_SELECTION_TEACHER_ID);
    assert.equal(trace.teacher_model_sha256,AQ_WEIGHTS_SHA256);
    assert.equal(trace.completed,true);
    assert.equal(trace.score[trace.winner],4);
    for(const phase of ['draft','selection','battle'])
      assert.ok(trace.examples.some(row=>
        row.phase===phase&&row.choice>=0&&row.choice<row.actions.length));
    assert.ok(trace.examples.filter(row=>row.phase==='selection')
      .every(row=>row.actions.length===60));
  });
