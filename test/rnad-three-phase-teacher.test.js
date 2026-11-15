import test from 'node:test';
import assert from 'node:assert/strict';
import {createTeacherMatch} from '../scripts/rnad/export-teacher.mjs';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA}
  from '../src/ml/rnad-phase-features.js';

test('trusted Basic/Stage0 teacher emits actual accepted all-phase examples',()=>{
  const r=createTeacherMatch(975321,{threePhase:true});
  assert.equal(r.schema,'rnad-basic-stage0-three-phase-teacher-v1');
  assert.equal(r.feature_schema,RNAD_THREE_PHASE_FEATURE_SCHEMA);
  assert.equal(r.completed,true);
  assert.equal(r.score[r.winner],4);
  assert.ok(r.rounds>=4);
  assert.equal(r.examples.length,
    r.action_counts[0]+r.action_counts[1]+4*r.rounds);
  for(const stage of ['draft','selection']){
    assert.equal(r.examples.filter(e=>e.phase===stage).length,2*r.rounds);
  }
  assert.equal(r.examples.filter(e=>e.phase==='battle').length,
    r.action_counts[0]+r.action_counts[1]);
  assert.ok(r.examples.every(e=>
    [0,1].includes(e.actor)&&e.actions.length>=1&&
    e.actions[e.choice]?.length===16&&e.state.length===32&&
    e.terminal_target===(e.actor===r.winner?1:-1)));
});
