import test from 'node:test';
import assert from 'node:assert/strict';
import {createTeacherMatch} from '../scripts/rnad/export-teacher.mjs';

test('v5 public AQ imitation emits only accepted real 4-win Match vectors',
  {timeout:120000},()=>{
  const r=createTeacherMatch(2310199,{threePhase:true,
    teacher:'aq-public-strategy-selection',featuresV5:true,
    utilityLabels:true});
  assert.equal(r.completed,true);
  assert.equal(r.feature_schema,'rnad-three-phase-features-v5');
  assert.equal(r.score[r.winner],4);
  assert.ok(r.examples.length>0);
  const phases=new Set(r.examples.map(x=>x.phase));
  assert.deepEqual([...phases].sort(),['battle','draft','selection']);
  for(const x of r.examples){
    assert.equal(x.state.length,96);
    assert.ok(x.actions.length>0);
    assert.ok(x.actions.every(a=>a.length===48));
    assert.ok(x.choice>=0&&x.choice<x.actions.length);
  }
  assert.ok(r.examples.filter(x=>x.phase==='selection')
    .every(x=>x.actions.length===60));
});
