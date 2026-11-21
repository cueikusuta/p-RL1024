import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeRnadThreePhaseDecisionV4,
  RNAD_THREE_PHASE_FEATURE_SCHEMA_V4}
  from '../src/ml/rnad-phase-features-v4.js';
import {encodeRnadThreePhaseDecisionV3}
  from '../src/ml/rnad-phase-features-v3.js';
import {projectRnadObservation} from '../src/ml/legal-action-adapter.js';

const own=Array.from({length:6},(_,i)=>({
  instanceId:'p'+i,setId:'s'+i,speciesId:'species'+i,
  cost:1,itemId:'item'+i,abilityId:'ability'+i,
  types:i%2?['Water']:['Fire'],
  moves:i%2?['surf','lightscreen']:['flamethrower','swordsdance'],
  stats:{hp:100,atk:100,def:100,spa:105,spd:95,spe:75+25*i},
  hp:100,maxHP:100
}));
function obs(phase='selection'){
  const raw={phase,round:2,score:[0,0],budget:13,grant:10,
    self:phase==='selection'?{credits:13,team:structuredClone(own)}:
      {credits:13,carry:[],fresh:structuredClone(own)},
    opponent:{species:phase==='selection'?['unknown']:[],
      hiddenTeam:[{moves:['hidden-attack'],itemId:'unknown-secret'}]}};
  return projectRnadObservation(raw,{seat:0});
}
test('v4 retains frozen v3 dimensions, but represents lead identities and roles',()=>{
  const x=obs();
  const legal=[['p0','p1','p2'],['p1','p0','p2'],['p0','p2','p1']];
  const v4=encodeRnadThreePhaseDecisionV4(x,legal);
  const v3=encodeRnadThreePhaseDecisionV3(x,legal);
  assert.equal(v4.schema,RNAD_THREE_PHASE_FEATURE_SCHEMA_V4);
  assert.equal(v4.state.length,48);
  assert.ok(v4.actions.every(a=>a.length===32&&a.every(Number.isFinite)));
  assert.deepEqual(v4.actions.map(a=>a.slice(0,16)),
                   v3.actions.map(a=>a.slice(0,16)));
  assert.notDeepEqual(v4.actions[0].slice(16),v4.actions[1].slice(16));
  assert.deepEqual(v4.actions[0].slice(16),v4.actions[2].slice(16),
    'reserve ordering is not a tactical feature');
});
test('v4 ignores opponent unrevealed set and move details',()=>{
  const a=obs();
  const b=obs();
  b.opponent.secretMove='never-public';
  b.opponent.hiddenTeam=[{itemId:'different',abilityId:'secret'}];
  const legal=[['p0','p1','p2']];
  assert.deepEqual(encodeRnadThreePhaseDecisionV4(a,legal),
                   encodeRnadThreePhaseDecisionV4(b,legal));
});
test('v4 draft is a set: offered ordering is not confused with a lead',()=>{
  const x=obs('draft'),team=['p0','p1','p2','p3','p4','p5'];
  const a=encodeRnadThreePhaseDecisionV4(x,[team]);
  const b=encodeRnadThreePhaseDecisionV4(x,[[...team].reverse()]);
  assert.deepEqual(a.actions[0].slice(16),b.actions[0].slice(16));
});
