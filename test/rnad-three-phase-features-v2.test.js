import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeRnadThreePhaseDecisionV2,
  RNAD_THREE_PHASE_FEATURE_SCHEMA_V2}
  from '../src/ml/rnad-phase-features-v2.js';
import {projectRnadObservation} from '../src/ml/legal-action-adapter.js';
const own=Array.from({length:6},(_,i)=>({
  instanceId:'p'+i,setId:'s'+i,speciesId:'species'+i,
  cost:1,itemId:'item'+i,types:i%2?['Water']:['Fire'],
  moves:i%2?['surf']:['flamethrower'],
  stats:{hp:100,atk:100,def:100,spa:105,spd:95,spe:100},
  hp:100,maxHP:100
}));
const raw=()=>({phase:'selection',round:2,score:[0,0],budget:13,grant:10,
  self:{credits:13,team:structuredClone(own)},
  opponent:{species:[],privateMoves:['hidden']}})
const legal=[['p0','p1','p2'],['p2','p3','p4']];
test('v2 semantic encoder is finite, fixed-dimension, and schema-isolated',()=>{
  const obs=projectRnadObservation(raw(),{seat:0});
  const enc=encodeRnadThreePhaseDecisionV2(obs,legal);
  assert.equal(enc.schema,RNAD_THREE_PHASE_FEATURE_SCHEMA_V2);
  assert.equal(enc.state.length,32);
  assert.equal(enc.actions.length,2);
  assert.ok(enc.actions.every(r=>r.length===16&&r.every(Number.isFinite)));
  assert.ok(enc.state.every(Number.isFinite));
  assert.ok(enc.actions.every(row=>row[5]>=0&&row[5]<=1&&
    row[14]>=0&&row[14]<=1&&row[15]>=0&&row[15]<=1));
});
test('v2 rejects secret candidate injection and ignores raw hidden opponent moves',()=>{
  const base=raw(),mutated=raw();
  mutated.opponent.privateMoves=['secret different'];
  mutated.opponent.fraudulentNested={hiddenItem:'zz'};
  const a=encodeRnadThreePhaseDecisionV2(
    projectRnadObservation(base,{seat:0}),legal);
  const b=encodeRnadThreePhaseDecisionV2(
    projectRnadObservation(mutated,{seat:0}),legal);
  assert.deepEqual(a,b);
  assert.throws(()=>encodeRnadThreePhaseDecisionV2(
    projectRnadObservation(base,{seat:0}),[['p0','p1','forged']]),
    /Invalid own draft\/selection candidate/);
});
test('v2 encodes genuine public type coverage instead of arbitrary species hashes',()=>{
  const obs=projectRnadObservation(raw(),{seat:0});
  const encoded=encodeRnadThreePhaseDecisionV2(obs,legal);
  assert.notEqual(encoded.actions[0][15],encoded.actions[1][15]);
  assert.notEqual(encoded.actions[0][14],encoded.actions[1][14]);
});
