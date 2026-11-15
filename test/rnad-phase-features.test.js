import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeRnadThreePhaseDecision,RNAD_THREE_PHASE_FEATURE_SCHEMA}
  from '../src/ml/rnad-phase-features.js';
import {projectRnadObservation,rnadLegalActions} from '../src/ml/legal-action-adapter.js';

const species=(i)=>({
  instanceId:'p'+i,setId:'s'+i,speciesId:'s'+i,
  types:i%2?['Grass']:['Fire'],itemId:'item'+i,
  cost:1,stats:{atk:110,def:90,spa:100,spd:95,spe:80},mega:null
});
function view(phase){
  const team=Array.from({length:6},(_,i)=>species(i));
  const base={phase,round:1,score:[0,0],budget:10,grant:10,
    self:{credits:10,carry:[],fresh:team,team},
    opponent:{species:['known-public'],ready:false,privateMove:'hidden'}};
  if(phase==='selection')delete base.self.fresh;
  return base;
}
test('three-phase encoder produces finite fixed-size candidate features',()=>{
  for(const phase of ['draft','selection']){
    const obs=projectRnadObservation(view(phase),{seat:0});
    const legal=rnadLegalActions(view(phase));
    const encoded=encodeRnadThreePhaseDecision(obs,legal);
    assert.equal(encoded.schema,RNAD_THREE_PHASE_FEATURE_SCHEMA);
    assert.equal(encoded.state.length,32);
    assert.ok(encoded.actions.length>0);
    assert.ok(encoded.actions.every(x=>x.length===16&&x.every(Number.isFinite)));
    assert.equal(encoded.state[phase==='draft'?0:1],1);
  }
});
test('hidden opponent properties never enter projected features',()=>{
  for(const phase of ['draft','selection']){
    const v=view(phase),other=structuredClone(v);
    other.opponent.privateMove='totally different secret';
    other.opponent.internalCheckpoint={secret:'zzzz'};
    const a=encodeRnadThreePhaseDecision(projectRnadObservation(v,{seat:0}),
      rnadLegalActions(v));
    const b=encodeRnadThreePhaseDecision(projectRnadObservation(other,{seat:0}),
      rnadLegalActions(other));
    assert.deepEqual(a,b);
  }
});
test('candidate with missing or repeated own Pokémon is rejected',()=>{
  const obs=projectRnadObservation(view('draft'),{seat:0});
  assert.throws(()=>encodeRnadThreePhaseDecision(obs,[
    ['p0','p1','p2','p3','p4','hidden']]),/Invalid own/);
  assert.throws(()=>encodeRnadThreePhaseDecision(obs,[
    ['p0','p0','p2','p3','p4','p5']]),/Invalid own/);
});
