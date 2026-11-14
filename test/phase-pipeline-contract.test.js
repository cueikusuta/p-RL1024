import test from 'node:test';
import assert from 'node:assert/strict';
import {createPhasePipeline,PHASE_PIPELINE_SCHEMA}
  from '../src/ai/phase-pipeline.js';
const team=Array.from({length:6},(_,i)=>({instanceId:'t'+i,
  setId:'set'+i,speciesId:'species'+i,moves:['thunderbolt'],types:['electric'],
  itemId:'item'+i,stats:{hp:99}}));
const draft=()=>({phase:'draft',self:{carry:[],fresh:team},opponent:{hidden:'SECRET'}});
const selection=()=>({phase:'selection',self:{team},opponent:{species:['public-mon'],secret:'HIDDEN'}});
const battle=()=>({requestType:'turn',self:{team},opponent:{speciesId:'public-mon',hpPercent:75,hiddenAbility:'SECRET'}});
function setup(observer){
  return createPhasePipeline({
    draft:{id:'draft-ai',select:({context})=>{observer.push(context);return {action:team.map(t=>t.instanceId),handoff:{goal:'defensive'}};}},
    selection:{id:'selection-ai',select:({context})=>{observer.push(context);return {action:team.slice(0,3).map(t=>t.instanceId),handoff:{lead:'t0'}};}},
    battle:{id:'battle-ai',select:({context})=>{observer.push(context);return {action:{kind:'move',index:0},handoff:{turn:1}};}}
  },{seat:0});
}
test('three distinct agents exchange only accepted own and opponent-public facts',()=>{
  const seen=[],p=setup(seen);
  const d=p.select('draft',draft(),{round:1});
  assert.equal(seen[0].schema,PHASE_PIPELINE_SCHEMA);
  assert.equal(seen[0].current.draft,null);
  assert.throws(()=>{seen[0].agents.draft='changed';},TypeError);
  p.accepted('draft',draft(),d);
  const s=p.select('selection',selection(),{round:1});
  assert.equal(seen[1].handoff.draft.goal,'defensive');
  assert.equal(seen[1].current.draft.roster[0].moves[0],'thunderbolt');
  assert.ok(!JSON.stringify(seen[1]).includes('SECRET'));
  p.accepted('selection',selection(),s);
  const a=p.select('battle',battle(),{round:1});
  assert.deepEqual(seen[2].current.selection.ids,['t0','t1','t2']);
  assert.deepEqual(seen[2].current.selection.opponentPublicSpecies,['public-mon']);
  assert.equal(seen[2].handoff.selection.lead,'t0');
  p.accepted('battle',battle(),a);
  const state=p.snapshot();
  assert.deepEqual(state.current.battle.lastPublicOpponent,{speciesId:'public-mon',hpPercent:75});
  assert.ok(!JSON.stringify(state).includes('hiddenAbility'));
  p.roundEnd(1,0);
  assert.equal(p.snapshot().completed[0].result.winner,0);
  const f=p.finish({completed:true,winner:0});
  assert.deepEqual(f,{draft:null,selection:null,battle:null});
});
test('no unaccepted decision or note leaks; seat and order are enforced',()=>{
  const p=setup([]);const d=p.select('draft',draft(),{round:1});
  assert.throws(()=>p.select('selection',selection(),{round:1}),/Invalid phase selection/);
  assert.throws(()=>p.accepted('draft',draft(),['invalid']),/No matching/);
  assert.equal(p.snapshot().handoff.draft,null);
  p.accepted('draft',draft(),d);
  assert.throws(()=>p.select('battle',battle(),{round:1}),/out of order/);
  assert.throws(()=>p.roundEnd(1,0),/Invalid/);
  assert.throws(()=>createPhasePipeline({draft:()=>[],selection:()=>[],battle:()=>({kind:'move'})},{seat:2}),/Expected/);
});
test('handoff is bounded, JSON-only and cannot overwrite reserved properties',()=>{
  const p=createPhasePipeline({
    draft:{id:'d',select:()=>({action:team.map(t=>t.instanceId),handoff:{payload:'x'.repeat(4098)}})},
    selection:()=>[],battle:()=>({kind:'move',index:0})
  },{seat:1});
  assert.throws(()=>p.select('draft',draft(),{round:1}),/4096/);
  assert.equal(p.snapshot().handoff.draft,null);
});
test('seat-local continuity crosses rounds without sharing opponents hidden data',()=>{
  const first=setup([]),second=setup([]);
  const d=first.select('draft',draft(),{round:1});
  first.accepted('draft',draft(),d);
  const s=first.select('selection',selection(),{round:1});
  first.accepted('selection',selection(),s);
  const a=first.select('battle',battle(),{round:1});
  first.accepted('battle',battle(),a);
  first.roundEnd(1,0);
  const d2=first.select('draft',draft(),{round:2});
  assert.equal(first.snapshot().completed[0].battle.decisions,1);
  assert.equal(first.snapshot().handoff.battle.turn,1);
  assert.equal(second.snapshot().completed.length,0);
  first.accepted('draft',draft(),d2);
});
