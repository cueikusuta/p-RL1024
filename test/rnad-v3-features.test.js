import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeRnadThreePhaseDecisionV2}
  from '../src/ml/rnad-phase-features-v2.js';
import {encodeRnadThreePhaseDecisionV3,
  RNAD_THREE_PHASE_FEATURE_SCHEMA_V3}
  from '../src/ml/rnad-phase-features-v3.js';
import {projectRnadObservation,RNAD_OBSERVATION_SCHEMA}
  from '../src/ml/legal-action-adapter.js';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha}
  from '../scripts/dev/simulator/rnad-match.js';
import provenance from '../provenance/manifest.json' with {type:'json'};

const own=Array.from({length:6},(_,i)=>({
  instanceId:'p'+i,setId:'s'+i,speciesId:'species'+i,
  cost:1,itemId:'item'+i,abilityId:'ability'+i,
  types:i%2?['Water']:['Fire'],moves:i%2?['surf']:['flamethrower'],
  stats:{hp:100,atk:100,def:100,spa:105,spd:95,spe:100},
  hp:100,maxHP:100
}));
const raw=()=>({phase:'selection',round:2,score:[0,0],budget:13,grant:10,
  self:{credits:13,team:structuredClone(own)},
  opponent:{species:['unknown'],privateMoves:['private-attack'],
    unobservedSet:{itemId:'secret'}}});
const legal=[['p0','p2','p4'],['p1','p3','p5']];
test('v3 preserves v2 coordinates with explicit versioned 48/32 features',()=>{
  const obs=projectRnadObservation(raw(),{seat:0});
  const v2=encodeRnadThreePhaseDecisionV2(obs,legal);
  const v3=encodeRnadThreePhaseDecisionV3(obs,legal);
  assert.equal(v3.schema,RNAD_THREE_PHASE_FEATURE_SCHEMA_V3);
  assert.equal(v3.state.length,48);
  assert.ok(v3.actions.every(x=>x.length===32&&x.every(Number.isFinite)));
  assert.deepEqual(v3.state.slice(0,32),v2.state);
  assert.deepEqual(v3.actions.map(a=>a.slice(0,16)),v2.actions);
  assert.notDeepEqual(v3.actions[0].slice(16),v3.actions[1].slice(16));
});
test('v3 does not consume private opponent fields',()=>{
  const a=projectRnadObservation(raw(),{seat:0});
  const other=raw();other.opponent.privateMoves=['changed-secret'];
  other.opponent.unobservedSet={abilityId:'totally-new'};
  const b=projectRnadObservation(other,{seat:0});
  assert.deepEqual(encodeRnadThreePhaseDecisionV3(a,legal),
                   encodeRnadThreePhaseDecisionV3(b,legal));
});
test('v3 changes on trusted history but rejects malformed injected history',()=>{
  const obs={schema:RNAD_OBSERVATION_SCHEMA,phase:'battle',seat:0,
    turn:4,requestType:'turn',self:{team:structuredClone(own.slice(0,3)),
    active:0,canSwitch:true,canMega:false,remaining:3},
    opponent:{types:['Grass'],hpPercent:100,remaining:3},
    field:{}};
  const actions=[{kind:'move',index:0},{kind:'switch',index:1}];
  const base=encodeRnadThreePhaseDecisionV3(obs,actions);
  obs.acceptedHistory={lastKind:'switch',consecutiveSwitches:2,
    lastSwitchTarget:1,lastActive:0};
  const modified=encodeRnadThreePhaseDecisionV3(obs,actions);
  assert.deepEqual(base.state.slice(0,32),modified.state.slice(0,32));
  assert.notDeepEqual(base.state.slice(32),modified.state.slice(32));
  obs.acceptedHistory.consecutiveSwitches=99999;
  assert.throws(()=>encodeRnadThreePhaseDecisionV3(obs,actions),
    /accepted-history/);
});
test('only authoritative completed Match receipts advance per-seat history',()=>{
  const result=runRnadAcceptanceSmoke({
    seed:22110,matchId:'v3-history-22110',
    sourceSha:verifiedRunningSourceSha(),
    engineSnapshotSha:provenance.sourceCommit,includeAcceptedHistory:true
  });
  assert.equal(result.completed,true);
  for(const seat of [0,1]){
    const rows=result.trajectories[seat].decisions;
    assert.ok(rows.length>1);
    assert.equal(rows[0].observation.acceptedHistory.lastKind,'none');
    for(let i=1;i<rows.length;i++){
      const prior=rows[i-1],current=rows[i];
      if(prior.phase==='battle'&&prior.observation.requestType==='turn'){
        assert.equal(current.observation.acceptedHistory.lastKind,
          prior.accepted_action.kind==='switch'?'switch':'move');
      }
      assert.equal(current.observation.seat,seat);
    }
  }
});

test('v3 represents authoritative Struggle without fabricating a normal move',()=>{
  const observation={
    schema:RNAD_OBSERVATION_SCHEMA,phase:'battle',seat:0,
    turn:3,requestType:'turn',
    self:{team:structuredClone(own.slice(0,3)),active:0,
      remaining:3,canSwitch:true,canMega:false},
    opponent:{types:['Grass'],hpPercent:100,remaining:3},field:{}
  };
  const result=encodeRnadThreePhaseDecisionV3(observation,[
    {kind:'move',index:'struggle'}, {kind:'switch',index:1}
  ]);
  assert.equal(result.actions.length,2);
  assert.equal(result.actions[0].length,32);
  assert.ok(result.actions[0].every(Number.isFinite));
});
