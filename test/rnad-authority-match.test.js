import test from 'node:test';
import assert from 'node:assert/strict';
import {runRnadAcceptanceSmoke,stage0FixedPolicy,verifiedRunningSourceSha}
  from '../scripts/dev/simulator/rnad-match.js';

const opts={
  seed:22003,matchId:'test-22003',experimentId:'contract',
  sourceSha:verifiedRunningSourceSha(),
  engineSnapshotSha:'ea81165993110ee9fb3a09dcb97c6708c0d7397b'
};
test('authoritative Match records only post-accepted actions for BOTH seats',()=>{
  // A real Battle with simultaneous turn submissions, no fake engine stub.
  const run=()=>runRnadAcceptanceSmoke({...opts,maxRounds:1});
  const result=run();
  assert.equal(result.rounds,1);
  assert.equal(result.completed,false);
  assert.equal(result.trajectories,null);
  for(const seat of [0,1]){
    assert.ok(result.actionCounts[seat]>0);
    assert.equal(result.agentStats[seat].battle.accepted_decisions,
      result.actionCounts[seat]);
    assert.equal(result.agentStats[seat].battle.pending,false);
  }
  assert.deepEqual(result,run()); // own-seat RNG and authority determinism
});
test('illegal policy proposal is rejected: no fallback to another action',()=>{
  assert.throws(()=>runRnadAcceptanceSmoke({...opts,maxRounds:1,
    policy:()=>({action:{kind:'move',index:999},
      behaviorLogProbability:0,referenceLogProbability:0,predictedValue:0})
  }),/not in the exact legal set/);
});
test('actor policy cannot receive an authority commit callback or raw Match',()=>{
  let seen=false;
  runRnadAcceptanceSmoke({...opts,maxRounds:1,policy:payload=>{
    seen=true;
    assert.deepEqual(Object.keys(payload).sort(),
      ['legalActions','observation','rng','round','seat']);
    assert.ok(!Object.hasOwn(payload,'commit'));
    assert.ok(!Object.hasOwn(payload,'accepted'));
    assert.ok(!Object.hasOwn(payload,'match'));
    assert.equal(payload.observation.phase,'battle');
    assert.ok(!Object.hasOwn(payload.observation,'request'));
    assert.ok(!Object.hasOwn(payload.observation,'internalCheckpoint'));
    return stage0FixedPolicy(payload);
  }});
  assert.equal(seen,true);
});

test('provenance mismatch rejects data before a simulation can start',()=>{
  assert.throws(()=>runRnadAcceptanceSmoke({...opts,
    engineSnapshotSha:'0'.repeat(40)}),/Invalid RNAD smoke provenance or policy/);
});
