import test from 'node:test';
import assert from 'node:assert/strict';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha,stage0FixedPolicy}
  from '../scripts/dev/simulator/rnad-match.js';
import provenance from '../provenance/manifest.json' with {type:'json'};
import {encodeRnadThreePhaseDecision} from '../src/ml/rnad-phase-features.js';

const base={seed:22115,matchId:'three-phase-22115',
  experimentId:'RNAD-002-contract',
  sourceSha:verifiedRunningSourceSha(),
  engineSnapshotSha:provenance.sourceCommit};
const safePolicy=ctx=>ctx.observation.phase==='battle'?
  stage0FixedPolicy(ctx):
  {action:ctx.legalActions[0],behaviorLogProbability:0,
   referenceLogProbability:-Math.log(ctx.legalActions.length),
   predictedValue:0};

test('authoritative real 4-win Match records accepted draft, selection, battle in order',()=>{
  let called=[0,0,0];
  const result=runRnadAcceptanceSmoke({...base,
    phasePolicy:ctx=>{
      const {observation,legalActions,round,seat}=ctx;
      const stage=['draft','selection','battle'].indexOf(observation.phase);
      called[stage]++;
      assert.ok([0,1].includes(seat));
      assert.ok(round>=1);
      assert.deepEqual(Object.keys(ctx).sort(),
        ['legalActions','observation','rng','round','seat']);
      assert.equal(observation.seat,seat);
      assert.equal(observation.opponent?.internalCheckpoint,undefined);
      const features=encodeRnadThreePhaseDecision(observation,legalActions);
      assert.equal(features.state.length,32);
      assert.equal(features.actions.length,legalActions.length);
      return safePolicy(ctx);
    }});
  assert.equal(result.completed,true);
  assert.equal(result.score[result.winner],4);
  assert.ok(result.rounds>=4);
  assert.deepEqual(called.slice(0,2),[result.rounds*2,result.rounds*2]);
  assert.equal(called[2],result.actionCounts[0]+result.actionCounts[1]);
  const chronology=result.jointTimeline.map(({seat,index})=>
    result.trajectories[seat].decisions[index]);
  assert.equal(chronology.length,called.reduce((a,b)=>a+b,0));
  assert.deepEqual(chronology.slice(0,2).map(x=>x.phase),
    ['draft','draft']);
  assert.deepEqual(chronology.slice(2,4).map(x=>x.phase),
    ['selection','selection']);
  for(const seat of [0,1]){
    const trace=result.trajectories[seat];
    assert.equal(trace.accepted_decisions,
      result.actionCounts[seat]+result.rounds*2);
    assert.equal(trace.completed,true);
    assert.equal(trace.terminal_reward,undefined);
    assert.ok(trace.decisions.every(x=>x.terminal_reward===(seat===result.winner?1:-1)));
  }
});

test('three-phase illegal proposed draft candidate cannot forge accepted record',()=>{
  assert.throws(()=>runRnadAcceptanceSmoke({...base,
    maxRounds:1,phasePolicy:ctx=>ctx.observation.phase==='draft'?
      {...safePolicy(ctx),action:['forged-instance']}:
      safePolicy(ctx)}),/not in the exact legal set/);
});
