import test from 'node:test';
import assert from 'node:assert/strict';
import {RnadSwitchLoopGuard} from '../scripts/rnad/switch-loop-guard.mjs';
import {createRnadTrajectoryCollector} from '../src/ml/trajectory-collector.js';

const move=index=>({kind:'move',index});
const sw=index=>({kind:'switch',index});
const makeCollector=()=>createRnadTrajectoryCollector({
  experimentId:'anti-loop-train',matchId:'match-3',seed:3,seat:0,
  policyVersion:'RNAD-003-v2-anti-loop-v1',
  opponentPolicyVersion:'RNAD-003-v2-anti-loop-v1',
  sourceSha:'a'.repeat(40),engineSnapshotSha:'b'.repeat(40)
});
const observation={schema:'rnad-actor-observation-v1',phase:'battle',
  seat:0,requestType:'turn',turn:2,
  self:{active:1,team:[{hp:100},{hp:100},{hp:100}]}};
const base={
  phase:'battle',round:1,observation,
  legalActions:[sw(0),sw(2),move(0)],
  behaviorLogProbability:0,referenceLogProbability:0,predictedValue:0
};

test('authoritative full legal actions and guard support are retained separately',()=>{
  const c=makeCollector();
  c.propose({...base,policyEligibleActions:[sw(2),move(0)],action:sw(2)});
  assert.equal(c.stats().pending,true);
  assert.throws(()=>c.commit(sw(2),Object.freeze({})),/authorit.*receipt/i);
  c.reject();
  assert.equal(c.stats().pending,false);
});

test('guard cannot create a candidate, duplicate the subset or change accepted action',()=>{
  const examples=[
    {policyEligibleActions:[sw(99),move(0)],action:move(0)},
    {policyEligibleActions:[move(0),move(0)],action:move(0)},
    {policyEligibleActions:[move(0)],action:sw(2)},
    {policyEligibleActions:[],action:sw(2)},
  ];
  for(const x of examples){
    const c=makeCollector();
    assert.throws(()=>c.propose({...base,...x}),/Guard support|Duplicate|length/);
    assert.equal(c.stats().accepted_decisions,0);
  }
});

test('after an accepted first switch, the exact reduced support follows model',()=>{
  const model={
    choose({legalActions}){return {
      action:legalActions[0],
      behaviorLogProbability:-Math.log(legalActions.length),
      referenceLogProbability:-Math.log(legalActions.length),
      predictedValue:0
    }}
  };
  const guard=new RnadSwitchLoopGuard(model);
  const a=guard.choose({seat:0,round:1,
    observation:{...observation,self:{...observation.self,active:0},turn:1},
    legalActions:[sw(1),sw(2),move(0)]});
  assert.deepEqual(a.policyEligibleActions,[sw(1),sw(2),move(0)]);
  guard.accepted({seat:0,round:1,phase:'battle',action:a.action});
  const b=guard.choose({seat:0,round:1,observation,
    legalActions:base.legalActions});
  assert.deepEqual(b.policyEligibleActions,[sw(2),move(0)]);
  assert.ok(Math.abs(Math.exp(b.referenceLogProbability)-.5)<1e-12);
  const c=makeCollector();
  c.propose({...base,policyEligibleActions:b.policyEligibleActions,
    action:b.action,behaviorLogProbability:b.behaviorLogProbability,
    referenceLogProbability:b.referenceLogProbability});
  assert.equal(c.stats().pending,true);
  c.reject();
});
