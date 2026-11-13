import test from 'node:test';
import assert from 'node:assert/strict';
import {SeededRandom} from '../src/battle.js';
import {createPureRlModel,createPureRlSession,rankPureRlActions}
  from '../src/ai/pure-rl-linear.js';
import {legalPureRlTeams,legalPureRlSelections,pureRlPhaseFeatures,
  rankPureRlPhaseActions} from '../src/ai/pure-rl-phases.js';
import {runStrategyMatch} from '../scripts/dev/simulator/strategy-match.js';

const offered=Array.from({length:7},(_,i)=>({
  instanceId:'s'+i,setId:'set'+i,speciesId:'species'+i,
  speciesClauseId:'species'+i,cost:i===6?4:1,
  itemId:'item'+i,types:i%2?['Fire']:['Water']
}));
function draft(){
  return {phase:'draft',round:1,score:[0,0],
    self:{credits:6,carry:[],fresh:structuredClone(offered)},
    opponent:{teamConfirmed:false,unrevealedMoves:['private']}};
}
function selection(){
  return {phase:'selection',round:1,score:[0,0],
    self:{credits:0,team:structuredClone(offered.slice(0,6)),
      fresh:[],carry:[]},
    opponent:{species:['somePublicSpecies'],opponentSecretSets:['hidden']}};
}
const ranker=(model,view)=>['draft','selection'].includes(view.phase)
  ?rankPureRlPhaseActions(model,view):rankPureRlActions(model,view);

test('I975 draft enumerates all and only affordable nonduplicate six-member teams',()=>{
  const v=draft();
  assert.deepEqual(legalPureRlTeams(v),[['s0','s1','s2','s3','s4','s5']]);
  v.self.credits=10;
  assert.equal(legalPureRlTeams(v).length,7);
  v.self.fresh[6].speciesClauseId='species0';
  assert.equal(legalPureRlTeams(v).length,2);
  v.self.fresh[6].speciesClauseId='species6';
  v.self.fresh[6].itemId='item0';
  assert.equal(legalPureRlTeams(v).length,2);
  assert.throws(()=>pureRlPhaseFeatures(v,['s0','s1']),/Illegal/);
});

test('I975 secret trio has 60 legal lead-aware candidates',()=>{
  const v=selection();
  const all=legalPureRlSelections(v);
  assert.equal(all.length,60);
  assert.deepEqual(all[0],['s0','s1','s2']);
  const a=rankPureRlPhaseActions(createPureRlModel(),v);
  assert.equal(a.length,60);
  const b=structuredClone(v);
  b.opponent.opponentSecretSets=['different'];
  b.opponent.trueSelectedThree=['s4','s5','s6'];
  assert.deepEqual(pureRlPhaseFeatures(v,all[0]),pureRlPhaseFeatures(b,all[0]));
  v.score=[1,3];
  assert.equal(pureRlPhaseFeatures(v,all[0],{seat:0})['our-score:1'],1);
  assert.equal(pureRlPhaseFeatures(v,all[0],{seat:1})['our-score:3'],1);
  v.opponent.species=['ミュウツー'];
  const keys=Object.keys(pureRlPhaseFeatures(v,all[0]));
  assert.ok(keys.some(k=>k.startsWith('selection:opponent-public:unicode-')));
});

test('I975 one SARSA trajectory spans draft, selection and battle without teacher',()=>{
  const model=createPureRlModel();
  const agent=createPureRlSession(model,{train:true,alpha:0.04,
    epsilon:0,ranker});
  assert.deepEqual(agent.select(draft()),['s0','s1','s2','s3','s4','s5']);
  agent.commit();
  const team=agent.select(selection());
  assert.equal(team.length,3);
  agent.commit();
  assert.equal(agent.stats().updates,1,
    'next own-seat decision updates previous phase after accepted action');
  agent.finish(1);
  assert.equal(agent.stats().updates,2);
  assert.ok(Object.keys(model.weights).some(x=>x.startsWith('draft:')));
  assert.ok(Object.keys(model.weights).some(x=>x.startsWith('selection:')));
});

test('I975 full-phase research Match reaches real Battle with learned decisions',()=>{
  function sample(){
    const model=createPureRlModel();
    const s=createPureRlSession(model,{epsilon:1,
      rng:new SeededRandom(97555),ranker});
    const outcome=runStrategyMatch({seed:9756,strategyMode:'p2',
      maxRounds:1,policies:['pure-rl-linear','basic'],
      pureRlSessions:[s,null],pureRlAllPhases:true});
    assert.equal(outcome.rounds,1);
    assert.equal(outcome.pureRlStats[0].decisions,
      outcome.actionCounts[0]+2,'one accepted draft + selection per round');
    return {score:outcome.score,roundWinners:outcome.roundWinners,
      actionCounts:outcome.actionCounts};
  }
  assert.deepEqual(sample(),sample());
});
