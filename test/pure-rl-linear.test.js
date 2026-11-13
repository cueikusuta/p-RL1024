import test from 'node:test';
import assert from 'node:assert/strict';
import {SeededRandom} from '../src/battle.js';
import {createPureRlModel,createPureRlSession,legalPureRlActions,
  pureRlFeatures,pureRlValue,rankPureRlActions,validatePureRlModel}
  from '../src/ai/pure-rl-linear.js';
import {runStrategyMatch} from '../scripts/dev/simulator/strategy-match.js';

const view=()=>({
  requestType:'turn',turn:2,field:{weather:null,terrain:null,trickRoomTurns:0},
  self:{active:0,canSwitch:true,canMega:true,confirmed:false,
    needsSwitch:false,team:[
      {hp:50,maxHP:100,status:'',speciesId:'pikachu',types:['Electric'],
        moves:[{id:'tackle',disabled:false},{id:'recover',disabled:true},
          {id:'revivalblessing',disabled:false}]},
      {hp:80,maxHP:100,status:'',speciesId:'blastoise',types:['Water'],moves:[]},
      {hp:0,maxHP:100,status:'',speciesId:'gengar',types:['Ghost'],moves:[]}
    ]},
  opponent:{hpPercent:64,status:'',remaining:2,speciesId:'snorlax',
    types:['Normal']}
});

test('I975 legal actions cover move, Mega, revival target and living switch',()=>{
  const actions=legalPureRlActions(view());
  assert.deepEqual(actions,[
    {kind:'move',index:0},{kind:'move',index:0,mega:true},
    {kind:'move',index:2,targetIndex:2},
    {kind:'move',index:2,targetIndex:2,mega:true},
    {kind:'switch',index:1}
  ]);
  const v=view();v.requestType='replacement';v.self.canMega=false;
  v.self.team[0].hp=0; // Authoritative replacement follows a faint.
  assert.deepEqual(legalPureRlActions(v),[{kind:'switch',index:1}]);
  v.requestType='wait';
  assert.deepEqual(legalPureRlActions(v),[]);
  v.requestType='turn';v.self.team[0].hp=50;v.self.canSwitch=false;
  v.self.team[0].moves.forEach(m=>{m.disabled=true;});
  assert.deepEqual(legalPureRlActions(v),[{kind:'move',index:'struggle'}]);
});

test('I975 features ignore injected opponent-private and v2-E fields',()=>{
  const a=view(),b=structuredClone(a);
  b.opponent.hiddenSelectedTeam=['p1','p2'];
  b.opponent.unrevealedMoves=['thunderbolt'];
  b.opponent.hiddenItem='choiceband';
  b.v2Scores=[900,-900];b.futureWinner=1;
  const action=legalPureRlActions(a)[0];
  assert.deepEqual(pureRlFeatures(a,action),pureRlFeatures(b,action));
  assert.equal(rankPureRlActions(createPureRlModel(),a).length,5);
  assert.throws(()=>pureRlFeatures(a,{kind:'move',index:1}),/not in/);
  assert.throws(()=>legalPureRlActions({}),/decision view/);
});

test('I975 accepted-only true-online SARSA learns terminal outcome exactly',()=>{
  const model=createPureRlModel();
  const v=view(),chosen={kind:'move',index:0};
  const features=pureRlFeatures(v,chosen);
  const session=createPureRlSession(model,{train:true,alpha:0.05,
    lambda:0.8,gamma:1});
  assert.deepEqual(session.select(v),chosen);
  assert.equal(Object.keys(model.weights).length,0,
    'select must not update before authoritative acceptance');
  session.commit();
  assert.equal(session.stats().updates,0,
    'first action has no previous same-seat decision');
  const done=session.finish(1);
  assert.equal(done.decisions,1);
  assert.equal(done.updates,1);
  for(const [k,x] of Object.entries(features))
    assert.ok(Math.abs(model.weights[k]-0.05*x)<1e-10,k);
  assert.ok(pureRlValue(model,features)>0);
  assert.equal(validatePureRlModel(model),model);
  assert.throws(()=>session.select(v),/not ready/);
  assert.throws(()=>session.finish(1),/finish/);
});

test('I975 two-step true-online Dutch traces and nonzero qOld match an independent oracle',()=>{
  const alpha=0.04,gamma=0.92,lambda=0.65;
  const state0=view(),state1=view();
  state1.opponent.hpPercent=11;state1.turn=7;
  const action={kind:'move',index:0};
  const f0=pureRlFeatures(state0,action),f1=pureRlFeatures(state1,action);
  const initial={bias:0.07};
  const dot=(w,x)=>Object.entries(x).reduce((sum,[key,v])=>
    sum+(w[key]??0)*v,0);
  const keys=new Set([...Object.keys(f0),...Object.keys(f1)]);
  const q0=dot(initial,f0),q1=dot(initial,f1);
  assert.ok(q1!==0);
  const delta0=gamma*q1-q0;
  const w1={...initial};
  for(const key of keys)
    w1[key]=(w1[key]??0)+alpha*delta0*(f0[key]??0);
  const afterQ=dot(w1,f1),correction=afterQ-q1;
  assert.ok(Math.abs(correction)>1e-8,'non-zero true-online qOld correction');
  const d=1-alpha*gamma*lambda*dot(f0,f1);
  const expected={...w1};
  for(const key of keys){
    const x1=f1[key]??0;
    const e=gamma*lambda*(f0[key]??0)+d*x1;
    expected[key]=(expected[key]??0)+alpha*(1-afterQ+correction)*e-
      alpha*correction*x1;
  }
  const model=createPureRlModel();model.weights={...initial};
  const session=createPureRlSession(model,{alpha,gamma,lambda,train:true});
  assert.deepEqual(session.select(state0),action);
  session.commit();
  assert.deepEqual(session.select(state1),action);
  session.commit();
  session.finish(1);
  assert.equal(session.stats().updates,2);
  for(const key of keys)
    assert.ok(Math.abs((model.weights[key]??0)-(expected[key]??0))<1e-9,
      'incorrect trace or qOld for '+key);
});

test('I975 independent policy never requires v2-E score/weights to play',()=>{
  function once(seed){
    const student=createPureRlSession(createPureRlModel(),{
      epsilon:1,rng:new SeededRandom(seed+10),train:false});
    const result=runStrategyMatch({seed,strategyMode:'p2',maxRounds:1,
      policies:['pure-rl-linear','basic'],
      pureRlSessions:[student,null]});
    assert.ok(result.actionCounts[0]>0);
    assert.equal(result.pureRlStats[0].decisions,result.actionCounts[0]);
    assert.equal(result.pureRlStats[0].updates,0);
    assert.equal(result.pureRlStats[0].train,false);
    return {score:result.score,roundWinners:result.roundWinners,
      actions:result.actionCounts,rounds:result.rounds};
  }
  assert.deepEqual(once(9755),once(9755));
});

test('I975 pure RL session contracts fail closed before or after acceptance',()=>{
  const model=createPureRlModel();
  const s=createPureRlSession(model,{train:true});
  assert.throws(()=>s.commit(),/No accepted/);
  assert.throws(()=>s.finish(0),/finish/);
  s.select(view());
  assert.throws(()=>s.select(view()),/not ready/);
  assert.throws(()=>s.finish(1),/finish/);
  s.commit();
  assert.throws(()=>createPureRlSession(model,{epsilon:1}),/requires a seeded RNG|^/);
  const invalid=createPureRlModel();invalid.weights.bias=Infinity;
  assert.throws(()=>validatePureRlModel(invalid),/Invalid/);
});
