import test from 'node:test';
import assert from 'node:assert/strict';
import {rnadLegalActions,validateRnadLegalActions,projectRnadObservation,
  RNAD_OBSERVATION_SCHEMA} from '../src/ml/legal-action-adapter.js';
import {createRnadTrajectoryCollector,RNAD_TRAJECTORY_SCHEMA}
  from '../src/ml/trajectory-collector.js';
const team=Array.from({length:6},(_,i)=>({
  instanceId:'p'+i,speciesId:'species'+i,speciesClauseId:'species'+i,
  setId:'set'+i,cost:1,itemId:'item'+i,
  moves:['tackle'],types:['normal'],hp:100,maxHP:100
}));
function collector(seat=0){return createRnadTrajectoryCollector({
  experimentId:'stage0',matchId:'m1',seed:7,seat,
  policyVersion:'RNAD-001@initial',opponentPolicyVersion:'baseline@frozen',
  sourceSha:'0123456789abcdef',engineSnapshotSha:'abcdef0123456789'
});}
test('legal phase vocabulary: ordered 60 selections, valid draft and battle only',()=>{
  const draft={phase:'draft',self:{fresh:team,carry:[],credits:12}};
  const draftChoices=rnadLegalActions(draft);
  assert.deepEqual(draftChoices,[team.map(p=>p.instanceId)]);
  const selection={phase:'selection',self:{team}};
  const choices=rnadLegalActions(selection);
  assert.equal(choices.length,60);
  assert.ok(choices.some(x=>x.join(',')==='p1,p0,p2'));
  assert.ok(choices.some(x=>x.join(',')==='p0,p1,p2'));
  const battle={requestType:'turn',self:{team:[{hp:100,maxHP:100,
    moves:[{id:'tackle',disabled:false}]}],active:0,confirmed:false,
    canSwitch:false,canMega:false}};
  assert.deepEqual(rnadLegalActions(battle),[{kind:'move',index:0}]);
  assert.throws(()=>validateRnadLegalActions([
    {kind:'move',index:0},{kind:'move',index:0}]),/Duplicate/);
  assert.throws(()=>validateRnadLegalActions([]),/length/);
});
test('actor projection is invariant to changes in opponent unknown fixed set',()=>{
  const view={requestType:'turn',turn:3,self:{team:[{speciesId:'own',
    hp:100,maxHP:100,moves:[{id:'tackle',disabled:false,pp:30}],
    abilityId:'own-ability',itemId:'own-item'}],
    active:0,canMega:true,canSwitch:false,confirmed:false},
    opponent:{speciesId:'public',hpPercent:73,types:['water'],
      status:'par',publicObservation:{appearanceId:'foo',
        usedMoves:['watergun'],observedItemIds:['leftovers'],
        hiddenAbility:'SECRET'},
      hiddenSet:{moves:['secretmove'],item:'secretitem'},hiddenAbility:'SECRET'},
    field:{weather:'rain',privateRngState:123},internalCheckpoint:'SECRET'};
  const first=projectRnadObservation(view,{seat:0});
  const altered=structuredClone(view);
  altered.opponent.hiddenSet={moves:['other'],item:'other'};
  altered.opponent.hiddenAbility='OTHER';
  altered.internalCheckpoint='OTHER';
  altered.field.privateRngState=999;
  altered.opponent.publicObservation.hiddenAbility='OTHER';
  const second=projectRnadObservation(altered,{seat:0});
  assert.deepEqual(first,second);
  assert.equal(first.schema,RNAD_OBSERVATION_SCHEMA);
  assert.ok(JSON.stringify(first).includes('own-ability'));
  assert.ok(!JSON.stringify(first).includes('SECRET'));
  assert.throws(()=>projectRnadObservation(view,{seat:1,context:{
    schema:'p-rl1024-phase-pipeline-v1',seat:0}}),/seat-scoped/);
});
test('only authoritative commit records a decision; rejection discards proposal',()=>{
  const c=collector();
  const observation=projectRnadObservation({requestType:'turn',turn:1,
    self:{team:[{hp:1,maxHP:1,moves:[{id:'tackle',disabled:false}]}],
      active:0,canSwitch:false,canMega:false,confirmed:false},
    opponent:{hpPercent:100}},{seat:0});
  const actions=[{kind:'move',index:0}];
  const args={phase:'battle',round:1,observation,legalActions:actions,
    action:actions[0],behaviorLogProbability:0,
    referenceLogProbability:0,predictedValue:0};
  c.propose(args);
  assert.equal(c.stats().accepted_decisions,0);
  assert.throws(()=>c.commit({kind:'switch',index:0}),/No matching/);
  assert.throws(()=>c.finish({completed:true,winner:0,score:[4,2]}),/Expected/);
  c.reject();
  assert.equal(c.stats().accepted_decisions,0);
  assert.throws(()=>c.propose({...args,action:{kind:'switch',index:0}}),/not in/);
  c.propose(args);
  assert.throws(()=>c.commit(actions[0]),/authority.*receipt/i);
  assert.equal(c.stats().accepted_decisions,0);
  assert.throws(()=>c.finish({completed:true,winner:0,score:[4,2]}),/Expected/);
  c.reject();
  assert.equal(c.stats().pending,false);
});
test('opponent seat receives negative terminal reward and cannot cross seats',()=>{
  const c=collector(1);
  const obs=projectRnadObservation({phase:'selection',round:1,
    self:{team},opponent:{species:['public']}},{seat:1});
  const action=team.slice(0,3).map(p=>p.instanceId);
  c.propose({phase:'selection',round:1,observation:obs,
    legalActions:rnadLegalActions({phase:'selection',self:{team}}),action,
    behaviorLogProbability:Math.log(1/60),
    referenceLogProbability:Math.log(1/60),predictedValue:0});
  // A second seat cannot self-certify an accepted action or terminal result.
  assert.throws(()=>c.commit(action,Object.freeze({})),/authority.*receipt/i);
  assert.equal(c.stats().accepted_decisions,0);
  c.reject();
  assert.throws(()=>c.finish({completed:true,winner:0,score:[4,3]},Object.freeze({})),/authority.*receipt/i);
});

test('untrusted cross-round handoff and draft history cannot reach actor',()=>{
  const view={phase:'draft',self:{fresh:team,carry:[],credits:12},
    history:[{opponent:{setId:'SECRET',itemId:'SECRET',moves:['SECRET']}}]};
  const context={schema:'p-rl1024-phase-pipeline-v1',seat:0,round:2,
    current:{opponent:{moves:['SECRET']}},
    completed:[{battle:{lastPublicOpponent:{hiddenSet:{moves:['SECRET']}}}}],
    handoff:{battle:{opponentItem:'SECRET'}}};
  const projected=projectRnadObservation(view,{seat:0,context});
  assert.ok(!JSON.stringify(projected).includes('SECRET'));
  assert.deepEqual(projected.context,{round:2});
  assert.ok(!Object.hasOwn(projected,'history'));
  const changed=structuredClone(view);
  changed.history[0].opponent.moves=['DIFFERENT'];
  const changedContext=structuredClone(context);
  changedContext.handoff.battle.opponentItem='DIFFERENT';
  assert.deepEqual(projected,projectRnadObservation(changed,{seat:0,context:changedContext}));
});
test('battle decisions fail closed on empty legal set',()=>{
  const view={requestType:'turn',self:{team:[],active:0,confirmed:false,
    canSwitch:false,canMega:false}};
  assert.throws(()=>rnadLegalActions(view),/Invalid public battle decision view|length/);
  const wait={requestType:'wait',phase:'wait',self:{team:[{hp:100}],active:0,confirmed:true}};
  assert.deepEqual(rnadLegalActions(wait),[]);
});
test('contradictory completed Match score cannot finish',()=>{
  const c=collector();
  assert.throws(()=>c.finish({completed:true,winner:0,score:[4,5]}),/four-win/);
  assert.throws(()=>c.finish({completed:true,winner:0,score:[4,4]}),/four-win/);
  assert.throws(()=>c.finish({completed:true,winner:0,score:[4,-1]}),/four-win/);
});

test('nested arena data and opponent public species are recursively allowlisted',()=>{
  const view={phase:'turn',requestType:'turn',
    self:{team:[{speciesId:'own',hp:100,maxHP:100,moves:[]}],active:0},
    opponent:{speciesId:'visible',hpPercent:90},
    field:{weather:'rain',sideEffects:[[{id:'stealthrock',layers:1,hiddenSet:'SECRET'}],[]]}};
  const projected=projectRnadObservation(view,{seat:0});
  assert.equal(projected.phase,'battle');
  assert.deepEqual(projected.field.sideEffects,[[{id:'stealthrock',layers:1}],[]]);
  assert.ok(!JSON.stringify(projected).includes('SECRET'));
  assert.throws(()=>projectRnadObservation({...view,self:{team:{hidden:'SECRET'}}},{seat:0}),
    /Missing own battle team/);
  const phase={phase:'selection',self:{team},opponent:{species:[{secret:'SECRET'}]}};
  assert.throws(()=>projectRnadObservation(phase,{seat:0}),/Invalid public opponent species/);
});

test('opponent stage and reveal lists never serialize a nested private set',()=>{
  const view={requestType:'turn',self:{team:[{speciesId:'own',hp:10,maxHP:10,
    moves:[{id:'tackle',disabled:false}]}],active:0,canSwitch:false,canMega:false},
    opponent:{speciesId:'public',hpPercent:60,
      stages:{atk:2,hiddenSet:{moves:['SECRET']}},
      publicObservation:{appearanceId:'visible',usedMoves:['tackle'],
        moves:['SECRET'],itemIds:['SECRET'],abilityIds:['SECRET']}}};
  const x=projectRnadObservation(view,{seat:0});
  assert.equal(x.opponent.stages.atk,2);
  assert.ok(!JSON.stringify(x).includes('SECRET'));
  const changed=structuredClone(view);
  changed.opponent.stages.hiddenSet={moves:['OTHER']};
  changed.opponent.publicObservation.moves=['OTHER'];
  assert.deepEqual(x,projectRnadObservation(changed,{seat:0}));
});
