import test from 'node:test';
import assert from 'node:assert/strict';
import {createMatch} from '../scripts/dev/simulator/runner.js';
import {createPureRlModel,createPureRlSession,rankPureRlActions}
  from '../src/ai/pure-rl-linear.js';
import {legalPureRlTeams,rankPureRlPhaseActions}
  from '../src/ai/pure-rl-phases.js';

// These tests exercise the actual authoritative Match rejection boundary.
// A proposal is never a SARSA transition until the corresponding confirm*
// succeeds. No mock Match can silently "accept" an illegal command.
test('I975 rejected draft and selection do not add SARSA updates',()=>{
  const {match}=createMatch({seed:975313,mode:'control'});
  match.ready(0);match.ready(1);
  const ranker=(model,view)=>['draft','selection'].includes(view.phase)
    ?rankPureRlPhaseActions(model,view,{seat:0})
    :rankPureRlActions(model,view);
  const session=createPureRlSession(createPureRlModel(),{
    train:true,epsilon:0,ranker
  });
  const draftView=match.view(0,{includeBattle:false});
  const chosenTeam=session.select(draftView);
  assert.equal(chosenTeam.length,6);
  assert.throws(()=>match.confirmTeam(0,['nonexistent']),/Expected six team members/);
  assert.equal(session.stats().decisions,0);
  assert.equal(session.stats().updates,0);
  match.confirmTeam(0,chosenTeam);
  session.commit();
  assert.equal(session.stats().decisions,1);
  assert.equal(session.stats().updates,0);

  // The nonlearning second seat uses its own authoritative public offer.
  const opponentView=match.view(1,{includeBattle:false});
  const opponentTeam=legalPureRlTeams(opponentView)[0];
  match.confirmTeam(1,opponentTeam);
  const selectionView=match.view(0,{includeBattle:false});
  const selected=session.select(selectionView);
  assert.equal(selected.length,3);
  assert.throws(()=>match.confirmSelection(0,['not-an-instance']),
    /Expected three distinct members/);
  assert.equal(session.stats().decisions,1);
  assert.equal(session.stats().updates,0);
  match.confirmSelection(0,selected);
  session.commit();
  assert.equal(session.stats().decisions,2);
  assert.equal(session.stats().updates,1);
});

test('I975 control-mode pure draft disallows cost 8 even with ample credit',()=>{
  const pool=Array.from({length:7},(_,i)=>({
    instanceId:'p'+i,speciesId:'species'+i,
    speciesClauseId:'species'+i,
    cost:i===6?8:1,itemId:'item'+i,types:['Normal']
  }));
  const offer={phase:'draft',round:1,score:[0,0],
    self:{fresh:pool,carry:[],credits:30},opponent:{}};
  assert.deepEqual(legalPureRlTeams(offer),[
    ['p0','p1','p2','p3','p4','p5']
  ]);
  // Carry cost is zero for budget purposes, NOT an exemption from
  // the authority's 1..7 base-cost acceptance requirement.
  offer.self.fresh=pool.slice(0,6);
  offer.self.carry=[pool[6]];
  assert.deepEqual(legalPureRlTeams(offer),[
    ['p0','p1','p2','p3','p4','p5']
  ]);
});

test('I975 reading public seat 1 score cannot depend on injected secrets',async()=>{
  const {pureRlPhaseFeatures}=await import('../src/ai/pure-rl-phases.js');
  const roster=Array.from({length:6},(_,i)=>({
    instanceId:'q'+i,speciesId:'species'+i,
    cost:1,itemId:'item'+i,types:['Electric']
  }));
  const view={phase:'selection',round:3,score:[1,3],
    self:{credits:4,team:roster,carry:[],fresh:[]},
    opponent:{species:['公開種族'],selected:['hidden'],moves:['secret']}};
  const action=roster.slice(0,3).map(x=>x.instanceId);
  const a=pureRlPhaseFeatures(view,action,{seat:1});
  view.opponent.selected=['different'];
  view.opponent.moves=['unknown'];
  const b=pureRlPhaseFeatures(view,action,{seat:1});
  assert.deepEqual(a,b);
  assert.equal(a['our-score:3'],1);
  assert.equal(a['our-score:1'],undefined);
});
