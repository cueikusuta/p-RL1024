import test from 'node:test';
import assert from 'node:assert/strict';
import {RnadSwitchLoopGuard,RNAD_SWITCH_GUARD}
  from '../scripts/rnad/switch-loop-guard.mjs';

const switchTo=index=>({kind:'switch',index});
const move=index=>({kind:'move',index});
function policy(){
  const calls=[];
  return {calls,choose({legalActions}){
    calls.push(structuredClone(legalActions));
    const a=legalActions.find(x=>x.kind==='switch')??legalActions[0];
    return {action:a,behaviorLogProbability:0,referenceLogProbability:-1,
      predictedValue:0};
  }};
}
const choose=(guard,{from=0,round=1,turn=1,requestType='turn',actions}={})=>
  guard.choose({seat:0,round,
    observation:{phase:'battle',requestType,turn,
      self:{active:from,team:[{hp:5},{hp:5},{hp:5}]}},
    legalActions:actions??[switchTo(1),switchTo(2),move(0)]
  });
const accepted=(guard,action,round=1)=>
  guard.accepted({seat:0,round,phase:'battle',action});

test('accepted only: two back-to-back voluntary switches then legal move; no reversal',()=>{
  const impl=policy(),g=new RnadSwitchLoopGuard(impl);
  const first=choose(g,{from:0,turn:1});
  assert.deepEqual(first.action,switchTo(1));
  assert.throws(()=>choose(g,{from:1,turn:2}),/not accepted/);
  assert.throws(()=>accepted(g,switchTo(2)),/Unaccepted/);
  assert.throws(()=>accepted(g,first.action,2),/Unaccepted/);
  accepted(g,first.action);
  const second=choose(g,{from:1,turn:2,
    actions:[switchTo(0),switchTo(2),move(0)]});
  assert.deepEqual(second.action,switchTo(2));
  assert.deepEqual(impl.calls[1],[switchTo(2),move(0)]);
  accepted(g,second.action);
  const third=choose(g,{from:2,turn:3,
    actions:[switchTo(0),switchTo(1),move(0)]});
  assert.deepEqual(third.action,move(0));
  assert.deepEqual(impl.calls[2],[move(0)]);
  accepted(g,third.action);
  // Once a move was accepted a tactically useful switch becomes legal again.
  const fourth=choose(g,{from:2,turn:4,
    actions:[switchTo(1),move(0)]});
  assert.deepEqual(fourth.action,switchTo(1));
  accepted(g,fourth.action);
});

test('mandatory replacement and no-move positions never have switches masked',()=>{
  const impl=policy(),g=new RnadSwitchLoopGuard(impl);
  const before=choose(g,{from:0,actions:[switchTo(1),move(0)]});
  accepted(g,before.action);
  const forced=choose(g,{from:1,turn:2,requestType:'replacement',
    actions:[switchTo(0),switchTo(2)]});
  assert.deepEqual(forced.action,switchTo(0));
  accepted(g,forced.action);
  const noMove=choose(g,{from:0,turn:3,actions:[switchTo(1)]});
  assert.deepEqual(noMove.action,switchTo(1));
  accepted(g,noMove.action);
  const pivot=choose(g,{from:1,turn:4,requestType:'pivot',
    actions:[switchTo(0),switchTo(2)]});
  assert.deepEqual(pivot.action,switchTo(0));
  accepted(g,pivot.action);
});

test('per-seat per-round reset and explicit guard version',()=>{
  assert.equal(RNAD_SWITCH_GUARD,'anti-loop-v1');
  const impl=policy(),g=new RnadSwitchLoopGuard(impl);
  const first=choose(g,{from:0,round:1,actions:[switchTo(1),move(0)]});
  accepted(g,first.action);
  const next=choose(g,{from:1,round:2,actions:[switchTo(0),move(0)]});
  assert.deepEqual(next.action,switchTo(0));
  assert.throws(()=>g.accepted({seat:1,round:2,phase:'battle',
    action:next.action}),/Unaccepted/);
  accepted(g,next.action,2);
  g.reset();
  const after=choose(g,{from:0,round:2,actions:[switchTo(1),move(0)]});
  assert.deepEqual(after.action,switchTo(1));
  accepted(g,after.action,2);
});

test('draft / selection are forwarded unmodified and only update after acceptance',()=>{
  const impl={choose({legalActions}){return {action:legalActions[0]};}};
  const g=new RnadSwitchLoopGuard(impl);
  const draft=['p0','p1','p2','p3','p4','p5'];
  const d=g.choose({seat:0,round:1,observation:{phase:'draft'},
    legalActions:[draft]});
  assert.deepEqual(d.action,draft);
  g.accepted({seat:0,round:1,phase:'draft',action:draft});
  const selection=['p0','p1','p2'];
  const s=g.choose({seat:0,round:1,observation:{phase:'selection'},
    legalActions:[selection]});
  assert.deepEqual(s.action,selection);
  g.accepted({seat:0,round:1,phase:'selection',action:selection});
});

test('reject illegal policy actions and mismatched active seat input',()=>{
  const g=new RnadSwitchLoopGuard({choose:()=>({action:switchTo(999)})});
  assert.throws(()=>choose(g,{actions:[move(0)]}),/nonlegal/);
  assert.throws(()=>g.accepted({seat:0,round:1,phase:'battle',
    action:move(0)}),/Unaccepted/);
  assert.throws(()=>new RnadSwitchLoopGuard(null),/needs a synchronous/);
  assert.throws(()=>g.choose({seat:4,round:1,observation:{phase:'battle'},
    legalActions:[move(0)]}),/Invalid guarded/);
});
