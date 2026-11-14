import test from 'node:test';
import assert from 'node:assert/strict';
import {runTabularRnad,softmax,regularizedZeroSumPayoff,exploitability}
  from '../scripts/rnad/toy-rnad.mjs';
const rps=[[[0,-1,1],[1,0,-1],[-1,1,0]]];
const pennies=[[[1,-1],[-1,1]]];
// P0 receives a private type (0 or 1); P1 does not know which.
// Simultaneous decisions: P1 must use one strategy over both payoff matrices.
const privateType=[[[1,-1],[-1,1]],[[-1,1],[1,-1]]];
const sum=xs=>xs.reduce((a,b)=>a+b,0);
test('regularized payoff is zero-sum with correct own/opponent log ratios',()=>{
  const args={payoff:0.7,eta:.2,actor0:.8,magnet0:.5,actor1:.25,magnet1:.5};
  const [p0,p1]=regularizedZeroSumPayoff(args);
  assert.ok(Math.abs((p0+p1))<1e-12);
  assert.ok(Math.abs(p0-(.7-.2*Math.log(.8/.5)+.2*Math.log(.25/.5)))<1e-12);
  assert.ok(p0<.7);
  assert.throws(()=>regularizedZeroSumPayoff({...args,magnet0:0}),/Invalid/);
});
test('RPS: magnet iterations reduce exact best-response exploitability',()=>{
  const r=runTabularRnad({games:rps,priors:[1]});
  assert.equal(r.magnetUpdates,10);
  assert.equal(r.innerUpdates,3500);
  assert.ok(r.initialExploitability>1.5);
  assert.ok(r.finalExploitability<.01,JSON.stringify(r.history));
  assert.ok(r.policy0[0].every(x=>x>.2&&x<.5));
  for(const p of [...r.policy0,r.policy1]){
    assert.ok(p.every(x=>Number.isFinite(x)&&x>0));
    assert.ok(Math.abs(sum(p)-1)<1e-10);
  }
});
test('Matching Pennies: mixed strategy rather than deterministic play',()=>{
  const r=runTabularRnad({games:pennies,priors:[1]});
  assert.ok(r.finalExploitability<.01);
  assert.ok(r.policy0[0].every(x=>x>.4&&x<.6));
  assert.ok(r.policy1.every(x=>x>.4&&x<.6));
});
test('private-type Bayesian game: one P1 strategy reduces exact exploitability',()=>{
  const r=runTabularRnad({games:privateType,priors:[.5,.5]});
  assert.equal(r.policy0.length,2);
  assert.equal(r.policy1.length,2); // P1 cannot condition on the hidden type
  assert.ok(r.initialExploitability>.5);
  assert.ok(r.finalExploitability<.01,
    'Expected reduction in an actual imperfect-information toy: '+r.finalExploitability);
  assert.equal(r.finalExploitability,exploitability(privateType,[.5,.5],
    r.policy0,r.policy1));
});
test('numerics reject invalid inputs and do not silently normalize illegal sets',()=>{
  assert.throws(()=>softmax([0,Infinity]),/finite/);
  assert.throws(()=>runTabularRnad({games:rps,priors:[.9]}),/prior/);
  assert.throws(()=>runTabularRnad({games:rps,priors:[1],
    innerSteps:0}),/bounded/);
});
