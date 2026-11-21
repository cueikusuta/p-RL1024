import test from 'node:test';
import assert from 'node:assert/strict';
import {PersistentRNADInference,RNADMixedPolicy,argmaxRnad}
  from '../scripts/rnad/infer.mjs';

const dummy=logits=>Object.assign(Object.create(PersistentRNADInference.prototype),{
  featureSchema:'rnad-three-phase-features-v1',
  inferProjected:()=>({logits,value:.25}),reset:()=>{},
});

test('greedy action choice uses argmax and point-mass behavior probability',()=>{
  const model=dummy([-2,4,1]),reference=dummy([0,0,0]);
  const agent=new RNADMixedPolicy({
    model,magnet:reference,selectionMode:'greedy',epsilon:0});
  const chosen=agent.choose({observation:{phase:'draft'},
    legalActions:['a','b','c'],seat:0});
  assert.equal(chosen.action,'b');
  assert.equal(chosen.behaviorLogProbability,0);
  assert.ok(Math.abs(chosen.referenceLogProbability-Math.log(1/3))<1e-12);
  assert.equal(chosen.predictedValue,.25);
  assert.equal(argmaxRnad([-1,2,2]),1);
});
test('default sampling still performs stochastic exploration',()=>{
  const model=dummy([-2,4,1]),reference=dummy([0,0,0]);
  const policy=new RNADMixedPolicy({model,magnet:reference,epsilon:0});
  const picked=policy.choose({observation:{phase:'draft'},
    legalActions:['a','b','c'],seat:0,rng:{integer:()=>0}});
  assert.equal(picked.action,'a');
  assert.ok(picked.behaviorLogProbability<0);
  assert.throws(()=>new RNADMixedPolicy({
    model,magnet:reference,epsilon:.2,selectionMode:'greedy'}));
  assert.throws(()=>argmaxRnad([NaN]));
});
