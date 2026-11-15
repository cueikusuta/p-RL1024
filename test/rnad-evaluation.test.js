import test from 'node:test';
import assert from 'node:assert/strict';
import {aggregateIndependentPaired,wilson} from '../scripts/rnad/aggregate-eval.mjs';
const fake=(seed=500000,won0=true,won1=false)=>({
  schema:'rnad-paired-full-match-eval-v1',
  model_sha256:'a'.repeat(64),engine_sha:'b'.repeat(40),
  opponent_id:'stage0-fixed-untrained',pairs:1,matches:2,
  rows:[{seed,candidateSeat:0,won:won0,score:won0?[4,1]:[1,4]},
        {seed,candidateSeat:1,won:won1,score:won1?[1,4]:[4,1]}]
});
test('Wilson CI for small evaluation cannot automatically promote champion',()=>{
  const e=aggregateIndependentPaired([fake()]);
  assert.equal(e.unique_paired_seeds,1);
  assert.equal(e.matches,2);
  assert.equal(e.champion_promoted,false);
  assert.equal(e.eligible_for_independent_champion_review,false);
  assert.ok(wilson(1,2)[0]<.5);
});
test('duplicate seeds, unmatched seat and mixed provenance fail closed',()=>{
  assert.throws(()=>aggregateIndependentPaired([fake(),fake()]),/Reused/);
  const changed=fake(700000);
  changed.model_sha256='c'.repeat(64);
  assert.throws(()=>aggregateIndependentPaired([fake(),changed]),/Mixed/);
  const corrupt=fake(700001);
  corrupt.rows[1].candidateSeat=0;
  assert.throws(()=>aggregateIndependentPaired([corrupt]),/Reused/);
});
test('one thousand independent paired matches meet count but not assumed strength',()=>{
  const reports=Array.from({length:500},(_,i)=>
    fake(200000+i, i%2===0, i%2===1));
  const result=aggregateIndependentPaired(reports);
  assert.equal(result.matches,1000);
  assert.equal(result.wins,500);
  assert.equal(result.eligible_for_independent_champion_review,false);
});
