import test from 'node:test';
import assert from 'node:assert/strict';
import {compareOriginalAQPaired}
  from '../scripts/rnad/compare-original-aq-paired.mjs';
import {FULL_AQ_EVALUATION_SCHEMA}
  from '../scripts/rnad/evaluate-vs-aq-full.mjs';
import {weightsSha256} from '../models/aq-base.mjs';
const make=(sha,winBoth=false)=>({
  schema:FULL_AQ_EVALUATION_SCHEMA,
  model_sha256:sha,aq_base_model_sha256:weightsSha256,
  source_sha:'a'.repeat(40),engine_sha:'b'.repeat(40),
  evaluation_policy:'greedy-no-training',
  phase_mode:'neural-all',switch_guard:'anti-loop-v1',
  seed:330001,pairs:4,matches:8,wins:winBoth?8:4,
  win_rate:winBoth?1:.5,champion_promoted:false,
  rows:Array.from({length:4},(_,i)=>[0,1].map(seat=>{
    const won=winBoth||seat===0;
    return {seed:330001+i,candidateSeat:seat,won,
      score:won?(seat===0?[4,1]:[1,4]):[4,1],
      rounds:6,accepted_decisions:70};
  })).flat()
});
test('old AQ report validator supports same-seed A/B without double counting seats',()=>{
  const b=make('c'.repeat(64)),c=make('d'.repeat(64),true);
  const r=compareOriginalAQPaired(b,c);
  assert.equal(r.independent_seeds,4);
  assert.equal(r.matches_per_model,8);
  assert.equal(r.paired_win_rate_gain,.5);
  assert.equal(r.gain_statistically_established,false);
  assert.equal(r.strength_established,false);
  assert.equal(r.champion_promoted,false);
});
test('paired original AQ report rejects source, seed, model or role mismatch',()=>{
  const b=make('c'.repeat(64)),c=make('d'.repeat(64),true);
  for(const bad of [
    {...c,source_sha:'e'.repeat(40)},
    {...c,seed:330011},
    {...c,model_sha256:b.model_sha256},
    {...c,rows:c.rows.map((v,i)=>i===0?{...v,candidateSeat:1}:v)},
    {...c,rows:c.rows.slice(1)},
  ])assert.throws(()=>compareOriginalAQPaired(b,bad));
});
