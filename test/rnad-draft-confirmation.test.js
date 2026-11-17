import test from 'node:test';
import assert from 'node:assert/strict';
import {readConfirmationPlan,plannedSeeds,plannedShard,
  validateConfirmationPlan,validateConfirmationShard,
  aggregateConfirmationShards,empiricalBernsteinFixed95,
  CONFIRM_SHARD_SCHEMA} from '../scripts/rnad/draft-confirmation-protocol.mjs';

const PLAN='docs/research/rnad/DRAFT_CONFIRMATION_PLAN_2026-11-17.json';
const SOURCE='f'.repeat(40);
const {plan,plan_sha256:PLAN_SHA}=readConfirmationPlan(PLAN);
const twoWays=[0,1];
const six=['A','B','C','D','E','F'];
function fakeReport(spec,mode,win){
  const rows=[],accepted_draft_examples=[];
  for(const seed of spec.seeds)for(const seat of twoWays){
    const won=win;
    const score=won?(seat===0?[4,1]:[1,4]):(seat===0?[1,4]:[4,1]);
    rows.push({seed,seat,won,score,rounds:5,accepted_decisions:45});
    accepted_draft_examples.push({seed,seat,score,
      winner:won?seat:1-seat,
      picks:Array.from({length:5},(_,i)=>({round:i+1,credits:10,
        accepted_spent:9,legal_candidates:220,
        accepted:six,carried:[],offered:[]}))});
  }
  return {schema:'rnad-i5-deepsets-draft-original-aq-paired-screen-v1',
    benchmark:plan.benchmark,mode,seed:null,pairs:spec.seeds.length,
    explicit_seeds:spec.seeds,
    matches:2*spec.seeds.length,wins:rows.filter(r=>r.won).length,
    win_rate:rows.filter(r=>r.won).length/rows.length,
    source_sha:SOURCE,engine_sha:plan.engine_sha,
    aq_base_model_sha256:plan.aq_sha256,
    v3_model_sha256:plan.v3_sha256,v4_model_sha256:plan.v4_sha256,
    draft_model_sha256:plan.candidate_sha256,
    strength_established:false,champion_promoted:false,
    rows,accepted_draft_examples};
}
function shard(index,candidateWin=true,baselineWin=false){
  const s=plannedShard(plan,index);
  return {schema:CONFIRM_SHARD_SCHEMA,
    plan_sha256:PLAN_SHA,source_sha:SOURCE,shard_index:index,
    start:s.start,seeds:s.seeds,match_count:4*s.seeds.length,
    baseline:fakeReport(s,'all-v3',baselineWin),
    candidate:fakeReport(s,'draft-deepsets-soft',candidateWin)};
}
test('plan is single-model, fixed-N, fresh deterministic seeds (not two-seat pseudo-replication)',()=>{
  assert.equal(plan.seed_count,3072);
  assert.equal(plan.shard_size,64);
  assert.equal(plan.seed_count/plan.shard_size,48);
  assert.equal(plan.candidate_mode,'draft-deepsets-soft');
  assert.equal(plan.stop_rule,'fixed-n-no-early-look');
  const seeds=plannedSeeds(plan);
  assert.equal(seeds.length,new Set(seeds).size);
  assert.ok(seeds.every(Number.isSafeInteger));
  assert.deepEqual(plannedShard(plan,0).seeds,seeds.slice(0,64));
  assert.deepEqual(plannedShard(plan,47).seeds,seeds.slice(-64));
  assert.throws(()=>plannedShard(plan,48),/shard index/);
  assert.throws(()=>validateConfirmationPlan({...plan,seed_count:8}),/Invalid immutable/);
  assert.throws(()=>validateConfirmationPlan({...plan,alpha:0.25}),/Invalid immutable/);
});
test('distribution-free independent-seed confidence intervals are not usable as small-N winner checks',()=>{
  const low=empiricalBernsteinFixed95(Array(8).fill(.125),-1,1,.025);
  assert.ok(low.lower<=0);
  assert.ok(low.half_width>1);
  const positive=empiricalBernsteinFixed95(Array(3072).fill(.10),-1,1,.025);
  assert.ok(positive.lower>0);
  const flat=empiricalBernsteinFixed95(Array(3072).fill(.5),0,1,.025);
  assert.ok(flat.lower<.5);
  assert.throws(()=>empiricalBernsteinFixed95([.1,2],-1,1,.05));
});
test('accepts full 64-seed both-seat shard, rejects missing and forged authority records',()=>{
  const s=shard(0);
  validateConfirmationShard(s,plan,PLAN_SHA,SOURCE);
  assert.throws(()=>aggregateConfirmationShards([s],plan,PLAN_SHA,SOURCE),
    /Incomplete frozen fixed-N shard set/);
  const missing=structuredClone(s);
  missing.baseline.rows.pop();
  assert.throws(()=>validateConfirmationShard(missing,plan,PLAN_SHA,SOURCE));
  const duplicated=structuredClone(s);
  duplicated.candidate.rows[1].seat=0;
  assert.throws(()=>validateConfirmationShard(duplicated,plan,PLAN_SHA,SOURCE));
  const overBudget=structuredClone(s);
  overBudget.candidate.accepted_draft_examples[0].picks[1].accepted_spent=11;
  assert.throws(()=>validateConfirmationShard(overBudget,plan,PLAN_SHA,SOURCE));
  const swapped=structuredClone(s);
  swapped.baseline.aq_base_model_sha256='a'.repeat(64);
  assert.throws(()=>validateConfirmationShard(swapped,plan,PLAN_SHA,SOURCE));
  const source=structuredClone(s);
  source.source_sha='e'.repeat(40);
  assert.throws(()=>validateConfirmationShard(source,plan,PLAN_SHA,SOURCE));
});
test('full 48-shard fixed-N confidence review is possible; champion is never promoted',()=>{
  const all=Array.from({length:48},(_,i)=>shard(i));
  const report=aggregateConfirmationShards(all,plan,PLAN_SHA,SOURCE);
  assert.equal(report.independent_seed_blocks,3072);
  assert.equal(report.complete_four_win_matches,12288);
  assert.equal(report.paired.mean,1);
  assert.equal(report.aq_absolute.mean,1);
  assert.equal(report.eligible_for_independent_review,true);
  assert.equal(report.champion_promoted,false);
  assert.throws(()=>aggregateConfirmationShards([...all.slice(0,47),all[0]],
    plan,PLAN_SHA,SOURCE),/Duplicate shard index/);
  const negative=Array.from({length:48},(_,i)=>shard(i,false,true));
  const failed=aggregateConfirmationShards(negative,plan,PLAN_SHA,SOURCE);
  assert.equal(failed.eligible_for_independent_review,false);
  assert.equal(failed.paired_direction,'worse');
  assert.equal(failed.aq_absolute_direction,'below-50-percent');
  assert.equal(failed.verdict,'statistically-below-required-target');
  assert.equal(failed.champion_promoted,false);
});
