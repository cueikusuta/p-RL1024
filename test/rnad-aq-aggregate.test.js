import test from 'node:test';
import assert from 'node:assert/strict';
import {aggregateOriginalAQReports}
  from '../scripts/rnad/aggregate-original-aq.mjs';
import {FULL_AQ_EVALUATION_SCHEMA}
  from '../scripts/rnad/evaluate-vs-aq-full.mjs';
import {weightsSha256} from '../models/aq-base.mjs';

const fixedSha='a'.repeat(64);
const source='b'.repeat(40);
const engine='c'.repeat(40);
function shard(seed,pairs=1){
  const rows=[];
  for(let s=seed;s<seed+pairs;s++)for(const seat of [0,1])
    rows.push({seed:s,candidateSeat:seat,won:seat===0,
      score:seat===0?[4,2]:[4,1],rounds:6,accepted_decisions:70});
  return {schema:FULL_AQ_EVALUATION_SCHEMA,
    aq_base_model_sha256:weightsSha256,model_sha256:fixedSha,
    source_sha:source,engine_sha:engine,
    evaluation_policy:'greedy-no-training',phase_mode:'neural-all',
    switch_guard:'anti-loop-v1',seed,pairs,matches:2*pairs,
    wins:pairs,win_rate:.5,rows,champion_promoted:false};
}
test('real AQ aggregations use independent seed pairs and remain unpromoted',()=>{
  const report=aggregateOriginalAQReports([shard(10,2),shard(12,3)],{
    seedStart:10,seedCount:5});
  assert.equal(report.completed_matches,10);
  assert.equal(report.independent_paired_seeds,5);
  assert.equal(report.wins,5);
  assert.equal(report.win_rate,.5);
  assert.equal(report.eligible_for_independent_champion_review,false);
  assert.equal(report.champion_promoted,false);
  assert.ok(report.paired_hoeffding_95pct[0]<.5);
});
test('refuse missing, duplicated, forged results and mixed model identity',()=>{
  const valid=[shard(10,2),shard(12,2)];
  for(const altered of [
    [valid[0]],
    [valid[0],valid[0]],
    [valid[0],{...valid[1],model_sha256:'d'.repeat(64)}],
    [valid[0],{...valid[1],source_sha:'0'.repeat(40)}],
    [valid[0],{...valid[1],wins:3}],
    [valid[0],{...valid[1],rows:valid[1].rows.map((r,i)=>
      i===0?{...r,score:[0,4]}:r)}],
    [valid[0],{...valid[1],rows:valid[1].rows.map((r,i)=>
      i===0?{...r,candidateSeat:1}:r)}]
  ])assert.throws(()=>aggregateOriginalAQReports(altered,{
    seedStart:10,seedCount:4}));
});
test('reject invalid total schedule before reading data',()=>{
  assert.throws(()=>aggregateOriginalAQReports([shard(10)],{
    seedStart:0,seedCount:1}),/Invalid bounded/);
  assert.throws(()=>aggregateOriginalAQReports([shard(10)],{
    seedStart:10,seedCount:2049}),/Invalid bounded/);
});
