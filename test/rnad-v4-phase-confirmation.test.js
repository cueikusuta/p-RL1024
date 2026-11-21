import test from 'node:test';
import assert from 'node:assert/strict';
import {aggregateV4PairedConfirmation}
  from '../scripts/rnad/aggregate-v4-phase-confirm.mjs';
import {COMPOSITE_SCHEMA}
  from '../scripts/rnad/evaluate-v4-composite-vs-aq.mjs';
import {weightsSha256} from '../models/aq-base.mjs';
import provenance from '../provenance/manifest.json' with {type:'json'};
function arm(seed,mode,allWin=false){
  const rows=[];
  for(let i=0;i<4;i++)for(const seat of [0,1]){
    const won=allWin||seat===0;
    rows.push({seed:seed+i,candidateSeat:seat,won,
      score:won?(seat===0?[4,1]:[1,4]):[4,1],
      rounds:6,accepted_decisions:60});
  }
  const wins=rows.filter(x=>x.won).length;
  return {schema:COMPOSITE_SCHEMA,source_sha:'a'.repeat(40),
    engine_sha:provenance.sourceCommit,aq_base_model_sha256:weightsSha256,
    v4_model_sha256:'b'.repeat(64),v3_model_sha256:'c'.repeat(64),
    mode,policy:'frozen-greedy-v4-v3-anti-loop-v1',
    seed,pairs:4,matches:8,wins,win_rate:wins/8,
    champion_promoted:false,rows};
}
const shard=(seed)=>({
  baseline:arm(seed,'all-v3'),
  candidate:arm(seed,'selection-threequarter-v4',true)
});
test('locked 75-percent v4 selection aggregates independent paired seed blocks',()=>{
  const result=aggregateV4PairedConfirmation([shard(10),shard(14)],{
    seedStart:10,seedCount:8});
  assert.equal(result.matches_per_arm,16);
  assert.equal(result.independent_seeds,8);
  assert.equal(result.selection_weights.v4,.75);
  assert.equal(result.candidate_wins,16);
  assert.equal(result.baseline_wins,8);
  assert.equal(result.delta,.5);
  assert.equal(result.champion_promoted,false);
});
test('reject any missing, duplicated, substituted or reused holdout seats',()=>{
  const good=[shard(10),shard(14)];
  for(const bad of [
    [good[0]],
    [good[0],good[0]],
    [good[0],{...good[1],candidate:arm(14,'selection-half-v4',true)}],
    [good[0],{...good[1],candidate:{...good[1].candidate,
      source_sha:'f'.repeat(40)}}],
  ])assert.throws(()=>aggregateV4PairedConfirmation(bad,{
    seedStart:10,seedCount:8}));
});
