import test from 'node:test';
import assert from 'node:assert/strict';
import {validatePartRows,FIRST_HELDOUT_SEED,OVERNIGHT_EVAL_SCHEMA,
  TRAINING_SOURCE_SHA,ORIGINAL_OVERNIGHT_RUN,PAIRED_SEEDS_PER_PART}
  from '../scripts/rnad/evaluate-overnight-20-vs-aq.mjs';
import {aggregateOvernightResults} from '../scripts/rnad/aggregate-overnight-20-vs-aq.mjs';
import {weightsSha256 as AQ_SHA} from '../models/aq-base.mjs';
import provenance from '../provenance/manifest.json' with {type:'json'};
const scores=(seedStart,n,shard)=>Array.from({length:n},(_,i)=>
  [0,1].map(seat=>{
    // Controlled outcome with no model-specific statistically significant edge.
    const won=(i+seat+shard)%2===0;
    return {seed:seedStart+i,candidateSeat:seat,won,
      score:seat===0?(won?[4,1]:[1,4]):(won?[1,4]:[4,1]),
      rounds:5,accepted_decisions:70};
  })).flat();
function make(shard,part){
  const mode=shard%4===3?'selfplay':'league';
  const seedStart=FIRST_HELDOUT_SEED+part*PAIRED_SEEDS_PER_PART;
  const rows=scores(seedStart,PAIRED_SEEDS_PER_PART,shard);
  const wins=rows.filter(x=>x.won).length;
  const matches=6000+shard,updatesPer=shard%2===0?4:1;
  return {schema:OVERNIGHT_EVAL_SCHEMA,
    evaluation_source_sha:'a'.repeat(40),engine_sha:provenance.sourceCommit,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    training_source_sha:TRAINING_SOURCE_SHA,
    original_training_run_id:ORIGINAL_OVERNIGHT_RUN,
    aq_base_model_sha256:AQ_SHA,
    training_shard:shard,part,mode,
    origin:mode==='selfplay'?'battle-focus256-v5':['base128','base256','c005'][shard%3],
    updates_per_match:updatesPer,trained_matches:matches,
    optimizer_updates:matches*updatesPer,model_sha256:(shard+1).toString(16).padStart(64,'0'),
    checkpoint_sha256:(shard+100).toString(16).padStart(64,'0'),
    policy:'frozen-greedy-neural-all-anti-loop-v1',
    seed_start:seedStart,independent_seed_pairs:PAIRED_SEEDS_PER_PART,
    matches:2*PAIRED_SEEDS_PER_PART,wins,win_rate:wins/(2*PAIRED_SEEDS_PER_PART),
    learner_updates:0,training_during_evaluation:false,selected_on_this_holdout:false,
    champion_promoted:false,rows};
}
test('20 source-bound independent complete Match models, 40 parts and conservative familywise CIs',()=>{
  const reports=Array.from({length:20},(_,s)=>[make(s,0),make(s,1)]).flat();
  const r=aggregateOvernightResults(reports);
  assert.equal(r.models_verified,20);
  assert.equal(r.total_complete_four_win_evaluation_matches,40960);
  assert.equal(r.independent_paired_seed_clusters_per_model,1024);
  assert.equal(r.pairwise.length,190);
  assert.equal(r.statistically_unique_best_shard_familywise_95,null);
  assert.equal(r.champion_promoted,false);
  assert.equal(r.model_selected_on_this_holdout,false);
  assert.ok(r.ranking.every(x=>x.win_rate===.5));
});
test('duplicate seed, missing swapped seat, forged outcome or altered source invalidates evaluation',()=>{
  const valid=make(0,0),copy=()=>structuredClone(valid);
  assert.equal(validatePartRows(valid.rows,{seedStart:valid.seed_start,
    pairs:PAIRED_SEEDS_PER_PART}),512);
  for(const mutation of [
    x=>{x.rows[0].candidateSeat=1;},
    x=>{x.rows.pop();},
    x=>{x.rows[1].score=[0,0];},
    x=>{x.rows[0].seed=FIRST_HELDOUT_SEED-1;}
  ]){
    const x=copy();mutation(x);
    assert.throws(()=>validatePartRows(x.rows,{
      seedStart:x.seed_start,pairs:PAIRED_SEEDS_PER_PART}));
  }
});
test('fail closed on missing/duplicate partitions, changed checkpoint or disguised non-4win outcome',()=>{
  const all=Array.from({length:20},(_,s)=>[make(s,0),make(s,1)]).flat();
  assert.throws(()=>aggregateOvernightResults(all.slice(1)),/Invalid overnight/);
  for(const mutation of [
    x=>{x[1].model_sha256='b'.repeat(64);},
    x=>{x[0].rows[0].score=[3,2];},
    x=>{x[0].training_during_evaluation=true;},
    x=>{x[3]=structuredClone(x[2]);},
    x=>{x[0].independent_seed_pairs=1;},
  ]){
    const a=structuredClone(all);mutation(a);
    assert.throws(()=>aggregateOvernightResults(a),/Invalid overnight/);
  }
});
