// 20 pre-registered policies x 1024 independent seed-clusters each, both seats.
// Simultaneous finite-sample Hoeffding intervals protect all 20 AQ comparisons;
// 190 pairwise differences are also controlled familywise. Never promote models.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {weightsSha256 as AQ_BASE_SHA} from '../../models/aq-base.mjs';
import {OVERNIGHT_EVAL_SCHEMA,TRAINING_SOURCE_SHA,
  ORIGINAL_OVERNIGHT_RUN,FIRST_HELDOUT_SEED,PAIRED_SEEDS_PER_PART,
  PARTS_PER_MODEL,validatePartRows} from './evaluate-overnight-20-vs-aq.mjs';

export const OVERNIGHT_AGG_SCHEMA='rnad-i5-overnight-20-real-aq-pre-registered-aggregate-v1';
const fail=(msg)=>{throw new Error('Invalid overnight heldout evidence: '+msg)};
const requireOk=(cond,msg)=>{if(!cond)fail(msg)};
const sha=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const commit=x=>typeof x==='string'&&/^[a-f0-9]{40}$/.test(x);
const id=x=>x.seed+':'+x.candidateSeat;
const rounds=x=>x.rounds;

export function aggregateOvernightResults(reports){
  requireOk(Array.isArray(reports)&&reports.length===20*PARTS_PER_MODEL,
    'must include all 40 immutable model partitions');
  const found=new Map(),sourceSet=new Set(),engineSet=new Set();
  for(const r of reports){
    requireOk(r?.schema===OVERNIGHT_EVAL_SCHEMA&&
      Number.isInteger(r.training_shard)&&r.training_shard>=0&&r.training_shard<20&&
      Number.isInteger(r.part)&&r.part>=0&&r.part<PARTS_PER_MODEL&&
      r.mode===(r.training_shard%4===3?'selfplay':'league')&&
      r.training_source_sha===TRAINING_SOURCE_SHA&&
      r.original_training_run_id===ORIGINAL_OVERNIGHT_RUN&&
      r.aq_base_model_sha256===AQ_BASE_SHA&&
      r.benchmark==='AQ-Base-001 exact integrated strategy-match'&&
      r.policy==='frozen-greedy-neural-all-anti-loop-v1'&&
      commit(r.evaluation_source_sha)&&commit(r.engine_sha)&&
      sha(r.model_sha256)&&sha(r.checkpoint_sha256)&&
      r.seed_start===FIRST_HELDOUT_SEED+r.part*PAIRED_SEEDS_PER_PART&&
      r.independent_seed_pairs===PAIRED_SEEDS_PER_PART&&
      r.matches===2*PAIRED_SEEDS_PER_PART&&
      r.learner_updates===0&&r.training_during_evaluation===false&&
      r.selected_on_this_holdout===false&&r.champion_promoted===false&&
      Number.isInteger(r.trained_matches)&&r.trained_matches>0&&
      Number.isInteger(r.optimizer_updates)&&r.optimizer_updates>0&&
      [1,4].includes(r.updates_per_match)&&
      r.optimizer_updates===r.trained_matches*r.updates_per_match&&
      typeof r.origin==='string'&&r.origin.length>0,
      'invalid source-bound outcome or nonfrozen policy');
    requireOk(validatePartRows(r.rows,{
      seedStart:r.seed_start,pairs:r.independent_seed_pairs})===r.wins&&
      r.win_rate===r.wins/r.matches,'win count or authoritative outcome mismatch');
    sourceSet.add(r.evaluation_source_sha);engineSet.add(r.engine_sha);
    const k=r.training_shard+':'+r.part;
    requireOk(!found.has(k),'duplicate model/partition');
    found.set(k,r);
  }
  requireOk(sourceSet.size===1&&engineSet.size===1,
    'mixed evaluation source or frozen engine');
  const seedCount=PAIRED_SEEDS_PER_PART*PARTS_PER_MODEL;
  const ranked=[];
  for(let shard=0;shard<20;shard++){
    const rs=[0,1].map(part=>{
      const r=found.get(shard+':'+part);
      requireOk(r,'missing required model partition');
      return r;
    });
    const a=rs[0],b=rs[1];
    for(const field of ['model_sha256','checkpoint_sha256','mode','origin',
      'updates_per_match','trained_matches','optimizer_updates'])
      requireOk(a[field]===b[field],'model/checkpoint changed between heldout parts');
    const rows=rs.flatMap(x=>x.rows);
    const wins=validatePartRows(rows,{seedStart:FIRST_HELDOUT_SEED,pairs:seedCount});
    const scorePerSeed=Array.from({length:seedCount},(_,i)=>
      (Number(rows[2*i].won)+Number(rows[2*i+1].won))/2);
    const winRate=wins/(2*seedCount);
    const error=Math.sqrt(Math.log(2*20/.05)/(2*seedCount));
    ranked.push({
      shard,mode:a.mode,origin:a.origin,
      updates_per_match:a.updates_per_match,
      trained_matches:a.trained_matches,
      optimizer_updates:a.optimizer_updates,
      model_sha256:a.model_sha256,checkpoint_sha256:a.checkpoint_sha256,
      independent_seed_clusters:seedCount,
      complete_four_win_matches:2*seedCount,
      wins,win_rate:winRate,
      simultaneous_95pct_aq_win_rate_bounds:[
        Math.max(0,winRate-error),Math.min(1,winRate+error)],
      statistically_beats_aq_familywise_95:winRate-error>.5,
      scores:scorePerSeed
    });
  }
  // All 190 differences on same seed/seat clusters; Bonferroni familywise 95%.
  // One independent observation per seed is the average of two swapped seats.
  // Differences in [-1,1]; 2 is the finite-sample range (not 1).
  const pairwise=[];
  const comparisonCount=20*19/2;
  const diffHalf=Math.sqrt(2*Math.log(2*comparisonCount/.05)/seedCount);
  for(let a=0;a<20;a++)for(let b=a+1;b<20;b++){
    const A=ranked[a],B=ranked[b];
    const difference=A.scores.reduce((total,x,i)=>
      total+x-B.scores[i],0)/seedCount;
    pairwise.push({shard_a:a,shard_b:b,
      paired_win_rate_difference_a_minus_b:difference,
      simultaneous_95pct_bounds:[
        Math.max(-1,difference-diffHalf),Math.min(1,difference+diffHalf)],
      a_beats_b_familywise_95:difference-diffHalf>0,
      b_beats_a_familywise_95:difference+diffHalf<0});
  }
  const descriptor=ranked.map(({scores,...r})=>r)
    .sort((a,b)=>b.win_rate-a.win_rate||a.shard-b.shard);
  const best=descriptor[0];
  const confirmedWinner=pairwise.every(p=>
    (p.shard_a===best.shard&&p.a_beats_b_familywise_95)||
    (p.shard_b===best.shard&&p.b_beats_a_familywise_95))?
      best.shard:null;
  return {
    schema:OVERNIGHT_AGG_SCHEMA,
    original_training_run_id:ORIGINAL_OVERNIGHT_RUN,
    original_training_source_sha:TRAINING_SOURCE_SHA,
    evaluation_source_sha:[...sourceSet][0],
    engine_sha:[...engineSet][0],
    aq_base_model_sha256:AQ_BASE_SHA,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    models_expected:20,models_verified:20,
    independent_paired_seed_clusters_per_model:seedCount,
    total_complete_four_win_evaluation_matches:20*2*seedCount,
    heldout_first_seed:FIRST_HELDOUT_SEED,
    heldout_last_seed:FIRST_HELDOUT_SEED+seedCount-1,
    familywise_alpha:.05,aq_comparisons:20,pairwise_comparisons:comparisonCount,
    aq_win_rate_simultaneous_95_halfwidth:
      Math.sqrt(Math.log(2*20/.05)/(2*seedCount)),
    model_pairwise_simultaneous_95_halfwidth:diffHalf,
    highest_point_estimate_shard:best.shard,
    statistically_unique_best_shard_familywise_95:confirmedWinner,
    performance_compared_with_training_initial_model:false,
    causal_ablation_of_one_vs_four_updates:false,
    model_selected_on_this_holdout:false,
    champion_promoted:false,main_merged:false,
    independent_review_p1_resolved:false,
    ranking:descriptor,pairwise};
}
export function overnightSummaryMd(x){
  const out=[
    '# I5 overnight R-NaD: 20 frozen models vs original AQ-Base',
    '',
    'This is an actual authoritative complete-four-win Match study, NOT teacher Top1.',
    `Source pinned: \`${x.original_training_source_sha}\`.`,
    `Independent identical seed clusters per model: ${x.independent_paired_seed_clusters_per_model}`,
    `Total complete 4-win Matches: ${x.total_complete_four_win_evaluation_matches}`,
    `Simultaneous 95% AQ margin: ±${(100*x.aq_win_rate_simultaneous_95_halfwidth).toFixed(2)} percentage points (20-way Bonferroni Hoeffding).`,
    `Simultaneous 95% model-model margin: ±${(100*x.model_pairwise_simultaneous_95_halfwidth).toFixed(2)} percentage points (190 paired comparisons).`,
    '',
    `Highest observed (not necessarily best): shard ${x.highest_point_estimate_shard}`,
    `Familywise confirmed unique winner: ${x.statistically_unique_best_shard_familywise_95??'NONE'}`,
    '',
    '| Shard | Mode | Origin | Updates/Match | Train matches | AQ wins / 2048 | AQ win rate | Familywise 95% CI | >AQ significant? |',
    '|---:|---|---|---:|---:|---:|---:|---|---|'];
  for(const s of x.ranking)
    out.push(`| ${s.shard} | ${s.mode} | ${s.origin} | ${s.updates_per_match} | ${s.trained_matches} | ${s.wins} | ${(100*s.win_rate).toFixed(2)}% | [${(100*s.simultaneous_95pct_aq_win_rate_bounds[0]).toFixed(2)}%, ${(100*s.simultaneous_95pct_aq_win_rate_bounds[1]).toFixed(2)}%] | ${s.statistically_beats_aq_familywise_95?'yes':'no'} | `);
  out.push('','All 20 models were evaluated on the same previously unused seed range and both seats.', 'Cross-model outcomes are correlated by seed and are compared as paired seed blocks.',
    'Different feature families / initialization / opponent curricula confound causal training-strategy comparisons.',
    'These 20 models are not averaged or merged. No training baseline was evaluated in this study.',
    'No Champion promotion, formal three-phase gate, or independent review clearance.');
  return out.join('\n')+'\n';
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  if(process.argv.length!==4)
    throw new Error('Usage: node aggregate-overnight-20-vs-aq.mjs EVIDENCE_DIRECTORY OUTPUT_DIRECTORY');
  const src=process.argv[2],dest=process.argv[3];
  const inputs=[];
  for(let shard=0;shard<20;shard++)for(let part=0;part<PARTS_PER_MODEL;part++)
    inputs.push(JSON.parse(readFileSync(join(src,
      `rnad-i5-overnight-aq-s${shard}-p${part}`,'study.json'),'utf8')));
  const result=aggregateOvernightResults(inputs);
  mkdirSync(dest,{recursive:true});
  writeFileSync(join(dest,'REPORT.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  writeFileSync(join(dest,'REPORT.md'),overnightSummaryMd(result),{flag:'wx'});
  console.log(overnightSummaryMd(result));
}
