// Registered matched-seed initial vs post-NeuRD outcomes. All 20 lanes, four frozen
// initial policies, original AQ authority. One 2-seat seed-cluster is one independent unit.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {aggregateOvernightResults} from './aggregate-overnight-20-vs-aq.mjs';
import {BASELINE_SCHEMA,BASELINE_ORIGINS} from './evaluate-overnight-initial-vs-aq.mjs';
import {FIRST_HELDOUT_SEED,PAIRED_SEEDS_PER_PART,
  validatePartRows} from './evaluate-overnight-20-vs-aq.mjs';

export const PREPOST_SCHEMA='rnad-i5-overnight-pre-post-matched-paired-v1';
const requireOk=(ok,reason)=>{if(!ok)throw new Error('Invalid matched pre/post evidence: '+reason);};
export function analyzePrepost({postReports,preReports}){
  const post=aggregateOvernightResults(postReports);
  requireOk(Array.isArray(preReports)&&preReports.length===8,
    'four origins each with two frozen pre-training partitions');
  const baselines=new Map();
  for(const r of preReports){
    requireOk(r?.schema===BASELINE_SCHEMA&&BASELINE_ORIGINS.includes(r.origin)&&
      [0,1].includes(r.part)&&r.original_training_run_id===post.original_training_run_id&&
      r.aq_base_model_sha256===post.aq_base_model_sha256&&
      r.benchmark===post.benchmark&&
      r.seed_start===FIRST_HELDOUT_SEED+r.part*PAIRED_SEEDS_PER_PART&&
      r.independent_seed_pairs===PAIRED_SEEDS_PER_PART&&
      r.matches===2*PAIRED_SEEDS_PER_PART&&
      r.additional_training===false&&r.champion_promoted===false&&
      /^[0-9a-f]{64}$/.test(r.model_sha256)&&
      /^[0-9a-f]{64}$/.test(r.source_model_sha256),
      'incorrect original baseline provenance');
    requireOk(validatePartRows(r.rows,{
      seedStart:r.seed_start,pairs:PAIRED_SEEDS_PER_PART})===r.wins,
      'baseline incomplete, duplicated or unpaired');
    const k=r.origin+':'+r.part;
    requireOk(!baselines.has(k),'duplicate initial origin/part');
    baselines.set(k,r);
  }
  const scores=new Map();
  for(const origin of BASELINE_ORIGINS){
    const a=baselines.get(origin+':0'),b=baselines.get(origin+':1');
    requireOk(a&&b&&a.model_sha256===b.model_sha256&&
      a.source_model_sha256===b.source_model_sha256,
      'baseline missing or changed between identical evaluation seeds');
    const rows=[...a.rows,...b.rows];
    validatePartRows(rows,{seedStart:FIRST_HELDOUT_SEED,pairs:1024});
    const wins=rows.filter(x=>x.won).length;
    scores.set(origin,{origin,model_sha256:a.model_sha256,
      source_model_sha256:a.source_model_sha256,
      wins,win_rate:wins/2048,
      bySeed:Array.from({length:1024},(_,i)=>
        (Number(rows[2*i].won)+Number(rows[2*i+1].won))/2)});
  }
  const ranked=[],alpha=.05;
  const half=Math.sqrt(2*Math.log(2*20/alpha)/1024); // difference ∈ [-1,1]
  for(const model of post.ranking){
    const base=scores.get(model.origin);
    requireOk(base&&base.model_sha256!==model.model_sha256,
      'untrained baseline missing or identical final model');
    const rs=postReports.filter(r=>r.training_shard===model.shard)
      .sort((a,b)=>a.part-b.part);
    requireOk(rs.length===2&&rs[0].part===0&&rs[1].part===1,
      'missing exact post seed schedule');
    const rows=rs.flatMap(r=>r.rows);
    validatePartRows(rows,{seedStart:FIRST_HELDOUT_SEED,pairs:1024});
    const diff=rows.reduce((sum,row,i)=>sum+
      (Number(row.won)-Number(
         (baselines.get(model.origin+':'+Math.floor((i/2)/PAIRED_SEEDS_PER_PART))
           .rows[(Math.floor(i/2)%PAIRED_SEEDS_PER_PART)*2+(i%2)]).won)),0)/2048;
    requireOk(Math.abs(diff-(model.win_rate-base.win_rate))<1e-10,
      'matched seed and seat difference inconsistent with frozen win counts');
    ranked.push({
      shard:model.shard,origin:model.origin,mode:model.mode,
      updates_per_match:model.updates_per_match,
      trained_matches:model.trained_matches,
      baseline_model_sha256:base.model_sha256,
      trained_model_sha256:model.model_sha256,
      baseline_aggregate_wins:base.wins,post_aggregate_wins:model.wins,
      baseline_rate:base.win_rate,post_rate:model.win_rate,
      paired_win_rate_gain:diff,
      simultaneous_95pct_gain_bounds:[
        Math.max(-1,diff-half),Math.min(1,diff+half)],
      significant_improvement_20way_95:diff-half>0,
      significant_degradation_20way_95:diff+half<0
    });
  }
  return {
    schema:PREPOST_SCHEMA,post_model_run_id:post.original_training_run_id,
    post_actual_match_evaluation_source_sha:post.evaluation_source_sha,
    matched_actual_benchmark:post.benchmark,
    frozen_aq_sha256:post.aq_base_model_sha256,
    independent_seed_clusters:1024,seat_swapped:true,
    model_count:20,unique_initial_policies:4,
    reused_paired_post_full_matches:40960,
    newly_played_paired_pre_full_matches:8192,
    familywise_alpha:alpha,
    simultaneous_95pct_gain_halfwidth:half,
    confirmed_positive:ranked.filter(x=>x.significant_improvement_20way_95).length,
    confirmed_negative:ranked.filter(x=>x.significant_degradation_20way_95).length,
    uncertain:ranked.filter(x=>!x.significant_improvement_20way_95&&
      !x.significant_degradation_20way_95).length,
    formal_league_gate_passed:false,champion_promoted:false,
    baseline_rates:[...scores.values()].map(({bySeed,...s})=>s),
    rows:ranked.sort((a,b)=>b.paired_win_rate_gain-a.paired_win_rate_gain)
  };
}
export function prepostMarkdown(x){
  const lines=['# I5 pre/post overnight R-NaD real AQ paired evaluation','',
    'All 20 final policies, four source-verified initial policies.',
    `Same ${x.independent_seed_clusters} seed clusters and both seats; 20-model familywise 95% Hoeffding margin ±${(x.simultaneous_95pct_gain_halfwidth*100).toFixed(2)}pt`,
    `Improved: ${x.confirmed_positive}; degraded: ${x.confirmed_negative}; unresolved: ${x.uncertain}`,'',
    '| Shard | Origin | Updates | Match trained | AQ initial | AQ final | Gain | Simultaneous 95% CI | Decision |',
    '|---:|---|---:|---:|---:|---:|---:|---|---|'];
  for(const r of x.rows)
    lines.push(`| ${r.shard} | ${r.origin} | ${r.updates_per_match} | ${r.trained_matches} | ${(r.baseline_rate*100).toFixed(2)}% | ${(r.post_rate*100).toFixed(2)}% | ${(r.paired_win_rate_gain*100).toFixed(2)}pt | [${(r.simultaneous_95pct_gain_bounds[0]*100).toFixed(2)}, ${(r.simultaneous_95pct_gain_bounds[1]*100).toFixed(2)}] | ${r.significant_improvement_20way_95?'up':r.significant_degradation_20way_95?'down':'unresolved'} | `);
  lines.push('','Research-only. No post-training model selection is validated on this same seed set; initial-vs-final is observational for training effect and holds all training identities fixed. Different opponent curricula, sources and training seeds confound cross-arm causal attribution.','');
  return lines.join('\n');
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  if(process.argv.length!==5)throw new Error('Usage: ... BASELINE-DIR POST-DIR REPORT-DIR');
  const [baseDir,postDir,out]=process.argv.slice(2);
  const pres=BASELINE_ORIGINS.flatMap(origin=>[0,1].map(part=>
    JSON.parse(readFileSync(join(baseDir,`initial-${origin}-p${part}`,'study.json')))));
  const posts=Array.from({length:20},(_,shard)=>[0,1].map(part=>
    JSON.parse(readFileSync(join(postDir,`rnad-i5-overnight-aq-s${shard}-p${part}`,
      'study.json'))))).flat();
  const result=analyzePrepost({preReports:pres,postReports:posts});
  mkdirSync(out,{recursive:true});
  writeFileSync(join(out,'REPORT.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  writeFileSync(join(out,'REPORT.md'),prepostMarkdown(result),{flag:'wx'});
  console.log(prepostMarkdown(result));
}
