// Compare actual AQ-Base win rates to public-AQ teacher Top1 agreement OF SAME frozen policy.
// Correlation across 20 separately trained models is descriptive only; not a causal effect.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {aggregateOvernightResults} from './aggregate-overnight-20-vs-aq.mjs';
import {IMITATION_SCHEMA,FIRST_TEACHER_SEED,TEACHER_SEED_CLUSTERS}
  from './evaluate-overnight-teacher-top1.mjs';
export const IMITATION_STRENGTH_SCHEMA='rnad-i5-20-frozen-model-public-teacher-top1-vs-real-aq-winrate-v1';
const requireOk=(v,msg)=>{if(!v)throw new Error('Untrusted imitation-versus-strength evidence: '+msg);};
export function pearson(x,y){
  requireOk(x.length===y.length&&x.length>=2,'bad correlation input');
  const mx=x.reduce((a,b)=>a+b,0)/x.length,my=y.reduce((a,b)=>a+b,0)/y.length;
  let vx=0,vy=0,cov=0;
  for(let i=0;i<x.length;i++){const a=x[i]-mx,b=y[i]-my;vx+=a*a;vy+=b*b;cov+=a*b;}
  return vx>0&&vy>0?cov/Math.sqrt(vx*vy):null;
}
function averageRanks(values){
  const result=new Array(values.length);
  const sorted=values.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v);
  for(let j=0;j<sorted.length;){
    let k=j+1;while(k<sorted.length&&sorted[k].v===sorted[j].v)k++;
    const rank=(j+k-1)/2;
    for(let z=j;z<k;z++)result[sorted[z].i]=rank;
    j=k;
  }
  return result;
}
export function spearman(x,y){return pearson(averageRanks(x),averageRanks(y));}
export function summarizeTeacherAgreement(reports,postReports){
  const post=aggregateOvernightResults(postReports);
  requireOk(Array.isArray(reports)&&reports.length===20,'require all 20 frozen teacher evaluations');
  const byShard=new Map(),modelSourceSet=new Set();
  const phases=['draft','selection','battle'];
  const half=Math.sqrt(Math.log(2*60/.05)/(2*TEACHER_SEED_CLUSTERS));
  for(const r of reports){
    requireOk(r?.schema===IMITATION_SCHEMA&&
      Number.isInteger(r.shard)&&r.shard>=0&&r.shard<20&&
      r.independent_complete_teacher_matches===TEACHER_SEED_CLUSTERS&&
      r.pre_registered_seed_start===FIRST_TEACHER_SEED&&
      r.teacher_sha256===post.aq_base_model_sha256&&
      r.engine_sha===post.engine_sha&&
      r.teacher_examples_are_strictly_accepted_public_own_view===true&&
      r.teacher_top1_is_not_original_aq_full_match_policy===true&&
      r.training_during_evaluation===false&&r.champion_promoted===false&&
      Array.isArray(r.rows)&&r.rows.length===TEACHER_SEED_CLUSTERS,
      'bad authority or teacher phase provenance');
    requireOk(!byShard.has(r.shard),'duplicate training model phase study');
    const m=post.ranking.find(x=>x.shard===r.shard);
    requireOk(m&&m.model_sha256===r.model_sha256&&
      m.checkpoint_sha256===r.checkpoint_sha256&&
      m.trained_matches===r.trained_matches&&
      m.origin===r.origin&&m.mode===r.mode&&
      m.updates_per_match===r.updates_per_match,
      'teacher study policy does not match real AQ frozen-model study');
    requireOk(r.feature_schema===(r.mode==='selfplay'?
      'rnad-three-phase-features-v5':'rnad-three-phase-features-v3'),
      'feature family incompatible with teacher decisions');
    const seen=new Set(),perPhase={};
    for(const phase of phases)perPhase[phase]={correct:0,count:0,seedMean:0};
    for(const row of r.rows){
      requireOk(Number.isInteger(row.seed)&&row.seed>=FIRST_TEACHER_SEED&&
        row.seed<FIRST_TEACHER_SEED+TEACHER_SEED_CLUSTERS&&
        !seen.has(row.seed),'Missing or duplicated independent teacher Match');
      seen.add(row.seed);
      for(const phase of phases){
        const correct=row.correct?.[phase],count=row.counts?.[phase];
        requireOk(Number.isInteger(correct)&&Number.isInteger(count)&&
          count>=1&&correct>=0&&correct<=count,'bad accepted teacher Top1 count');
        perPhase[phase].correct+=correct;
        perPhase[phase].count+=count;
        perPhase[phase].seedMean+=correct/count;
      }
    }
    requireOk(seen.size===TEACHER_SEED_CLUSTERS,
      'not exactly 512 independent matched teacher Match seeds');
    const result={...m,seed_clusters:TEACHER_SEED_CLUSTERS,teacher_agreement:{}};
    for(const phase of phases){
      const p=perPhase[phase],rate=p.seedMean/TEACHER_SEED_CLUSTERS;
      result.teacher_agreement[phase]={
        accepted_decisions:p.count,matched_choices:p.correct,
        pooled_top1:p.correct/p.count,mean_seed_top1:rate,
        simultaneous_95pct_seed_cluster_bounds:[Math.max(0,rate-half),Math.min(1,rate+half)]
      };
    }
    modelSourceSet.add(r.teacher_sha256);
    byShard.set(r.shard,result);
  }
  requireOk(byShard.size===20&&modelSourceSet.size===1,'incomplete 20-model comparison');
  const models=[...byShard.values()].sort((a,b)=>a.shard-b.shard);
  const correlations={};
  for(const [group,pool] of [
    ['all20',models],['v3_league15',models.filter(x=>x.mode==='league')],
    ['v5_selfplay5',models.filter(x=>x.mode==='selfplay')]]){
    correlations[group]={n_models:pool.length,descriptive_only:true};
    for(const phase of phases){
      const teacher=pool.map(m=>m.teacher_agreement[phase].mean_seed_top1);
      const win=pool.map(m=>m.win_rate);
      correlations[group][phase]={
        spearman_point:spearman(teacher,win),
        pearson_point:pearson(teacher,win)};
    }
  }
  return {schema:IMITATION_STRENGTH_SCHEMA,
    original_training_run_id:post.original_training_run_id,
    strength_evaluation_source_sha:post.evaluation_source_sha,
    benchmark:'AQ-Base-001 true complete 4-win Match',
    teacher_reference:'AQ public accepted teacher, not original AQ exact battle policy',
    same_frozen_model_for_strength_and_imitation:true,
    models:20,independent_teacher_seed_clusters_per_model:TEACHER_SEED_CLUSTERS,
    real_aq_independent_seed_clusters_per_model:1024,
    separate_teacher_seed_block_not_reused_for_training_or_strength:true,
    teacher_phase_familywise_95_error_halfwidth:half,
    teacher_model_sha256:[...modelSourceSet][0],
    correlations_of_only_20_models_do_not_establish_causality_or_significance:true,
    model_selected_on_teacher_holdout:false,
    champion_promoted:false,formal_league_gate_passed:false,
    correlations,per_model:models};
}
export function strengthImitationMarkdown(x){
  const lines=['# I5: same frozen 20 R-NaD policies: teacher Top1 vs real AQ strength','',
    `Teacher decisions: real 4-win accepted public AQ teacher, NOT original AQ full-Match policy. ${x.independent_teacher_seed_clusters_per_model} independent teacher seeds.`,
    'The AQ strength values are reused from the independent 40,960-match study; no weight updates.',
    `All-model/all-phase simultaneous 95% Hoeffding Top1 margin: ±${(100*x.teacher_phase_familywise_95_error_halfwidth).toFixed(2)}pt over seed-cluster means.`,
    'Correlation is descriptive across just 20 heterogeneous models, NOT a causal result nor a proven correlation.','',
    '| Shard | Family | AQ win % | Draft public-teacher Top1 % | Selection % | Battle % |',
    '|---:|---|---:|---:|---:|---:|'];
  for(const m of [...x.per_model].sort((a,b)=>b.win_rate-a.win_rate))
    lines.push(`| ${m.shard} | ${m.mode} | ${(m.win_rate*100).toFixed(2)} | ${(m.teacher_agreement.draft.mean_seed_top1*100).toFixed(2)} | ${(m.teacher_agreement.selection.mean_seed_top1*100).toFixed(2)} | ${(m.teacher_agreement.battle.mean_seed_top1*100).toFixed(2)} |`);
  lines.push('','| Group | Models | Draft Spearman | Selection Spearman | Battle Spearman |',
    '|---|---:|---:|---:|---:|');
  for(const [n,d] of Object.entries(x.correlations))
    lines.push(`| ${n} | ${d.n_models} | ${d.draft.spearman_point?.toFixed(3)??'NA'} | ${d.selection.spearman_point?.toFixed(3)??'NA'} | ${d.battle.spearman_point?.toFixed(3)??'NA'} |`);
  lines.push('','Public teacher agreement is not identical to the full AQ benchmark policy. Both teacher Top1 and win rate must be validated independently; no champion promotion.','');
  return lines.join('\n');
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  if(process.argv.length!==5)throw Error('Usage: aggregate teacher results OUT and existing 40 AQ report directory');
  const [teacherDir,postDir,outDir]=process.argv.slice(2);
  const teacher=Array.from({length:20},(_,i)=>JSON.parse(
    readFileSync(join(teacherDir,`teacher-imitation-shard${i}`,'study.json'))));
  const post=Array.from({length:20},(_,i)=>[0,1].map(part=>
    JSON.parse(readFileSync(join(postDir,
      `rnad-i5-overnight-aq-s${i}-p${part}`,'study.json'))))).flat();
  const report=summarizeTeacherAgreement(teacher,post);
  mkdirSync(outDir,{recursive:true});
  writeFileSync(join(outDir,'REPORT.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  writeFileSync(join(outDir,'REPORT.md'),strengthImitationMarkdown(report),{flag:'wx'});
  console.log(strengthImitationMarkdown(report));
}
