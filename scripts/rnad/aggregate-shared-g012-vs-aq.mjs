// Strict paired heldout analysis for G0/G1/G2 complete four-win matches.
// Seed is the independent cluster; two seats must never be counted as two seeds.
import {readFileSync,writeFileSync,mkdirSync,readdirSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {SCHEMA,FIRST_SEED,SEEDS_PER_PART,PARTS} from './evaluate-shared-g012-vs-aq.mjs';
import {weightsSha256 as AQ_SHA} from '../../models/aq-base.mjs';
const ALPHA=.05;
const COMPARISONS=6;
const TOTAL=SEEDS_PER_PART*PARTS;
const assert=(condition,msg)=>{if(!condition)throw Error('G012 heldout evidence: '+msg);};
const avg=values=>values.reduce((a,b)=>a+b,0)/values.length;
const width=(range,n)=>range*Math.sqrt(Math.log(2*COMPARISONS/ALPHA)/(2*n));
const bounded=(value,m)=>[Math.max(0,value-m),Math.min(1,value+m)];
const percent=x=>(100*x).toFixed(2)+'%';

export function aggregate(reports){
  assert(Array.isArray(reports)&&reports.length===3*PARTS,'Expected exactly 24 frozen shards');
  const byGeneration=new Map([[0,new Map()],[1,new Map()],[2,new Map()]]);
  const headIDs=new Map();
  let sourceSHA=null,engineSHA=null,policyID=null;
  for(const r of reports){
    assert(r.schema===SCHEMA&&[0,1,2].includes(r.generation)&&
      r.checkpoint_generation===r.generation+1&&
      Number.isSafeInteger(r.part)&&r.part>=0&&r.part<PARTS,
      'Invalid registered generation or part');
    const bucket=byGeneration.get(r.generation);
    assert(!bucket.has(r.part),'Duplicate part');
    assert(r.evaluation_policy==='frozen-greedy-no-training'&&
      r.benchmark==='AQ-Base-001 exact integrated strategy-match'&&
      r.aq_base_model_sha256===AQ_SHA&&
      r.champion_promoted===false&&
      r.seed_start===FIRST_SEED+r.part*SEEDS_PER_PART&&
      r.seed_clusters===SEEDS_PER_PART&&r.matches===2*SEEDS_PER_PART&&
      Number.isInteger(r.wins)&&r.wins>=0&&r.wins<=r.matches&&
      Array.isArray(r.rows)&&r.rows.length===r.matches,
      'Untrusted incomplete or non-real evaluation results');
    for(const field of ['source_sha','engine_sha','research_policy_id']){
      const current=field==='source_sha'?sourceSHA:field==='engine_sha'?engineSHA:policyID;
      assert(typeof r[field]==='string'&&r[field].length>0&&
        (current===null||current===r[field]),'Mixed evaluation provenance: '+field);
    }
    sourceSHA=r.source_sha;engineSHA=r.engine_sha;policyID=r.research_policy_id;
    const ids=JSON.stringify([r.head_sha256,r.sealed_model_manifest_sha256]);
    if(headIDs.has(r.generation))assert(headIDs.get(r.generation)===ids,
      'Mixed weights across a single generation');
    else headIDs.set(r.generation,ids);
    const map=new Map();
    for(const row of r.rows){
      assert(Number.isSafeInteger(row.seed)&&row.seed>=r.seed_start&&
        row.seed<r.seed_start+SEEDS_PER_PART&&
        [0,1].includes(row.candidateSeat)&&typeof row.won==='boolean'&&
        Array.isArray(row.score)&&row.score.length===2&&
        row.score[row.won?row.candidateSeat:1-row.candidateSeat]===4&&
        row.score[1-(row.won?row.candidateSeat:1-row.candidateSeat)]<4&&
        Number.isSafeInteger(row.rounds)&&row.rounds>=4&&
        Number.isSafeInteger(row.accepted_decisions)&&row.accepted_decisions>0,
        'Invalid exact four-win Match row');
      const key=row.seed+':'+row.candidateSeat;
      assert(!map.has(key),'Repeated seed/seat row');
      map.set(key,row);
    }
    assert(map.size===2*SEEDS_PER_PART&&
      [...Array(SEEDS_PER_PART).keys()].every(i=>
        [0,1].every(seat=>map.has((r.seed_start+i)+':'+seat)))&&
      [...map.values()].filter(x=>x.won).length===r.wins,
      'Seed/seat population incomplete or wrong win count');
    bucket.set(r.part,map);
  }
  const winsByGen=[];
  const valuesByGen=[];
  for(const g of [0,1,2]){
    const parts=byGeneration.get(g);
    assert(parts.size===PARTS,'Missing generation part');
    const values=[];
    let wins=0;
    for(let seed=FIRST_SEED;seed<FIRST_SEED+TOTAL;seed++){
      const part=Math.floor((seed-FIRST_SEED)/SEEDS_PER_PART);
      const partRows=parts.get(part);
      assert(partRows,'Missing frozen shard');
      const a=partRows.get(seed+':0'),b=partRows.get(seed+':1');
      assert(a&&b,'Missing paired seat');
      wins+=Number(a.won)+Number(b.won);
      values.push((Number(a.won)+Number(b.won))/2);
    }
    valuesByGen.push(values);
    winsByGen.push(wins);
  }
  const perGeneration=[0,1,2].map(g=>{
    const rate=avg(valuesByGen[g]),margin=width(1,TOTAL);
    return {training_generation:g,checkpoint_generation:g+1,
      model_sha256:JSON.parse(headIDs.get(g))[1],
      head_sha256:JSON.parse(headIDs.get(g))[0],
      wins:winsByGen[g],matches:2*TOTAL,independent_seed_clusters:TOTAL,
      win_rate:rate,familywise_simultaneous_95pct_ci:bounded(rate,margin),
      significantly_above_50pct:rate-margin>.5,
      significantly_below_50pct:rate+margin<.5};
  });
  const comparisons=[];
  for(const [a,b] of [[1,0],[2,1],[2,0]]){
    const delta=avg(valuesByGen[a].map((v,i)=>v-valuesByGen[b][i]));
    const margin=width(2,TOTAL);
    comparisons.push({newer_training_generation:a,older_training_generation:b,
      paired_win_rate_delta:delta,
      familywise_simultaneous_95pct_ci:[delta-margin,delta+margin],
      significant_improvement:delta-margin>0,
      significant_regression:delta+margin<0});
  }
  return {schema:'rnad-i5-g012-8192-each-aq-paired-aggregate-v1',
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    source_sha:sourceSHA,engine_sha:engineSHA,
    policy_id:policyID,first_heldout_seed:FIRST_SEED,
    last_heldout_seed:FIRST_SEED+TOTAL-1,
    heldout_seed_clusters:TOTAL,seat_swapped:true,
    matches_per_generation:2*TOTAL,total_complete_four_win_matches:6*TOTAL,
    simultaneous_familywise_alpha:ALPHA,comparisons_count:COMPARISONS,
    method:'Bonferroni-Hoeffding two-sided seed-cluster confidence bounds; one 2-seat cluster per seed',
    post_selection_champion_claim:false,
    champion_promoted:false,per_generation:perGeneration,paired_comparisons:comparisons};
}

export function reportMarkdown(x){
  const out=['# R-NaD G0/G1/G2 frozen AQ-Base heldout evaluation','',
    'All rows are true accepted four-win Matches; no teacher Top1 or training smoke.',
    'G0/G1/G2 label the training generation, and use the model **after** its 40-step Learner.',
    `Independent heldout seed clusters: ${x.heldout_seed_clusters}, both seats, ${x.total_complete_four_win_matches.toLocaleString()} complete Matches.`,
    'Familywise simultaneous 95% intervals use Bonferroni + two-sided seed-cluster Hoeffding across six planned comparisons.',
    '',
    '| Finished generation | AQ wins / 8192 | AQ win rate | Simultaneous 95% CI | Significantly >AQ? |',
    '|---|---:|---:|---|---|'];
  for(const r of x.per_generation)
    out.push(`| G${r.training_generation} | ${r.wins}/${r.matches} | ${percent(r.win_rate)} | [${percent(r.familywise_simultaneous_95pct_ci[0])}, ${percent(r.familywise_simultaneous_95pct_ci[1])}] | ${r.significantly_above_50pct?'yes':'no'} |`);
  out.push('','| Paired generations | Win rate delta | Simultaneous 95% CI | Decision |',
    '|---|---:|---|---|');
  for(const c of x.paired_comparisons){
    const f=p=>(p>=0?'+':'')+(100*p).toFixed(2)+'pp';
    out.push(`| G${c.newer_training_generation} − G${c.older_training_generation} | ${f(c.paired_win_rate_delta)} | [${f(c.familywise_simultaneous_95pct_ci[0])}, ${f(c.familywise_simultaneous_95pct_ci[1])}] | ${c.significant_improvement?'improved':c.significant_regression?'regressed':'inconclusive'} |`);
  }
  out.push('','**Research-only.** No Champion promotion, BestVerified update, or main merge.','');
  return out.join('\n');
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  assert(process.argv.length===4,'Usage: node aggregate-shared-g012-vs-aq.mjs DIR OUT');
  const directory=process.argv[2],target=process.argv[3];
  const entries=readdirSync(directory,{withFileTypes:true})
    .filter(x=>x.isDirectory()).map(x=>x.name).sort();
  assert(entries.length===3*PARTS,'Incomplete Artifact population');
  const reports=entries.map(name=>{
    const p=join(directory,name,'study.json');
    return JSON.parse(readFileSync(p,'utf8'));
  });
  const result=aggregate(reports);
  mkdirSync(target,{recursive:true});
  writeFileSync(join(target,'REPORT.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  writeFileSync(join(target,'REPORT.md'),reportMarkdown(result),{flag:'wx'});
  console.log(reportMarkdown(result));
}
