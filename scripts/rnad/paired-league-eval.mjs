// RNAD I5: prospective paired league evaluation, no training or promotion.
// Frozen versions and one two-seat seed cluster are required for every model.
// Direct head-to-head (candidate vs Champion) is NOT the same estimand as
// a difference of two independently-run models against common AQ.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {pairedClusterCI,pairedSignSensitivity} from './paired-league-stats.mjs';

export const PAIRED_LEAGUE_PLAN_SCHEMA='rnad-i5-prospective-paired-league-plan-v1';
export const PAIRED_LEAGUE_EVIDENCE_SCHEMA='rnad-i5-frozen-paired-match-evidence-v1';
export const PAIRED_LEAGUE_REPORT_SCHEMA='rnad-i5-prospective-paired-league-report-v1';
const validSha=(s,length)=>typeof s==='string'&&
  new RegExp('^[0-9a-f]{'+length+'}$').test(s);
const isId=s=>typeof s==='string'&&/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/.test(s);
const fail=s=>{throw new TypeError('RNAD prospective paired evaluation: '+s);};
const requireOk=(p,s)=>{if(!p)fail(s);};
function verifyPlan(p){
  requireOk(p?.schema===PAIRED_LEAGUE_PLAN_SCHEMA&&
    ['development','final'].includes(p.stage)&&
    ['shared-opponent','direct-head-to-head'].includes(p.evaluation_type)&&
    validSha(p.source_sha,40)&&validSha(p.engine_sha,40)&&
    isId(p.opponent_id)&&
    Number.isSafeInteger(p.seed_start)&&p.seed_start>0&&
    Number.isInteger(p.seed_blocks)&&p.seed_blocks>=128&&p.seed_blocks<=65536&&
    p.seed_start+p.seed_blocks-1<=0xffffffff&&
    [0.20,0.05].includes(p.familywise_alpha)&&
    p.confidence_level===1-p.familywise_alpha&&
    p.method==='paired-seed-block-student-t-bonferroni'&&
    p.fixed_sample_size===true&&p.model_selection_before_holdout===true&&
    p.automatic_champion_promotion===false&&
    Number.isFinite(p.minimum_practical_gain)&&
    p.minimum_practical_gain>=0&&p.minimum_practical_gain<1&&
    Array.isArray(p.comparisons)&&p.comparisons.length>=1&&
    p.comparisons.length<=190&&
    Array.isArray(p.excluded_seed_ranges),
    'missing or malformed preregistered evaluation plan');
  const seen=new Set(),mentioned=new Set();
  for(const c of p.comparisons){
    requireOk(isId(c.candidate_model_id)&&isId(c.baseline_model_id)&&
      c.candidate_model_id!==c.baseline_model_id,
      'invalid or identical compared models');
    const key=c.candidate_model_id+':'+c.baseline_model_id;
    requireOk(!seen.has(key),'duplicate intended comparison');
    seen.add(key);
    mentioned.add(c.candidate_model_id);mentioned.add(c.baseline_model_id);
    if(p.evaluation_type==='direct-head-to-head')
      requireOk(c.baseline_model_id===p.opponent_id,
        'direct head-to-head requires one fixed Champion/opponent');
    else requireOk(!mentioned.has(p.opponent_id),
      'common-opponent models may not be the benchmark');
  }
  const covered=new Set();
  for(const r of p.excluded_seed_ranges){
    requireOk(Number.isSafeInteger(r?.start)&&r.start>0&&
      Number.isInteger(r.count)&&r.count>0&&r.count<=1000000&&
      r.start+r.count-1<=0xffffffff,
      'malformed previously consumed seed range');
    const a=Math.max(r.start,p.seed_start),
      b=Math.min(r.start+r.count-1,p.seed_start+p.seed_blocks-1);
    requireOk(a>b,'training/development/consumed seed range overlaps evaluation');
    const name=r.start+':'+r.count;
    requireOk(!covered.has(name),'repeated exclusion declaration');
    covered.add(name);
  }
  if(p.stage==='final')
    requireOk(p.excluded_seed_ranges.length>=1&&
      p.independent_holdout_precommitted===true,
      'final promotion proof requires a new, declared disjoint holdout');
  return p;
}

function validateShard(shard,plan){
  requireOk(shard?.schema===PAIRED_LEAGUE_EVIDENCE_SCHEMA&&
    isId(shard.model_id)&&validSha(shard.model_sha256,64)&&
    shard.source_sha===plan.source_sha&&
    shard.engine_sha===plan.engine_sha&&
    shard.opponent_id===plan.opponent_id&&
    shard.evaluation_policy==='frozen-greedy-no-training'&&
    shard.training_during_evaluation===false&&
    shard.champion_promoted===false&&
    Number.isSafeInteger(shard.seed_start)&&
    Number.isInteger(shard.seed_blocks)&&shard.seed_blocks>0&&
    shard.seed_start>=plan.seed_start&&
    shard.seed_start+shard.seed_blocks<=plan.seed_start+plan.seed_blocks&&
    Array.isArray(shard.rows)&&shard.rows.length===2*shard.seed_blocks&&
    shard.matches===2*shard.seed_blocks&&
    Number.isInteger(shard.wins)&&shard.wins>=0&&shard.wins<=shard.matches,
    'invalid frozen model, policy, provenance or incomplete shard');
  let wins=0;
  const seats=new Set();
  for(const row of shard.rows){
    requireOk(Number.isSafeInteger(row.seed)&&
      row.seed>=shard.seed_start&&row.seed<shard.seed_start+shard.seed_blocks&&
      [0,1].includes(row.candidateSeat)&&typeof row.won==='boolean'&&
      row.completed===true&&
      Number.isSafeInteger(row.rounds)&&row.rounds>=4&&row.rounds<=64&&
      Number.isSafeInteger(row.accepted_decisions)&&row.accepted_decisions>0&&
      Array.isArray(row.score)&&row.score.length===2&&
      row.score.every(x=>Number.isInteger(x)&&x>=0&&x<=4)&&
      row.score[row.won?row.candidateSeat:1-row.candidateSeat]===4&&
      row.score[row.won?1-row.candidateSeat:row.candidateSeat]<4,
      'invalid real completed four-win seed/seat outcome');
    const key=row.seed+':'+row.candidateSeat;
    requireOk(!seats.has(key),'duplicate seed/seat in one shard');
    seats.add(key);
    wins+=Number(row.won);
  }
  for(let seed=shard.seed_start;seed<shard.seed_start+shard.seed_blocks;seed++)
    requireOk(seats.has(seed+':0')&&seats.has(seed+':1'),
      'missing seat-swapped Match in shard');
  requireOk(wins===shard.wins,'win count differs from authoritative rows');
  return shard;
}

export function evaluatePairedLeague(plan,reports){
  verifyPlan(plan);
  requireOk(Array.isArray(reports)&&reports.length>=1,
    'missing frozen Match evidence');
  const byModel=new Map(),modelSha=new Map();
  for(const raw of reports){
    const s=validateShard(raw,plan);
    const seenSha=modelSha.get(s.model_id);
    requireOk(seenSha===undefined||seenSha===s.model_sha256,
      'model weights changed between evaluation shards');
    modelSha.set(s.model_id,s.model_sha256);
    const map=byModel.get(s.model_id)??new Map();
    for(const row of s.rows){
      const key=row.seed+':'+row.candidateSeat;
      requireOk(!map.has(key),'duplicate model/seed/seat across shards');
      map.set(key,row);
    }
    byModel.set(s.model_id,map);
  }
  const requested=new Set(plan.comparisons.flatMap(c=>[
    c.candidate_model_id,...(plan.evaluation_type==='shared-opponent'?
      [c.baseline_model_id]:[])]));
  requireOk(byModel.size===requested.size&&
    [...requested].every(x=>byModel.has(x)),
    'missing or unexpected model in evaluation family');
  const scores=new Map();
  for(const id of requested){
    const rows=byModel.get(id),perSeed=[];
    requireOk(rows.size===2*plan.seed_blocks,
      'incomplete model seed-block cohort '+id);
    for(let seed=plan.seed_start;seed<plan.seed_start+plan.seed_blocks;seed++){
      const a=rows.get(seed+':0'),b=rows.get(seed+':1');
      requireOk(a&&b,'missing identical seed/seat between models');
      perSeed.push((Number(a.won)+Number(b.won))/2);
    }
    scores.set(id,perSeed);
  }
  const comparisons=plan.comparisons.map(c=>{
    const candidate=scores.get(c.candidate_model_id);
    const baseline=plan.evaluation_type==='shared-opponent'
      ?scores.get(c.baseline_model_id):null;
    if(baseline)requireOk(modelSha.get(c.candidate_model_id)!==
      modelSha.get(c.baseline_model_id),
      'identical checkpoint weights cannot establish improvement');
    const blocks=candidate.map((v,i)=>baseline?v-baseline[i]:v);
    const range=baseline?[-1,1]:[0,1],nullValue=baseline?0:.5;
    const inference=pairedClusterCI(blocks,{
      familywiseAlpha:plan.familywise_alpha,comparisons:plan.comparisons.length,
      range,nullValue,
    });
    const sensitivity=pairedSignSensitivity(blocks,{nullValue});
    return {
      candidate_model_id:c.candidate_model_id,
      baseline_model_id:c.baseline_model_id,
      estimand:baseline?'paired-difference-vs-same-frozen-opponent':
        'direct-head-to-head-seat-swapped-win-probability',
      candidate_model_sha256:modelSha.get(c.candidate_model_id),
      baseline_model_sha256:baseline?modelSha.get(c.baseline_model_id):null,
      candidate_wins:candidate.reduce((n,x)=>n+2*x,0),
      baseline_wins:baseline?baseline.reduce((n,x)=>n+2*x,0):null,
      matches_per_model:2*plan.seed_blocks,
      ...inference,
      exploratory_sign_sensitivity:sensitivity,
      statistically_supported_gain:inference.confidence_interval[0]>
        nullValue+plan.minimum_practical_gain,
    };
  });
  return {
    schema:PAIRED_LEAGUE_REPORT_SCHEMA,
    stage:plan.stage,evaluation_type:plan.evaluation_type,
    source_sha:plan.source_sha,engine_sha:plan.engine_sha,
    opponent_id:plan.opponent_id,
    first_seed:plan.seed_start,last_seed:plan.seed_start+plan.seed_blocks-1,
    independent_seed_blocks:plan.seed_blocks,
    complete_four_win_matches:reports.reduce((n,x)=>n+x.matches,0),
    distinct_models:byModel.size,
    familywise_alpha:plan.familywise_alpha,
    simultaneous_confidence_level:plan.confidence_level,
    multiple_testing:'Bonferroni over all preregistered comparisons',
    comparisons,model_ids:[...requested].sort(),
    diagnostic_only:plan.stage!=='final',
    statistical_gate_passed:comparisons.every(x=>x.statistically_supported_gain),
    requires_independent_chatgpt_review:true,
    requires_authoritative_artifact_sha_verification:true,
    champion_promoted:false,main_merged:false,
    caution:'Student-t is a model-based block CI, not a distribution-free guarantee. Raw evidence must be authenticated against a GitHub run and evaluated once after fixed N. Neither a statistical flag nor a retrospective reanalysis authorizes automatic Champion promotion.',
  };
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2);
  requireOk(args.length>=3&&args[0]?.endsWith('.json')&&
    args[1]?.endsWith('.json'), 'usage: node paired-league-eval.mjs PLAN.json OUT.json SHARD.json ...');
  const plan=JSON.parse(readFileSync(args[0],'utf8'));
  const reports=args.slice(2).map(x=>JSON.parse(readFileSync(x,'utf8')));
  const result=evaluatePairedLeague(plan,reports);
  writeFileSync(args[1],JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({schema:result.schema,comparison_count:result.comparisons.length,
    complete_four_win_matches:result.complete_four_win_matches,
    statistical_gate_passed:result.statistical_gate_passed,
    champion_promoted:false}));
}
