// One planned causal comparison: same weights, opponents, Match-seed stream,
// accepted Match budget & R-NaD algorithm; sole changed variable is updates/Match.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {CONTROLLED_EVAL_SCHEMA,CONTROLLED_EVAL_SEED,
  CONTROLLED_SEED_PARTS,CONTROLLED_SEEDS_PER_PART}
  from './evaluate_controlled_updates_vs_aq.mjs';
import {weightsSha256 as AQ_SHA} from '../../models/aq-base.mjs';
export const CONTROLLED_ANALYSIS_SCHEMA='rnad-i5-controlled-u1-vs-u4-paired-real-aq-analysis-v1';
const requireOk=(ok,msg)=>{if(!ok)throw Error('Invalid controlled 1-vs-4 study: '+msg);};
export function analyzeControlled(reports){
  requireOk(Array.isArray(reports)&&reports.length===2*CONTROLLED_SEED_PARTS,
    'need eight independent frozen both-seat partitions');
  const arms=new Map();
  for(const report of reports){
    const r=report;
    requireOk(r?.schema===CONTROLLED_EVAL_SCHEMA&&
      [1,4].includes(r.updates_per_match)&&
      Number.isInteger(r.part)&&r.part>=0&&r.part<CONTROLLED_SEED_PARTS&&
      r.origin==='c005-v2-to-v3'&&r.training_seed===6500001&&
      r.training_matches===2048&&
      r.engine_sha==='ea81165993110ee9fb3a09dcb97c6708c0d7397b'&&
      r.aq_base_model_sha256===AQ_SHA&&
      r.benchmark==='AQ-Base-001 exact integrated strategy-match'&&
      r.evaluation_seed_start===CONTROLLED_EVAL_SEED+r.part*CONTROLLED_SEEDS_PER_PART&&
      r.independent_paired_seeds===CONTROLLED_SEEDS_PER_PART&&
      r.matches===2*CONTROLLED_SEEDS_PER_PART&&
      r.training_during_evaluation===false&&r.champion_promoted===false&&
      Array.isArray(r.rows)&&r.rows.length===r.matches&&
      typeof r.training_source_sha==='string'&&/^[0-9a-f]{40}$/.test(r.training_source_sha)&&
      ['initial_model_sha256','initial_v2_origin_sha256',
       'frozen_opponent_pool_sha256','final_checkpoint_sha256',
       'model_sha256'].every(k=>/^[0-9a-f]{64}$/.test(r[k])),
      'wrong pre-registered training arm, frozen source or Match count');
    const index=r.updates_per_match+':'+r.part;
    requireOk(!arms.has(index),'duplicate arm/part');
    let wins=0;
    for(let j=0;j<CONTROLLED_SEEDS_PER_PART;j++){
      for(const seat of [0,1]){
        const t=r.rows[2*j+seat];
        requireOk(t?.seed===r.evaluation_seed_start+j&&
          t.candidateSeat===seat&&typeof t.won==='boolean'&&
          Number.isInteger(t.rounds)&&t.rounds>=4&&
          Number.isInteger(t.accepted_decisions)&&t.accepted_decisions>=2*t.rounds&&
          Array.isArray(t.score)&&t.score.length===2&&
          t.score[t.won?seat:1-seat]===4&&
          Number.isInteger(t.score[t.won?1-seat:seat])&&
          t.score[t.won?1-seat:seat]<4,
          'not a complete four-win accepted seat-swapped Match');
        wins+=Number(t.won);
      }
    }
    requireOk(wins===r.wins,'forged win counter');
    arms.set(index,r);
  }
  requireOk(arms.size===8,'missing complete paired controlled training arms');
  const N=CONTROLLED_SEEDS_PER_PART*CONTROLLED_SEED_PARTS;
  const result={};
  for(const u of [1,4]){
    const parts=[0,1,2,3].map(p=>arms.get(u+':'+p));
    requireOk(parts.every(Boolean),'missing controlled model evaluation part');
    const first=parts[0];
    for(const x of parts.slice(1))
      for(const key of ['origin','training_seed','training_source_sha','training_matches',
        'initial_model_sha256','initial_v2_origin_sha256','frozen_opponent_pool_sha256',
        'model_sha256','final_checkpoint_sha256'])
        requireOk(first[key]===x[key],'mixed same-arm trained model across seed parts');
    const wins=parts.reduce((sum,x)=>sum+x.wins,0);
    result[u]={updates_per_match:u,wins,win_rate:wins/(2*N),
      trained_matches:first.training_matches,
      initial_model_sha256:first.initial_model_sha256,
      initial_v2_origin_sha256:first.initial_v2_origin_sha256,
      frozen_opponent_pool_sha256:first.frozen_opponent_pool_sha256,
      training_seed:first.training_seed,
      training_source_sha:first.training_source_sha,
      final_model_sha256:first.model_sha256,
      final_checkpoint_sha256:first.final_checkpoint_sha256};
  }
  for(const key of ['trained_matches','initial_model_sha256','initial_v2_origin_sha256',
    'frozen_opponent_pool_sha256','training_seed','training_source_sha'])
    requireOk(result[1][key]===result[4][key],
      'noncontrolled initial conditions, sources, seeds or training budget');
  requireOk(result[1].final_model_sha256!==result[4].final_model_sha256,
    'no model distinction despite different optimizer pass counts');
  let gain=0;const rowPairs=[];
  for(let part=0;part<CONTROLLED_SEED_PARTS;part++){
    const a=arms.get('1:'+part),b=arms.get('4:'+part);
    for(let j=0;j<CONTROLLED_SEEDS_PER_PART;j++){
      const seed=CONTROLLED_EVAL_SEED+part*CONTROLLED_SEEDS_PER_PART+j;
      const a_score=(Number(a.rows[2*j].won)+Number(a.rows[2*j+1].won))/2;
      const b_score=(Number(b.rows[2*j].won)+Number(b.rows[2*j+1].won))/2;
      gain+=a_score-b_score;
      rowPairs.push({seed,u1_score:a_score,u4_score:b_score});
    }
  }
  gain/=N;
  requireOk(Math.abs(gain-(result[1].win_rate-result[4].win_rate))<1e-10,
    'paired result does not reproduce aggregate win rates');
  const half=Math.sqrt(2*Math.log(40)/N); // score diff in [-1,1]
  return {schema:CONTROLLED_ANALYSIS_SCHEMA,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    independent_seed_clusters:N,real_complete_four_win_matches:4*N,
    independent_train_matches_per_arm:2048,
    heldout_seed_first:CONTROLLED_EVAL_SEED,
    heldout_seed_last:CONTROLLED_EVAL_SEED+N-1,
    only_pre_registered_treatment_difference:'1 vs 4 optimizer updates per accepted Match',
    same_initial_weights_opponents_seed_stream_and_accepted_training_matches:true,
    same_heldout_seeds_and_both_seat_positions:true,
    two_sided_independent_seed_cluster_hoeffding_95_halfwidth:half,
    paired_win_rate_delta_u1_minus_u4:gain,
    paired_gain_simultaneous_95pct_bounds:[Math.max(-1,gain-half),Math.min(1,gain+half)],
    u1_significantly_stronger_at_95:gain-half>0,
    u4_significantly_stronger_at_95:gain+half<0,
    statistical_decision:gain-half>0?'u1_better':
      gain+half<0?'u4_better':'inconclusive',
    no_training_on_heldout:true,champion_promoted:false,
    formal_league_gate_passed:false,arms:result};
}
export function controlledMarkdown(r){
  return [
    '# I5: randomized seed-matched optimizer update count 1 vs 4',
    '',
    'Identical c005 v2->v3 initial weights, fixed old opponent pool, 2048 accepted training Match seeds per arm. One difference: optimizer updates per accepted Match.',
    `Original frozen AQ; ${r.independent_seed_clusters} independent heldout seeds, both seats; ${r.real_complete_four_win_matches} complete 4-win Matches.`,
    `1 update: ${(r.arms[1].win_rate*100).toFixed(2)}%; 4 updates: ${(r.arms[4].win_rate*100).toFixed(2)}%.`,
    `Corresponding paired gain (1 minus 4): ${(r.paired_win_rate_delta_u1_minus_u4*100).toFixed(2)} percentage points; conservative two-sided 95% confidence interval [${(r.paired_gain_simultaneous_95pct_bounds[0]*100).toFixed(2)},${(r.paired_gain_simultaneous_95pct_bounds[1]*100).toFixed(2)}]pt.`,
    `Decision: ${r.statistical_decision}.`,
    '',
    'Only an update-frequency comparison for this one selected training curriculum, source and data budget; may not generalize to other initial models or training durations.',
    'No formal league entry or Champion promotion.',''
  ].join('\n');
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  if(process.argv.length!==4)throw Error('Usage: ... STUDY-DIR OUT-DIR');
  const [studyDir,outDir]=process.argv.slice(2);
  const reports=[1,4].flatMap(u=>[0,1,2,3].map(part=>JSON.parse(
    readFileSync(join(studyDir,`controlled-u${u}-p${part}`,'study.json')))));
  const result=analyzeControlled(reports);
  mkdirSync(outDir,{recursive:true});
  writeFileSync(join(outDir,'REPORT.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  writeFileSync(join(outDir,'REPORT.md'),controlledMarkdown(result),{flag:'wx'});
  console.log(controlledMarkdown(result));
}
