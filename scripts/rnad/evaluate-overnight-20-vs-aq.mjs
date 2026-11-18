// Frozen overnight research policies vs REAL AQ-Base-001, complete four-win Matches.
// Both seats, one independent seed cluster; no training, no model selection on holdout.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {join} from 'node:path';
import {PersistentRNADInference} from './infer.mjs';
import {evaluateAgainstOriginalAQ} from './evaluate-vs-aq-full.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

export const OVERNIGHT_EVAL_SCHEMA='rnad-i5-overnight-20-real-aq-paired-part-v1';
export const TRAINING_SOURCE_SHA='833f4039b69bbad64dc7a28c059d4b934004e5bf';
export const ORIGINAL_OVERNIGHT_RUN=44221678145;
export const FIRST_HELDOUT_SEED=8600001;
export const PAIRED_SEEDS_PER_PART=512;
export const PARTS_PER_MODEL=2;
const sha=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
const load=path=>JSON.parse(readFileSync(path,'utf8'));
const requireOk=(condition,reason)=>{if(!condition)throw new Error(reason);};
const expectedMode=shard=>shard%4===3?'selfplay':'league';
const expectedSchema=mode=>'rnad-three-phase-features-'+
  (mode==='selfplay'?'v5':'v3');

export function verifyFrozenOvernightModel(dir,shard){
  requireOk(Number.isInteger(shard)&&shard>=0&&shard<20,'Invalid registered shard');
  const ident=load(join(dir,'series-identity.json'));
  const report=load(join(dir,'wave-summary.json'));
  const index=load(join(dir,'checkpoints/latest.json'));
  const md=index.metadata,mode=expectedMode(shard);
  const modelPath=join(dir,'final-model.json');
  requireOk(ident.schema==='rnad-i5-early-night-league-exploratory-v1'&&
    report.schema===ident.schema&&ident.shard===shard&&report.shard===shard&&
    ident.mode===mode&&report.mode===mode&&report.wave===2&&
    report.wave_completed_matches>0&&
    report.series_source_sha===TRAINING_SOURCE_SHA&&
    ident.series_source_sha===TRAINING_SOURCE_SHA&&
    md.source_sha===TRAINING_SOURCE_SHA&&
    md.training_code_sha===TRAINING_SOURCE_SHA&&
    md.initial_seed===2600001+50000*shard&&
    md.observation_schema_version===expectedSchema(mode)&&
    md.completed_matches===report.cumulative_completed_matches&&
    md.updates===report.optimizer_updates&&
    md.completed_match_ids.length===md.completed_matches&&
    new Set(md.completed_match_ids).size===md.completed_matches&&
    ident.league_eligibility_gate_passed===false&&
    report.champion_promoted===false&&
    index.checkpoint_sha256===sha(join(dir,'checkpoints',index.checkpoint_file))&&
    report.last_model_sha256===sha(modelPath),
    'Missing, altered or incoherent frozen overnight wave2 checkpoint');
  const network=new PersistentRNADInference(modelPath);
  requireOk(network.featureSchema===expectedSchema(mode),
    'Frozen model and expected full phase schema differ');
  return {shard,mode,origin:ident.origin,
    updates_per_match:ident.updates_per_match,
    trained_matches:md.completed_matches,
    optimizer_updates:md.updates,
    model_sha256:report.last_model_sha256,
    checkpoint_sha256:index.checkpoint_sha256,modelPath};
}
export function validatePartRows(rows,{seedStart,pairs}){
  requireOk(Array.isArray(rows)&&rows.length===pairs*2,
    'Missing paired complete four-win Match rows');
  const keys=new Set();
  let wins=0;
  for(const r of rows){
    requireOk(Number.isInteger(r.seed)&&r.seed>=seedStart&&
      r.seed<seedStart+pairs&&[0,1].includes(r.candidateSeat)&&
      typeof r.won==='boolean'&&Number.isInteger(r.rounds)&&
      r.rounds>=4&&r.rounds<=64&&
      Number.isInteger(r.accepted_decisions)&&
      r.accepted_decisions>=2*r.rounds&&
      Array.isArray(r.score)&&r.score.length===2&&
      r.score.every(n=>Number.isInteger(n)&&n>=0&&n<=4)&&
      r.score[r.won?r.candidateSeat:1-r.candidateSeat]===4&&
      r.score[r.won?1-r.candidateSeat:r.candidateSeat]<4,
      'Invalid or incomplete accepted seat-swapped Match');
    const key=r.seed+':'+r.candidateSeat;
    requireOk(!keys.has(key),'Duplicate paired seed and seat');
    keys.add(key);wins+=Number(r.won);
  }
  for(let s=seedStart;s<seedStart+pairs;s++)
    for(const seat of [0,1])
      requireOk(keys.has(s+':'+seat),'Absent pre-registered seed/seat');
  return wins;
}
export function evaluateOvernightPart({dir,shard,part,pairs=PAIRED_SEEDS_PER_PART}){
  requireOk(Number.isInteger(part)&&part>=0&&part<PARTS_PER_MODEL&&
    Number.isInteger(pairs)&&pairs>=1&&pairs<=PAIRED_SEEDS_PER_PART,
    'Invalid one-shot evaluation partition/budget');
  const frozen=verifyFrozenOvernightModel(dir,shard);
  const source=verifiedRunningSourceSha();
  const seedStart=FIRST_HELDOUT_SEED+part*PAIRED_SEEDS_PER_PART;
  const rows=[];
  for(let offset=0;offset<pairs;offset+=100){
    const count=Math.min(100,pairs-offset);
    const one=evaluateAgainstOriginalAQ({
      modelPath:frozen.modelPath,seed:seedStart+offset,pairs:count,
      phaseMode:'neural-all',switchGuard:'anti-loop-v1'});
    requireOk(one.source_sha===source&&one.engine_sha===provenance.sourceCommit&&
      one.model_sha256===frozen.model_sha256&&one.matches===2*count&&
      one.rows.length===2*count&&one.champion_promoted===false,
      'Authoritative AQ study changed source, model or outcomes');
    rows.push(...one.rows);
    console.log(JSON.stringify({shard,part,completed_seed_pairs:rows.length/2,
      target_seed_pairs:pairs,complete_four_win_matches:rows.length}),true);
  }
  const wins=validatePartRows(rows,{seedStart,pairs});
  return {schema:OVERNIGHT_EVAL_SCHEMA,
    evaluation_source_sha:source,engine_sha:provenance.sourceCommit,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    training_source_sha:TRAINING_SOURCE_SHA,
    original_training_run_id:ORIGINAL_OVERNIGHT_RUN,
    training_shard:shard,part,mode:frozen.mode,origin:frozen.origin,
    updates_per_match:frozen.updates_per_match,
    trained_matches:frozen.trained_matches,optimizer_updates:frozen.optimizer_updates,
    model_sha256:frozen.model_sha256,checkpoint_sha256:frozen.checkpoint_sha256,
    policy:'frozen-greedy-neural-all-anti-loop-v1',
    seed_start:seedStart,independent_seed_pairs:pairs,
    matches:2*pairs,wins,win_rate:wins/(2*pairs),
    learner_updates:0,training_during_evaluation:false,
    selected_on_this_holdout:false,
    champion_promoted:false,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2);
  const opt=new Map();
  if(args.length!==8)throw new Error('Expected --shard N --part N --dir PATH --out FILE');
  for(let i=0;i<args.length;i+=2){
    if(!['--shard','--part','--dir','--out'].includes(args[i])||!args[i+1]||
        opt.has(args[i]))throw new Error('Invalid/duplicate evaluation option');
    opt.set(args[i],args[i+1]);
  }
  const r=evaluateOvernightPart({dir:opt.get('--dir'),
    shard:Number(opt.get('--shard')),part:Number(opt.get('--part'))});
  writeFileSync(opt.get('--out'),JSON.stringify(r,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({shard:r.training_shard,part:r.part,
    wins:r.wins,matches:r.matches,rate:r.win_rate,finished:true}));
}
