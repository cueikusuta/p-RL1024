// Exact frozen original AQ-Base comparison of one-vs-four-update controlled
// c005-origin v3 models trained for exactly 2048 accepted Match boundaries each.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {evaluateAgainstOriginalAQ} from './evaluate-vs-aq-full.mjs';
import {weightsSha256 as AQ_SHA} from '../../models/aq-base.mjs';

export const CONTROLLED_EVAL_SCHEMA='rnad-i5-controlled-c005-u1-vs-u4-real-aq-v1';
export const CONTROLLED_EVAL_SEED=8800001;
export const CONTROLLED_SEED_PARTS=4;
export const CONTROLLED_SEEDS_PER_PART=512;
const sha=x=>createHash('sha256').update(readFileSync(x)).digest('hex');
export function evalControlled({dir,updates,part}){
  if(![1,4].includes(updates)||!Number.isInteger(updates)||
    !Number.isInteger(part)||part<0||part>=CONTROLLED_SEED_PARTS)
    throw Error('Invalid pre-registered independent controlled arm/part');
  const manifest=JSON.parse(readFileSync(join(dir,'controlled-train-report.json')));
  const checkpoint=JSON.parse(readFileSync(join(dir,'checkpoints/latest.json')));
  const file=join(dir,'final-model.json'),fileSHA=sha(file);
  const frozen=manifest.frozen_opponent_model_sha256;
  const required=['base128','base256','c005'];
  if(!frozen||Object.keys(frozen).sort().join(',')!==
      required.slice().sort().join(',')||
     required.some(name=>!(/^[0-9a-f]{64}$/.test(frozen[name]))||
       sha(join(dir,'frozen',name+'-v3.json'))!==frozen[name]))
    throw Error('Altered frozen opponent model weights between controlled A/B arms');
  if(manifest.schema!=='rnad-i5-controlled-c005-v3-update1-vs-update4-v1'||
     manifest.optimizer_updates_per_match!==updates||
     manifest.optimizer_updates!==2048*updates||
     manifest.completed_matches!==2048||manifest.target_complete_matches!==2048||
     manifest.initial_seed!==6500001||
     manifest.initial_model!=='c005-v2-to-v3'||
     manifest.engine_sha!=='ea81165993110ee9fb3a09dcb97c6708c0d7397b'||
     manifest.completed!==true||
     manifest.accepted_match_ids_unique!==true||
     manifest.same_accepted_match_seed_stream_for_both_arms!==true||
     manifest.same_model_weights_and_frozen_opponents_for_both_arms!==true||
     sha(join(dir,'initial-model.json'))!==manifest.initial_v3_model_sha256||
     sha(join(dir,'checkpoints',checkpoint.checkpoint_file))!==
       manifest.final_checkpoint_sha256||
     checkpoint.checkpoint_sha256!==manifest.final_checkpoint_sha256||
     fileSHA!==manifest.final_model_sha256||
     checkpoint.metadata.initialization_sha256!==manifest.initial_v2_origin_sha256||
     checkpoint.metadata.initial_seed!==manifest.initial_seed||
     checkpoint.metadata.league_pool_sha256!==manifest.frozen_opponent_pool_sha256||
     checkpoint.metadata.completed_matches!==2048||
     checkpoint.metadata.updates!==2048*updates)
    throw Error('Incomplete or altered controlled training checkpoint');
  const seed=CONTROLLED_EVAL_SEED+part*CONTROLLED_SEEDS_PER_PART,rows=[];
  for(let k=0;k<CONTROLLED_SEEDS_PER_PART;k+=100){
    const n=Math.min(100,CONTROLLED_SEEDS_PER_PART-k);
    const one=evaluateAgainstOriginalAQ({modelPath:file,seed:seed+k,
      pairs:n,phaseMode:'neural-all',switchGuard:'anti-loop-v1'});
    if(one.model_sha256!==fileSHA||one.aq_base_model_sha256!==AQ_SHA||
       one.matches!==2*n||one.champion_promoted!==false)
      throw Error('Real AQ true complete Match evidence mismatches controlled actor');
    rows.push(...one.rows);
    console.log(JSON.stringify({updates,part,pairs_done:rows.length/2}));
  }
  return {schema:CONTROLLED_EVAL_SCHEMA,
    origin:'c005-v2-to-v3',updates_per_match:updates,part,
    training_source_sha:manifest.training_source_sha,
    training_seed:manifest.initial_seed,training_matches:manifest.completed_matches,
    initial_model_sha256:manifest.initial_v3_model_sha256,
    initial_v2_origin_sha256:manifest.initial_v2_origin_sha256,
    frozen_opponent_pool_sha256:manifest.frozen_opponent_pool_sha256,
    frozen_opponent_model_sha256:frozen,
    model_sha256:fileSHA,final_checkpoint_sha256:manifest.final_checkpoint_sha256,
    engine_sha:manifest.engine_sha,
    aq_base_model_sha256:AQ_SHA,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    evaluation_seed_start:seed,
    independent_paired_seeds:CONTROLLED_SEEDS_PER_PART,
    matches:rows.length,wins:rows.filter(x=>x.won).length,
    training_during_evaluation:false,champion_promoted:false,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const opt=new Map(),a=process.argv.slice(2);
  if(a.length!==8)throw Error('Need --dir PATH --updates N --part N --out FILE');
  for(let i=0;i<a.length;i+=2){
    if(!['--dir','--updates','--part','--out'].includes(a[i])||
      opt.has(a[i]))throw Error('Invalid controlled evaluation CLI');
    opt.set(a[i],a[i+1]);
  }
  const r=evalControlled({dir:opt.get('--dir'),updates:Number(opt.get('--updates')),
    part:Number(opt.get('--part'))});
  writeFileSync(opt.get('--out'),JSON.stringify(r,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({updates:r.updates_per_match,wins:r.wins,matches:r.matches}));
}
