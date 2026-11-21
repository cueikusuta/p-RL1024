// Pre-training starting policies of the overnight 20-lane experiment.
// Four distinct baselines: three exact deterministic v2->v3 migrates and one v5 teacher warm-start.
// Each baseline is evaluated ONCE against frozen real AQ on same seed/seat as the 20 post-training runs.
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {PersistentRNADInference} from './infer.mjs';
import {verifyFrozenOvernightModel} from './evaluate-overnight-20-vs-aq.mjs';
import {evaluateAgainstOriginalAQ} from './evaluate-vs-aq-full.mjs';
import {weightsSha256 as AQ_SHA} from '../../models/aq-base.mjs';
import {FIRST_HELDOUT_SEED,PAIRED_SEEDS_PER_PART,PARTS_PER_MODEL,
  ORIGINAL_OVERNIGHT_RUN} from './evaluate-overnight-20-vs-aq.mjs';
const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
export const BASELINE_SCHEMA='rnad-i5-overnight-initial-real-aq-paired-v1';
export const BASELINE_ORIGINS=['base128','base256','c005','battle-focus256-v5'];
export function frozenInitialModel({seriesDir,originIndex,v5Dir,archiveDir}){
  if(!Number.isInteger(originIndex)||originIndex<0||originIndex>3)
    throw new Error('Unknown pre-registered baseline origin');
  const sourceShard=originIndex;
  const frozen=verifyFrozenOvernightModel(seriesDir,sourceShard);
  const manifest=JSON.parse(readFileSync(join(archiveDir,'manifest.json')));
  if(manifest.schema!=='rnad-i5-queued-evaluation-models-v1'||
     manifest.source_sha!=='f08a9587d6a7033546eab00ffa6479399ca71c2f'||
     manifest.engine_sha!==frozenEngine())
    throw new Error('Historical archived source manifest mismatch');
  for(const key of ['base128','base256','c005'])
    if(sha(join(archiveDir,key+'.json'))!==manifest.models[key])
      throw new Error('Original v2 source artifact SHA changed');
  const ident=JSON.parse(readFileSync(join(seriesDir,'series-identity.json')));
  const index=JSON.parse(readFileSync(join(seriesDir,'checkpoints/latest.json')));
  let modelPath,modelSHA,sourceModelSHA;
  if(originIndex<3){
    const name=BASELINE_ORIGINS[originIndex];
    if(ident.origin!==name||ident.mode!=='league'||
       ident.initial_model_file_sha256!==manifest.models[name]||
       index.metadata.initialization_sha256!==manifest.models[name])
      throw new Error('Wrong original v2 initial model');
    const frozenPool=JSON.parse(readFileSync(join(seriesDir,'league-pool.json')));
    const entry=frozenPool.models.find(x=>x.name===name);
    modelPath=join(seriesDir,'frozen',name+'-v3.json');
    modelSHA=sha(modelPath);sourceModelSHA=manifest.models[name];
    if(!entry||entry.sha256!==modelSHA)
      throw new Error('Not immutable deterministic v2->v3 initial migrator output');
  }else{
    if(ident.mode!=='selfplay'||ident.origin!=='battle-focus256-v5')
      throw new Error('Not v5 original warm-start');
    modelPath=join(v5Dir,'battle','trained-model.json');
    modelSHA=sha(modelPath);sourceModelSHA=modelSHA;
    const teacher=JSON.parse(readFileSync(join(v5Dir,'battle','teacher-report.json')));
    if(ident.initial_model_file_sha256!==modelSHA||
       index.metadata.initialization_sha256!==modelSHA||
       teacher.trained_model_sha256!==undefined&&
         !/^[a-f0-9]{64}$/.test(teacher.trained_model_sha256))
      throw new Error('Wrong v5 inherited warm-start artifact');
  }
  const model=new PersistentRNADInference(modelPath);
  if(model.featureSchema!==(originIndex===3?
      'rnad-three-phase-features-v5':'rnad-three-phase-features-v3'))
    throw new Error('Wrong original actor feature schema');
  return {origin:BASELINE_ORIGINS[originIndex],modelPath,modelSHA,sourceModelSHA};
}
function frozenEngine(){
  return JSON.parse(readFileSync('provenance/manifest.json')).sourceCommit;
}
export function evaluateInitialPart({seriesDir,originIndex,part,v5Dir,archiveDir}){
  if(!Number.isInteger(part)||part<0||part>=PARTS_PER_MODEL)
    throw new Error('Invalid exact seed part');
  const frozen=frozenInitialModel({seriesDir,originIndex,v5Dir,archiveDir});
  const seed=FIRST_HELDOUT_SEED+part*PAIRED_SEEDS_PER_PART;
  const rows=[];
  for(let k=0;k<PAIRED_SEEDS_PER_PART;k+=100){
    const count=Math.min(100,PAIRED_SEEDS_PER_PART-k);
    const r=evaluateAgainstOriginalAQ({modelPath:frozen.modelPath,
      seed:seed+k,pairs:count,phaseMode:'neural-all',switchGuard:'anti-loop-v1'});
    if(r.model_sha256!==frozen.modelSHA||r.aq_base_model_sha256!==AQ_SHA||
      r.matches!==count*2||r.champion_promoted!==false)
      throw new Error('Original AQ or initial model changed during holdout');
    rows.push(...r.rows);
    console.log(JSON.stringify({origin:frozen.origin,part,pairs_done:rows.length/2}));
  }
  return {schema:BASELINE_SCHEMA,origin:frozen.origin,part,
    original_training_run_id:ORIGINAL_OVERNIGHT_RUN,
    model_sha256:frozen.modelSHA,source_model_sha256:frozen.sourceModelSHA,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    aq_base_model_sha256:AQ_SHA,
    seed_start:seed,independent_seed_pairs:PAIRED_SEEDS_PER_PART,
    matches:rows.length,wins:rows.filter(x=>x.won).length,
    additional_training:false,champion_promoted:false,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const opt=new Map(),args=process.argv.slice(2);
  if(args.length!==12)throw new Error('Need six option pairs');
  for(let i=0;i<args.length;i+=2){
    if(!['--series','--origin-index','--part','--v5','--archive','--out'].includes(args[i])||
       opt.has(args[i]))throw new Error('Invalid baseline eval options');
    opt.set(args[i],args[i+1]);
  }
  const result=evaluateInitialPart({seriesDir:opt.get('--series'),
    originIndex:Number(opt.get('--origin-index')),part:Number(opt.get('--part')),
    v5Dir:opt.get('--v5'),archiveDir:opt.get('--archive')});
  writeFileSync(opt.get('--out'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({origin:result.origin,part:result.part,wins:result.wins,
    matches:result.matches}));
}
