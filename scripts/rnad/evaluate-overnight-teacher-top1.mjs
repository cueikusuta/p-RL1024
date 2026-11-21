// Frozen post-training 3-phase models vs SAME accepted, publicly visible AQ teacher decisions.
// This is agreement with public AQ teacher (not actual frozen AQ-base full-match actions).
// Same 512 independent full 4-win teacher Match seeds across models; GRU memory is seat-local.
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {PersistentRNADInference,argmaxRnad} from './infer.mjs';
import {verifyFrozenOvernightModel} from './evaluate-overnight-20-vs-aq.mjs';
import {AQ_WEIGHTS_SHA256 as AQ_SHA,AQ_STRATEGY_SELECTION_TEACHER_ID}
  from './aq-teacher-policy.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};
export const IMITATION_SCHEMA='rnad-i5-overnight-20-public-aq-teacher-agreement-v1';
export const FIRST_TEACHER_SEED=8700001;
export const TEACHER_SEED_CLUSTERS=512;
const requireOk=(condition,msg)=>{if(!condition)throw new Error('Invalid overnight teacher evaluation: '+msg);};
function traceSchema(mode){
  return mode==='selfplay'?'rnad-three-phase-features-v5':'rnad-three-phase-features-v3';
}
export function evaluateImitation({dir,shard,tracesDir,seedStart=FIRST_TEACHER_SEED,
  seedCount=TEACHER_SEED_CLUSTERS}){
  requireOk(Number.isInteger(seedStart)&&seedStart>=1&&
    Number.isInteger(seedCount)&&seedCount>0&&seedCount<=TEACHER_SEED_CLUSTERS,
    'invalid pre-registered teacher seed schedule');
  const frozen=verifyFrozenOvernightModel(dir,shard);
  const net=new PersistentRNADInference(frozen.modelPath);
  const phases=['draft','selection','battle'],rows=[];
  requireOk(net.featureSchema===traceSchema(frozen.mode),'wrong frozen model schema');
  for(let seed=seedStart;seed<seedStart+seedCount;seed++){
    const trace=JSON.parse(readFileSync(join(tracesDir,'teacher-'+seed+'.json')));
    requireOk(trace.schema==='rnad-aq-public-strategy-selection-three-phase-teacher-v1'&&
      trace.feature_schema===net.featureSchema&&
      trace.teacher===AQ_STRATEGY_SELECTION_TEACHER_ID&&
      trace.teacher_model_sha256===AQ_SHA&&
      trace.engine_sha===provenance.sourceCommit&&
      typeof trace.source_sha==='string'&&/^[0-9a-f]{40}$/.test(trace.source_sha)&&
      trace.seed===seed&&trace.completed===true&&
      [0,1].includes(trace.winner)&&trace.score[trace.winner]===4&&
      Array.isArray(trace.examples)&&
      trace.examples.length===trace.action_counts[0]+trace.action_counts[1]+
        4*trace.rounds,'teacher source, score or coverage changed');
    net.reset();
    const counts={draft:0,selection:0,battle:0};
    const correct={draft:0,selection:0,battle:0};
    const countsBySeat=[0,0];
    for(const event of trace.examples){
      const p=event.phase,seat=event.actor;
      requireOk(phases.includes(p)&&[0,1].includes(seat)&&
        Number.isInteger(event.choice)&&event.choice>=0&&
        Array.isArray(event.actions)&&event.choice<event.actions.length,
        'invalid accepted public-teacher action or seat');
      const output=net.infer(event.state,event.actions,seat);
      correct[p]+=Number(argmaxRnad(output.logits)===event.choice);
      counts[p]++;countsBySeat[seat]++;
    }
    requireOk(counts.draft===2*trace.rounds&&
      counts.selection===2*trace.rounds&&
      counts.battle===trace.action_counts[0]+trace.action_counts[1]&&
      countsBySeat.every(n=>n>0)&&
      phases.every(p=>counts[p]>0),'incorrect accepted teacher phase coverage');
    rows.push({seed,counts,correct});
    if((seed-seedStart+1)%64===0)
      console.log(JSON.stringify({shard,teacher_full_matches:rows.length,
        target_matches:seedCount,feature_schema:net.featureSchema}));
  }
  return {schema:IMITATION_SCHEMA,
    shard,mode:frozen.mode,origin:frozen.origin,
    trained_matches:frozen.trained_matches,
    updates_per_match:frozen.updates_per_match,
    model_sha256:frozen.model_sha256,
    checkpoint_sha256:frozen.checkpoint_sha256,
    feature_schema:net.featureSchema,
    teacher_id:AQ_STRATEGY_SELECTION_TEACHER_ID,
    teacher_sha256:AQ_SHA,engine_sha:provenance.sourceCommit,
    pre_registered_seed_start:seedStart,independent_complete_teacher_matches:seedCount,
    teacher_examples_are_strictly_accepted_public_own_view:true,
    teacher_top1_is_not_original_aq_full_match_policy:true,
    training_during_evaluation:false,champion_promoted:false,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const argv=process.argv.slice(2),opt=new Map();
  if(argv.length!==8)throw new Error('Need --series PATH --traces PATH --shard N --out FILE');
  for(let i=0;i<argv.length;i+=2){
    if(!['--series','--traces','--shard','--out'].includes(argv[i])||
      opt.has(argv[i])||!argv[i+1])throw new Error('Invalid imitation evaluation argument');
    opt.set(argv[i],argv[i+1]);
  }
  const result=evaluateImitation({dir:opt.get('--series'),tracesDir:opt.get('--traces'),
    shard:Number(opt.get('--shard'))});
  writeFileSync(opt.get('--out'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
}
