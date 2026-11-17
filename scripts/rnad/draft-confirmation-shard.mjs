// One idempotent authority-evaluation shard. Each partition is all-v3
// and exactly one frozen candidate over the same cryptographically mapped seeds.
import {mkdirSync,writeFileSync,renameSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {evaluateDraftDeepSetsVsAQ}
  from './evaluate-deepsets-draft-vs-aq.mjs';
import {verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {weightsSha256 as ORIGINAL_AQ_SHA}
  from '../../models/aq-base.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};
import {readConfirmationPlan,plannedShard,
  sha256File,CONFIRM_SHARD_SCHEMA,validateConfirmationShard}
  from './draft-confirmation-protocol.mjs';

export function runConfirmationShard({planPath,expectedPlanSha,
  expectedSourceSha,shardIndex,v3Path,v4Path,draftPath,outDir}){
  const {plan,plan_sha256}=readConfirmationPlan(planPath);
  if(plan_sha256!==expectedPlanSha)
    throw new Error('Preregistered plan bytes have changed');
  const source=verifiedRunningSourceSha();
  if(source!==expectedSourceSha||provenance.sourceCommit!==plan.engine_sha||
     ORIGINAL_AQ_SHA!==plan.aq_sha256||
     sha256File(v3Path)!==plan.v3_sha256||
     sha256File(v4Path)!==plan.v4_sha256||
     sha256File(draftPath)!==plan.candidate_sha256)
    throw new Error('Evaluation source, frozen engine or model SHA mismatch');
  const spec=plannedShard(plan,shardIndex);
  const opts={v3Path,v4Path,draftPath,
    expectedDraftSha:plan.candidate_sha256,
    seed:spec.seeds[0],pairs:spec.seeds.length,
    explicitSeeds:spec.seeds};
  const baseline=evaluateDraftDeepSetsVsAQ({...opts,mode:plan.baseline_mode});
  const candidate=evaluateDraftDeepSetsVsAQ({...opts,mode:plan.candidate_mode});
  const shard={schema:CONFIRM_SHARD_SCHEMA,
    plan_sha256,source_sha:source,
    shard_index:spec.index,start:spec.start,
    seeds:spec.seeds,match_count:4*spec.seeds.length,
    baseline,candidate};
  validateConfirmationShard(shard,plan,plan_sha256,source);
  mkdirSync(outDir,{recursive:true});
  const dest=join(outDir,'shard.json'),tmp=join(outDir,'shard.json.tmp');
  writeFileSync(tmp,JSON.stringify(shard,null,2)+'\n',{flag:'wx'});
  renameSync(tmp,dest);
  return {schema:CONFIRM_SHARD_SCHEMA,index:spec.index,
    complete_seed_blocks:spec.seeds.length,
    complete_four_win_matches:shard.match_count,
    plan_sha256,source_sha:source};
}
function options(argv,expected){
  if(argv.length!==expected.length*2)
    throw new Error('Missing required confirmation shard parameters');
  const out=new Map();
  for(let i=0;i<argv.length;i+=2){
    if(!expected.includes(argv[i])||out.has(argv[i])||!argv[i+1])
      throw new Error('Invalid duplicate shard option');
    out.set(argv[i],argv[i+1]);
  }
  return out;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const p=options(process.argv.slice(2),
    ['--plan','--plan-sha','--source-sha','--index','--v3','--v4','--draft','--out']);
  const result=runConfirmationShard({
    planPath:p.get('--plan'),
    expectedPlanSha:p.get('--plan-sha'),
    expectedSourceSha:p.get('--source-sha'),
    shardIndex:Number(p.get('--index')),
    v3Path:p.get('--v3'),v4Path:p.get('--v4'),
    draftPath:p.get('--draft'),outDir:p.get('--out')
  });
  // Do not expose interim win rates or enable manual result-based stopping.
  console.log(JSON.stringify(result));
}
