// Aggregate individually verified shard witnesses without downloading eighteen
// redundant multi-MB checkpoint archives. Full archives remain immutable Actions
// artifacts and their GitHub-provided ZIP digests are included in the manifest.
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {join,dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
import {planShards} from './hosted-shard-plan.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
const isSha=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const assert=(ok,message)=>{if(!ok)throw Error(message);};
export function validateHostedWitness(w,{index,planned,runId,runAttempt,commit}){
  assert(w&&w.schema==='p-rl1024-hosted-shard-witness-v1','Invalid witness schema');
  const {witnessSha256,...body}=w;
  assert(isSha(witnessSha256)&&sha(JSON.stringify(body))===witnessSha256,'Invalid witness body SHA');
  assert(w.runId===runId&&w.runAttempt===runAttempt&&w.commit===commit&&
    w.sourceCommit===commit&&w.index===index,'Invalid run/attempt/commit witness provenance');
  assert(w.model==='RL-Deep'&&w.opponent==='AQ-Base-001'&&w.seedRoot===975020&&
    w.shards===18&&w.totalMatches===256,'Invalid witness experiment scope');
  assert(isSha(w.sourceFingerprint),'Invalid witness source digest');
  const x=w.row;
  assert(x&&x.artifact==='shard-'+index&&x.offset===planned.offset&&
    x.count===planned.count&&x.seedStart===planned.seedStart&&
    x.seedEnd===planned.seedStart+planned.count-1,'Invalid witness seed/Match range');
  assert(Number.isSafeInteger(x.studentWins)&&x.studentWins>=0&&x.studentWins<=x.count&&
    Number.isSafeInteger(x.startedUnix)&&Number.isSafeInteger(x.finishedUnix)&&
    x.startedUnix>1577836800&&x.finishedUnix>=x.startedUnix&&
    typeof x.durationSeconds==='number'&&Number.isFinite(x.durationSeconds)&&
    x.durationSeconds>=0,'Invalid witness training time');
  assert(isSha(x.modelSha256)&&isSha(x.modelFileSha256)&&
    isSha(x.normalizedEvaluationDigestSha256),'Invalid witness model or evaluation digest');
  const f=w.fileDigests;
  assert(f&&Object.values(f).length===5&&
    Object.values(f).every(isSha)&&f.trainingSha256===x.modelFileSha256,
    'Missing source file SHA evidence');
  assert(Array.isArray(x.checkpoints)&&x.checkpoints.length===Math.floor(x.count/8),
    'Incomplete checkpoint sequence');
  for(let k=0;k<x.checkpoints.length;k++){
    const cp=x.checkpoints[k],match=(k+1)*8;
    assert(cp.name==='training-m'+match+'.json'&&cp.completedMatches===match&&
      isSha(cp.fileSha256)&&isSha(cp.stateSha256),
      'Invalid checkpoint name or integrity digest');
  }
  return x;
}
export function mergeHostedWitnesses(root,{runId,runAttempt,commit}){
  assert(/^\d+$/.test(runId??'')&&/^\d+$/.test(runAttempt??'')&&
    /^[a-f0-9]{40}$/.test(commit??''),'Invalid run identity');
  const plan=planShards({mode:'training',total:256,shards:18,seed:975020});
  const meta=JSON.parse(readFileSync(join(root,'full-artifacts.json'),'utf8'));
  assert(meta.runId===runId&&meta.runAttempt===runAttempt&&
    Array.isArray(meta.artifacts)&&meta.artifacts.length===18,
    'Incomplete full checkpoint archive metadata');
  let fingerprint=null,wins=0;
  const rows=[];
  for(let i=0;i<18;i++){
    const name=`i1-fast-witness-${runId}-${runAttempt}-shard-${i}`;
    const w=JSON.parse(readFileSync(join(root,name,'witness.json'),'utf8'));
    const row=validateHostedWitness(w,{index:i,planned:plan.include[i],runId,runAttempt,commit});
    if(fingerprint===null)fingerprint=w.sourceFingerprint;
    assert(fingerprint===w.sourceFingerprint,'Inconsistent shard source fingerprints');
    const archive=meta.artifacts[i];
    assert(archive&&archive.name===`i1-fast-matrix-${runId}-${runAttempt}-shard-${i}`&&
      Number.isSafeInteger(archive.id)&&archive.id>0&&
      /^sha256:[a-f0-9]{64}$/.test(archive.digest)&&
      Number.isSafeInteger(archive.size)&&archive.size>0,
      'Missing immutable full checkpoint artifact or invalid GitHub digest');
    rows.push({...row,sourceFileDigests:w.fileDigests,
      fullArchiveSha256:archive.digest.slice(7),fullArchiveArtifactId:archive.id});
    wins+=row.studentWins;
  }
  const start=Math.min(...rows.map(x=>x.startedUnix)),end=Math.max(...rows.map(x=>x.finishedUnix));
  const evidence={schema:'p-rl1024-hosted-training-aggregate-v1',
    model:'RL-Deep',opponent:'AQ-Base-001',seedRoot:975020,totalMatches:256,
    shards:18,mode:'independent-shard-models-not-continuous-single-model',
    witnessMode:'full-raw-validated-on-producer-and-retained-as-immutable-github-artifacts',
    sourceFingerprint:fingerprint,studentWins:wins,studentLosses:256-wins,
    wallclockSeconds:end-start,shardReports:rows};
  return {...evidence,manifestSha256:sha(JSON.stringify(evidence))};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  if(process.argv.length!==4)throw Error('Usage: node scripts/merge-hosted-witnesses.mjs folder output.json');
  const report=mergeHostedWitnesses(process.argv[2],{runId:process.env.GITHUB_RUN_ID,
    runAttempt:process.env.GITHUB_RUN_ATTEMPT,commit:process.env.GITHUB_SHA});
  mkdirSync(dirname(process.argv[3]),{recursive:true});
  writeFileSync(process.argv[3],JSON.stringify(report,null,2)+'\n');
  console.log('HOSTED_TRAIN_AGGREGATE '+JSON.stringify(report));
}
