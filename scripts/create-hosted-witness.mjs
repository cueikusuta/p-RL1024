// Creates a compact, complete per-shard integrity witness while retaining
// the full artifact separately. Validation uses the same function as full aggregate.
import {writeFileSync,mkdirSync,readFileSync} from 'node:fs';
import {dirname,join} from 'node:path';
import {createHash} from 'node:crypto';
import {verifyHostedShard} from './hosted-shard-evidence.mjs';
import {planShards} from './hosted-shard-plan.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
const argv=process.argv.slice(2);
if(argv.length!==4||argv[0]!=='--index'||argv[2]!=='--out')throw Error('Expected --index N --out file');
const index=Number(argv[1]),out=argv[3];
const n=Number(process.env.RL_SHARDS||18);
if(!Number.isSafeInteger(n)||n<1||n>20)throw Error('Invalid witness shard count');
const plan=planShards({mode:'training',total:256,shards:n,seed:975020});
if(!Number.isSafeInteger(index)||index<0||index>=n||!out||
   !/^\d+$/.test(process.env.GITHUB_RUN_ID??'')||
   !/^\d+$/.test(process.env.GITHUB_RUN_ATTEMPT??'')||
   !/^[a-f0-9]{40}$/.test(process.env.GITHUB_SHA??''))
  throw Error('Invalid witness provenance');
const target=plan.include[index];
const verified=verifyHostedShard({root:'artifacts',artifact:'shard-'+index,
  index,offset:target.offset,model:'RL-Deep',opponent:'AQ-Base-001',base:975020});
const body={schema:'p-rl1024-hosted-shard-witness-v1',runId:process.env.GITHUB_RUN_ID,
  runAttempt:process.env.GITHUB_RUN_ATTEMPT,commit:process.env.GITHUB_SHA,
  model:'RL-Deep',opponent:'AQ-Base-001',seedRoot:975020,shards:n,totalMatches:256,
  index,sourceFingerprint:verified.sourceFingerprint,sourceCommit:verified.sourceCommit,
  row:verified.row,fileDigests:verified.fileDigests};
const report={...body,witnessSha256:sha(JSON.stringify(body))};
mkdirSync(dirname(out),{recursive:true});
writeFileSync(out,JSON.stringify(report)+'\n');
console.log('FAST_SHARD_WITNESS '+JSON.stringify({index,modelSha256:report.row.modelSha256,
  checkpointCount:report.row.checkpoints.length,witnessSha256:report.witnessSha256}));
