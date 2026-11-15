import {createHash} from 'node:crypto';
import {readFileSync,readdirSync,writeFileSync,mkdirSync} from 'node:fs';
import {join,dirname} from 'node:path';
import {verifyHostedShard} from './hosted-shard-evidence.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
const argv=process.argv.slice(2),arg=new Map();
if(argv.length%2)throw new Error('Expected pairs');
for(let i=0;i<argv.length;i+=2){if(!['--dir','--model','--opponent','--total','--shards','--seed','--out'].includes(argv[i])||arg.has(argv[i]))throw new Error('Unknown option');arg.set(argv[i],argv[i+1]);}
const [root,model,opponent,total,shards,seed,out]=['--dir','--model','--opponent','--total','--shards','--seed','--out'].map(k=>arg.get(k));
if(!root||!out||!['RL-Linear','RL-Deep'].includes(model)||!['AQ-Base-001','V2E','RL-Deep'].includes(opponent)||
   !Number.isInteger(Number(total))||!Number.isInteger(Number(shards))||!Number.isInteger(Number(seed)))throw new Error('Bad aggregate arguments');
const m=Number(total),n=Number(shards),base=Number(seed),artifacts=readdirSync(root,{withFileTypes:true}).filter(x=>x.isDirectory()).map(x=>x.name).sort((a,b)=>Number(a.match(/(?:^|-)shard-(\d+)$/)?.[1])-Number(b.match(/(?:^|-)shard-(\d+)$/)?.[1]));
if(artifacts.length!==n||n<1||n>20||m<9||m>256)throw new Error('Missing per-shard artifacts');
const rows=[];
let offset=0,trainingWins=0,sourceFingerprint=null,sourceCommit=null;
for(const [index,artifact] of artifacts.entries()){
  const verified=verifyHostedShard({
    root,artifact,index,offset,model,opponent,base,sourceFingerprint,sourceCommit
  });
  rows.push(verified.row);
  trainingWins+=verified.row.studentWins;
  sourceFingerprint=verified.sourceFingerprint;
  sourceCommit=verified.sourceCommit;
  offset+=count;
}
if(offset!==m)throw new Error('Missing planned training matches');
const start=Math.min(...rows.map(x=>x.startedUnix)),finish=Math.max(...rows.map(x=>x.finishedUnix));
const evidence={schema:'p-rl1024-hosted-training-aggregate-v1',model,opponent,seedRoot:base,totalMatches:m,
  shards:n,mode:'independent-shard-models-not-continuous-single-model',
  studentWins:trainingWins,studentLosses:m-trainingWins,
  wallclockSeconds:finish-start,shardReports:rows};
const report={...evidence,manifestSha256:sha(JSON.stringify(evidence))};
mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(report,null,2)+'\n');
console.log('HOSTED_TRAIN_AGGREGATE '+JSON.stringify(report));
