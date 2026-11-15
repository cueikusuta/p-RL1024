import {createHash} from 'node:crypto';
import {readFileSync,readdirSync,writeFileSync,mkdirSync} from 'node:fs';
import {join,dirname} from 'node:path';
import {compare} from './verify-evaluation.mjs';
import {mixSeed,weightsSha256} from '../models/aq-runner.mjs';
import {verifyTrainingCheckpoints,verifyShardClock} from './hosted-proof.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
const argv=process.argv.slice(2),arg=new Map();
if(argv.length%2)throw new Error('Expected pairs');
for(let i=0;i<argv.length;i+=2){if(!['--dir','--model','--opponent','--total','--shards','--seed','--out'].includes(argv[i])||arg.has(argv[i]))throw new Error('Unknown option');arg.set(argv[i],argv[i+1]);}
const [root,model,opponent,total,shards,seed,out]=['--dir','--model','--opponent','--total','--shards','--seed','--out'].map(k=>arg.get(k));
if(!root||!out||!['RL-Linear','RL-Deep'].includes(model)||!['AQ-Base-001','V2E','RL-Deep'].includes(opponent)||
   !Number.isInteger(Number(total))||!Number.isInteger(Number(shards))||!Number.isInteger(Number(seed)))throw new Error('Bad aggregate arguments');
const m=Number(total),n=Number(shards),base=Number(seed),artifacts=readdirSync(root,{withFileTypes:true}).filter(x=>x.isDirectory()).map(x=>x.name).sort((a,b)=>Number(a.match(/-shard-(\d+)$/)?.[1])-Number(b.match(/-shard-(\d+)$/)?.[1]));
if(artifacts.length!==n||n<1||n>20||m<9||m>256)throw new Error('Missing per-shard artifacts');
const sub=opponent==='AQ-Base-001'?model+'-vs-aq':model==='RL-Linear'?'i975':'rl-deep';
const rows=[];
let offset=0,trainingWins=0,sourceFingerprint=null,sourceCommit=null;
for(const [index,artifact] of artifacts.entries()){
  if(!artifact.endsWith('-shard-'+index))throw new Error('Missing or duplicate shard index');
  const folder=join(root,artifact,sub);
  const trainingPath=join(folder,'training.json'),bytes=readFileSync(trainingPath);
  const training=JSON.parse(bytes),proof=JSON.parse(readFileSync(join(folder,'comparison.json')));
  const actual=compare(...['a','b'].map(x=>JSON.parse(readFileSync(join(folder,'eval-'+x+'.json')))));
  if(actual.normalizedOutcomeSha256!==proof.normalizedOutcomeSha256||actual.modelSha256!==proof.modelSha256||
    training.modelSha256!==sha(JSON.stringify(training.model))||proof.modelSha256!==training.modelSha256||
    sha(bytes)!==proof.modelFileSha256)throw new Error('Weight or evaluation SHA mismatch: '+artifact);
  const t=JSON.parse(readFileSync(join(root,artifact,'wallclock-shard.json')));
  const count=t.count,rootSeed=base+offset,history=training.history||training.matches;
  verifyShardClock(t,{model,opponent,index,offset,count,seed:rootSeed,runId:process.env.GITHUB_RUN_ID});
  if(!Number.isSafeInteger(count)||count<9||count>256||
    training.completedMatches!==count||training.seedRoot!==rootSeed||
    !Array.isArray(history)||history.length!==count||
    !/^[a-f0-9]{64}$/.test(training.sourceSha256??'')||
    (opponent==='AQ-Base-001'&&training.opponentModelSha256!==weightsSha256)||
    (process.env.GITHUB_SHA&&training.sourceCommit!==process.env.GITHUB_SHA))
    throw new Error('Shard source/seed/completion mismatch: '+artifact);
  if(sourceFingerprint===null){sourceFingerprint=training.sourceSha256;sourceCommit=training.sourceCommit??null;}
  if(training.sourceSha256!==sourceFingerprint||(training.sourceCommit??null)!==sourceCommit)
    throw new Error('Incompatible source fingerprint or commit across shards');
  const tag=opponent==='AQ-Base-001'?0xa975:model==='RL-Deep'?0x976:0x975;
  for(let j=0;j<history.length;j++){
    const r=history[j];
    if(!r||r.seed!==mixSeed(rootSeed,j,tag)||r.studentSeat!==j%2)
      throw new Error('Training Match seed/seat mismatch');
    if(![0,1].includes(r.winner)||!Array.isArray(r.score)||r.score.length!==2||
      r.score[r.winner]!==4||!Number.isSafeInteger(r.rounds)||r.rounds<4)
      throw new Error('Invalid completed four-win Match');
  }
  const wins=history.filter(r=>r.studentSeat!==undefined&&r.winner===r.studentSeat).length;
  if(opponent==='AQ-Base-001'&&wins+history.filter(r=>r.studentSeat!==undefined&&r.winner!==r.studentSeat).length!==count)
    throw new Error('Missing AQ outcome');
  trainingWins+=opponent==='AQ-Base-001'||model==='RL-Deep'?wins:training.studentWins??wins;
  const checkpoints=verifyTrainingCheckpoints(folder,{
    model,opponent,count,seed:rootSeed,sourceSha256:training.sourceSha256
  });
  rows.push({artifact,offset,count,seedStart:rootSeed,seedEnd:rootSeed+count-1,
    durationSeconds:t.durationSeconds,startedUnix:t.startedUnix,finishedUnix:t.finishedUnix,
    studentWins:opponent==='AQ-Base-001'||model==='RL-Deep'?wins:training.studentWins??wins,
    modelSha256:training.modelSha256,modelFileSha256:proof.modelFileSha256,
    normalizedEvaluationDigestSha256:actual.normalizedOutcomeSha256,
    checkpoints});
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
