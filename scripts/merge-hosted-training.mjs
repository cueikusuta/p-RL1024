import {createHash} from 'node:crypto';
import {readFileSync,readdirSync,writeFileSync,mkdirSync} from 'node:fs';
import {join,dirname} from 'node:path';
import {compare} from './verify-evaluation.mjs';
import {mixSeed} from '../models/aq-runner.mjs';
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
let offset=0,trainingWins=0;
for(const artifact of artifacts){
  const folder=join(root,artifact,sub);
  const trainingPath=join(folder,'training.json'),bytes=readFileSync(trainingPath);
  const training=JSON.parse(bytes),proof=JSON.parse(readFileSync(join(folder,'comparison.json')));
  const actual=compare(...['a','b'].map(x=>JSON.parse(readFileSync(join(folder,'eval-'+x+'.json')))));
  if(actual.normalizedOutcomeSha256!==proof.normalizedOutcomeSha256||actual.modelSha256!==proof.modelSha256||
    training.modelSha256!==sha(JSON.stringify(training.model))||proof.modelSha256!==training.modelSha256||
    sha(bytes)!==proof.modelFileSha256)throw new Error('Weight or evaluation SHA mismatch: '+artifact);
  const t=JSON.parse(readFileSync(join(root,artifact,'wallclock-shard.json')));
  const count=t.count,rootSeed=base+offset,history=training.history||training.matches;
  if(t.model!==model||t.opponent!==opponent||t.seed!==rootSeed||t.offset!==offset||
    t.exitCode!==0||t.count<9||training.completedMatches!==count||
    training.seedRoot!==rootSeed||!Array.isArray(history)||history.length!==count)
    throw new Error('Shard seed range or completion mismatch: '+artifact);
  const tag=opponent==='AQ-Base-001'?0xa975:model==='RL-Deep'?0x976:0x975;
  for(let j=0;j<history.length;j++){
    const r=history[j];
    if(r.seed!==mixSeed(rootSeed,j,tag))throw new Error('Training Match seed mismatch');
    if(r.score&&r.score[r.winner]!==4)throw new Error('Invalid training score');
  }
  const wins=history.filter(r=>r.studentSeat!==undefined&&r.winner===r.studentSeat).length;
  if(opponent==='AQ-Base-001'&&wins+history.filter(r=>r.studentSeat!==undefined&&r.winner!==r.studentSeat).length!==count)
    throw new Error('Missing AQ outcome');
  trainingWins+=opponent==='AQ-Base-001'||model==='RL-Deep'?wins:training.studentWins??wins;
  const checkpoints=readdirSync(folder).filter(x=>/^training-m\d+\.json$/.test(x)).map(name=>{
    const data=readFileSync(join(folder,name)),cp=JSON.parse(data),{checkpointSha256,...state}=cp;
    if(checkpointSha256&&sha(JSON.stringify(state))!==checkpointSha256)throw new Error('Checkpoint SHA mismatch');
    return {name,fileSha256:sha(data),stateSha256:checkpointSha256??null};
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
const report={schema:'p-rl1024-hosted-training-aggregate-v1',model,opponent,seedRoot:base,totalMatches:m,
  shards:n,mode:'independent-shard-models-not-continuous-single-model',
  studentWins:trainingWins,studentLosses:m-trainingWins,
  wallclockSeconds:finish-start,shardReports:rows,
  manifestSha256:sha(JSON.stringify(rows))};
mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(report,null,2)+'\n');
console.log('HOSTED_TRAIN_AGGREGATE '+JSON.stringify(report));
