import {createHash} from 'node:crypto';
import {mkdirSync,writeFileSync} from 'node:fs';
import {dirname,join} from 'node:path';
import {performance} from 'node:perf_hooks';
import {availableParallelism} from 'node:os';
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
import {createModel,validateModel,match,mixSeed,weightsSha256} from '../models/aq-runner.mjs';
const hash=s=>createHash('sha256').update(s).digest('hex');
function parse(argv) {
  if(argv.length%2)throw new Error('Expected option/value pairs');
  const m=new Map();
  for(let i=0;i<argv.length;i+=2){if(!['--model','--opponent','--total','--offset','--count','--seed','--workers','--out'].includes(argv[i])||m.has(argv[i]))throw new Error('Invalid argument');m.set(argv[i],argv[i+1]);}
  const o={model:m.get('--model'),opponent:m.get('--opponent'),total:Number(m.get('--total')),offset:Number(m.get('--offset')),
    count:Number(m.get('--count')),seed:Number(m.get('--seed')),workers:Number(m.get('--workers')),out:m.get('--out')};
  if(!['RL-Linear','RL-Deep'].includes(o.model)||o.opponent!=='AQ-Base-001'||
    !Number.isSafeInteger(o.total)||o.total<1||o.total>100000||
    !Number.isSafeInteger(o.offset)||o.offset<0||
    !Number.isSafeInteger(o.count)||o.count<1||o.offset+o.count>o.total||
    !Number.isSafeInteger(o.seed)||o.seed<1||o.seed+o.total-1>0xffffffff||
    !Number.isSafeInteger(o.workers)||o.workers<1||o.workers>4||!o.out)
    throw new Error('Invalid evaluation configuration');
  return o;
}
function partition(o,indices){
  const model=createModel(o.model,o.seed);
  validateModel(o.model,model);
  const before=hash(JSON.stringify(model)),records=[];
  for(const index of indices) {
    const seed=mixSeed(o.seed,index,0xe975),seat=index%2;
    const result=match({id:o.model,model,seed,seat,train:false});
    if(result.pureRlStats[seat]?.train||result.pureRlStats[seat]?.updates!==0)
      throw new Error('Frozen evaluation mutated weights');
    records.push({index,seed,seat,winner:result.winner,won:Number(result.winner===seat),
      score:result.score,rounds:result.rounds,actionCounts:result.actionCounts});
  }
  if(hash(JSON.stringify(model))!==before)throw new Error('Model changed during frozen evaluation');
  return {modelSha256:before,records};
}
if(!isMainThread){
  try{parentPort.postMessage({ok:true,result:partition(workerData.options,workerData.indices)});}
  catch(error){parentPort.postMessage({ok:false,error:String(error.stack||error)});}
}else{
  const o=parse(process.argv.slice(2));
  const startedAtMs=Date.now(),begin=performance.now();
  const indices=Array.from({length:o.count},(_,i)=>i+o.offset);
  const model=createModel(o.model,o.seed);
  validateModel(o.model,model);
  const modelText=JSON.stringify(model)+'\n',modelSha256=hash(JSON.stringify(model));
  const chunks=Array.from({length:Math.min(o.workers,availableParallelism(),o.count)},()=>[]);
  indices.forEach((index,i)=>chunks[i%chunks.length].push(index));
  const results=chunks.length===1?[partition(o,chunks[0])]:await Promise.all(chunks.map(indices=>new Promise((resolve,reject)=>{
    const w=new Worker(new URL(import.meta.url),{workerData:{options:o,indices}});
    let received=false;
    w.on('message',msg=>{if(received){reject(new Error('Duplicate worker response'));return;}received=true;
      if(msg.ok)resolve(msg.result);else reject(new Error(msg.error));});
    w.on('error',reject);
    w.on('exit',code=>{if(code!==0||!received)reject(new Error('Worker exited before result: '+code));});
  })));
  if(results.some(x=>x.modelSha256!==modelSha256))throw new Error('Cross-worker model SHA mismatch');
  const records=results.flatMap(x=>x.records).sort((a,b)=>a.index-b.index);
  if(records.length!==o.count||records.some((r,i)=>r.index!==o.offset+i||
    r.seed!==mixSeed(o.seed,r.index,0xe975)||r.seat!==r.index%2))throw new Error('Missing, duplicate or incorrectly seeded Match');
  if(new Set(records.map(x=>x.seed)).size!==records.length)throw new Error('Duplicate Match seed');
  const durationMs=performance.now()-begin;
  const report={schema:'p-rl1024-hosted-frozen-evaluation-v1',model:o.model,opponent:o.opponent,modelStatus:'deterministic-initial-frozen',
    modelSha256,modelFileSha256:hash(modelText),opponentModelSha256:weightsSha256,
    sourceCommit:process.env.GITHUB_SHA||null,runId:process.env.GITHUB_RUN_ID||null,
    seedRoot:o.seed,total:o.total,offset:o.offset,count:o.count,seedStart:o.seed+o.offset,
    seedEnd:o.seed+o.offset+o.count-1,workers:chunks.length,startedAtMs,durationMs,
    outcomeDigestSha256:hash(JSON.stringify(records)),
    wins:records.reduce((n,r)=>n+r.won,0),records};
  mkdirSync(dirname(o.out),{recursive:true});
  writeFileSync(join(dirname(o.out),'model.json'),modelText);
  writeFileSync(o.out,JSON.stringify(report,null,2)+'\n');
  console.log('HOSTED_EVAL '+JSON.stringify({model:o.model,offset:o.offset,count:o.count,
    workers:chunks.length,durationMs,modelSha256,opponentModelSha256:weightsSha256,outcomeDigestSha256:report.outcomeDigestSha256,wins:report.wins}));
}
