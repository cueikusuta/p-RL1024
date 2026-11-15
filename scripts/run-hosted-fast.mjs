import {spawn} from 'node:child_process';
import {mkdirSync,writeFileSync} from 'node:fs';
import {join,dirname,resolve} from 'node:path';
import {performance} from 'node:perf_hooks';
import {planShards} from './hosted-shard-plan.mjs';

function options(argv){
  if(argv.length%2)throw new Error('Expected key/value arguments');
  const m=new Map();
  for(let i=0;i<argv.length;i+=2){
    if(!['--model','--opponent','--total','--seed','--shards','--parallel','--out'].includes(argv[i])||m.has(argv[i]))
      throw new Error('Unknown or repeated argument');
    m.set(argv[i],argv[i+1]);
  }
  const o={model:m.get('--model'),opponent:m.get('--opponent'),total:Number(m.get('--total')),
    seed:Number(m.get('--seed')),shards:Number(m.get('--shards')),parallel:Number(m.get('--parallel')),
    out:m.get('--out')};
  if(!['RL-Linear','RL-Deep'].includes(o.model)||
     !['V2E','RL-Deep','AQ-Base-001'].includes(o.opponent)||
     !Number.isSafeInteger(o.parallel)||o.parallel<1||o.parallel>4||!o.out)
    throw new Error('Invalid CPU-only fast training configuration');
  if((o.model==='RL-Linear'&&o.opponent==='RL-Deep')||
     (o.model==='RL-Deep'&&o.opponent==='V2E'))
    throw new Error('Unsupported model/opponent combination');
  o.plan=planShards({mode:'training',total:o.total,shards:o.shards,seed:o.seed});
  return o;
}
function runNode(args,env){
  return new Promise((ok,fail)=>{
    const child=spawn(process.execPath,args,{env,stdio:'inherit'});
    let error=null;
    child.once('error',e=>{error=e;});
    child.once('close',(code,signal)=>error||code!==0?fail(error||new Error('Node child failed: '+code+'/'+signal)):ok());
  });
}
const o=options(process.argv.slice(2));
const root=resolve(dirname(o.out),'fast-training-shards');
mkdirSync(root,{recursive:true});
const start=performance.now(),report=[];
let cursor=0;
async function lane(){
  for(;;) {
    const index=cursor++;
    if(index>=o.plan.include.length)return;
    const shard=o.plan.include[index],dir=join(root,'shard-'+shard.index);
    mkdirSync(dir,{recursive:true});
    const startedMs=Date.now(),started=performance.now();
    let code=0;
    try {
      await runNode(['scripts/run-experiment.mjs'],{...process.env,
        RL_MODEL:o.model,RL_OPPONENT:o.opponent,RL_MATCHES:String(shard.count),
        RL_SEED:String(shard.seedStart),RL_OUTPUT_ROOT:dir});
    } catch(err) {
      code=1;
      console.error('FAST_SHARD_FAILED '+shard.index+' '+String(err));
    }
    const finishedMs=Date.now(),elapsed=(performance.now()-started)/1000;
    const clock={schema:'p-rl1024-hosted-shard-clock-v1',model:o.model,opponent:o.opponent,
      seed:shard.seedStart,offset:shard.offset,count:shard.count,shardIndex:shard.index,
      durationSeconds:elapsed,startedUnix:Math.floor(startedMs/1000),
      finishedUnix:Math.ceil(finishedMs/1000),exitCode:code,runId:process.env.GITHUB_RUN_ID||null};
    writeFileSync(join(dir,'wallclock-shard.json'),JSON.stringify(clock,null,2)+'\n');
    if(code)throw new Error('Shard '+shard.index+' failed');
    report.push({index:shard.index,matches:shard.count,elapsedSeconds:elapsed});
  }
}
await Promise.all(Array.from({length:Math.min(o.parallel,o.shards)},()=>lane()));
await runNode(['scripts/merge-hosted-training.mjs','--dir',root,'--model',o.model,
  '--opponent',o.opponent,'--total',String(o.total),'--shards',String(o.shards),
  '--seed',String(o.seed),'--out',o.out],process.env);
const seconds=(performance.now()-start)/1000;
console.log('FAST_TRAINING_WALLCLOCK '+JSON.stringify({seconds,model:o.model,
  opponent:o.opponent,total:o.total,shards:o.shards,parallel:o.parallel,shardTimes:report}));
