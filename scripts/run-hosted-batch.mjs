import {spawn} from 'node:child_process';
import {mkdirSync,writeFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {planShards} from './hosted-shard-plan.mjs';

function parse(args){
  if(args.length%2)throw new Error('Expected key/value pairs');
  const m=new Map();
  for(let i=0;i<args.length;i+=2) {
    if(!['--job-index','--parallel','--shards','--total','--seed','--out'].includes(args[i])||m.has(args[i]))
      throw new Error('Unexpected batch option');
    m.set(args[i],args[i+1]);
  }
  const config={index:Number(m.get('--job-index')),parallel:Number(m.get('--parallel')),
    shards:Number(m.get('--shards')),total:Number(m.get('--total')),
    seed:Number(m.get('--seed')),out:m.get('--out')};
  if(!Number.isSafeInteger(config.index)||config.index<0||config.index>19||
    !Number.isSafeInteger(config.parallel)||config.parallel<1||config.parallel>4||
    !Number.isSafeInteger(config.shards)||config.shards<1||config.shards>20||
    config.shards%config.parallel!==0||
    config.index>=config.shards/config.parallel||!config.out)
    throw new Error('Invalid bounded batch config');
  config.plan=planShards({mode:'training',total:config.total,shards:config.shards,seed:config.seed});
  return config;
}
function runShardNode(env){
  return new Promise((resolve,reject)=>{
    const child=spawn(process.execPath,['scripts/run-experiment.mjs'],{env,stdio:'inherit'});
    let failed=false;
    child.on('error',error=>{failed=true;reject(error);});
    child.on('close',(code,signal)=>{
      if(failed)return;
      if(code!==0)reject(new Error('Shard child failed '+code+' '+signal));
      else resolve();
    });
  });
}
const cfg=parse(process.argv.slice(2));
const shardJobs=cfg.plan.include.slice(cfg.index*cfg.parallel,(cfg.index+1)*cfg.parallel);
if(shardJobs.length!==cfg.parallel)throw new Error('Incomplete local shard batch');
const all=await Promise.allSettled(shardJobs.map(async shard=>{
  const root=resolve(cfg.out,'shard-'+shard.index);
  mkdirSync(root,{recursive:true});
  const started=Date.now();
  let exitCode=0;
  try {
    await runShardNode({...process.env,RL_MODEL:'RL-Deep',RL_OPPONENT:'AQ-Base-001',
      RL_MATCHES:String(shard.count),RL_SEED:String(shard.seedStart),RL_OUTPUT_ROOT:root});
  } catch(error){ exitCode=1;console.error('BATCH_SHARD_FAIL '+shard.index+': '+String(error)); }
  const ended=Date.now();
  const result={schema:'p-rl1024-hosted-shard-clock-v1',
    model:'RL-Deep',opponent:'AQ-Base-001',seed:shard.seedStart,
    offset:shard.offset,count:shard.count,shardIndex:shard.index,
    durationSeconds:(ended-started)/1000,
    startedUnix:Math.floor(started/1000),finishedUnix:Math.ceil(ended/1000),
    exitCode,runId:process.env.GITHUB_RUN_ID||null};
  writeFileSync(join(root,'wallclock-shard.json'),JSON.stringify(result,null,2)+'\n');
  if(exitCode)throw new Error('Shard '+shard.index+' failure');
  return result;
}));
const bad=all.filter(x=>x.status==='rejected');
if(bad.length)throw new Error('Batch failed '+bad.map(x=>String(x.reason)).join('; '));
console.log('BATCH_FULL_MATCHES '+JSON.stringify({job:cfg.index,shards:cfg.parallel,
  matches:shardJobs.reduce((sum,s)=>sum+s.count,0),results:all.map(x=>x.value)}));
