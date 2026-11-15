import {spawnSync,spawn} from 'node:child_process';
import {Worker} from 'node:worker_threads';
import {mkdirSync,writeFileSync,readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {compare} from './verify-evaluation.mjs';
import {getModel} from '../models/registry.mjs';
import {loadModel} from '../models/linear.mjs';
import {loadModel as loadDeepModel} from '../models/deep.mjs';

const args=process.argv.slice(2);
if (args.length%2) throw new Error('Expected --model ID and --opponent ID');
const options=new Map();
for (let i=0;i<args.length;i+=2) {
  if (!['--model','--opponent'].includes(args[i]) || options.has(args[i])) throw new Error('Unknown or duplicate option');
  options.set(args[i],args[i+1]);
}
const modelId=options.get('--model')||process.env.RL_MODEL||'RL-Linear';
const opponentId=options.get('--opponent')||process.env.RL_OPPONENT||'V2E';
const selected=getModel(modelId),opponent=getModel(opponentId);
if (selected.status!=='trainable' || !opponent.opponent || !((selected.id==='RL-Linear' && opponent.id==='V2E')||(selected.id==='RL-Deep' && opponent.id==='RL-Deep')||(['RL-Linear','RL-Deep'].includes(selected.id)&&opponent.id==='AQ-Base-001'))) throw new Error('Unsupported model/opponent combination');
const matches = Number(process.env.RL_MATCHES || '9');
const seed = Number(process.env.RL_SEED || '975020');
if (!Number.isSafeInteger(matches) || matches < 9 || matches > 256 || !Number.isSafeInteger(seed) || seed < 1 || seed > 0xffffffff) throw new Error('RL_MATCHES must be 9–256 and RL_SEED must be 1–4294967295');
const artifactRoot=process.env.RL_OUTPUT_ROOT||'artifacts';
const dir = opponent.id==='AQ-Base-001'?artifactRoot+'/'+selected.id+'-vs-aq':selected.id==='RL-Linear'?artifactRoot+'/i975':artifactRoot+'/rl-deep';
mkdirSync(dir,{recursive:true});
function run(script,args) {
  const result=spawnSync(process.execPath,[script,...args],{stdio:'inherit',timeout:290*60*1000});
  if (result.error || result.status !== 0) throw result.error || new Error(`${script} exited ${result.status}`);
}
async function runFastWorker(script,argv,env=process.env) {
  const url=new URL('./'+script.replace(/^scripts\//,''),import.meta.url);
  return new Promise((resolve,reject)=>{
    const worker=new Worker(url,{argv,env});
    let error=null;
    worker.once('error',e=>{error=e;});
    worker.once('exit',code=>{
      if(error)reject(error);
      else if(code!==0)reject(new Error('Fast training worker '+script+' exited '+code));
      else resolve();
    });
  });
}
const fastWorker=process.env.RL_FAST_WORKER==='1';
const fastInline=process.env.RL_FAST_INLINE==='1';
if(fastInline&&!fastWorker)throw new Error('Inline training requires isolated fast Worker');
// A fast Worker can sequentially run training and frozen evaluations in one
// module realm. Save argv through awaited imports: no concurrent globals.
async function runInline(script,argv,tag){
  const old=process.argv;
  if(!script.startsWith('scripts/'))throw new Error('Unsafe inline script path');
  const url=new URL('./'+script.slice('scripts/'.length),import.meta.url);
  url.searchParams.set('inline',tag);
  process.argv=[process.execPath,url.pathname,...argv];
  try {await import(url.href);} finally {process.argv=old;}
}
const model=`${dir}/training.json`;
const trainArgs=['--matches',String(matches),'--seed',String(seed),'--checkpoint-every','8','--out',model];
if(opponent.id==='AQ-Base-001')trainArgs.unshift('--model',selected.id);
else if(selected.id==='RL-Linear')trainArgs.unshift('--all-phases','true');
const trainer=opponent.id==='AQ-Base-001'?'scripts/train-vs-aq.mjs':selected.trainingEntry;
if(fastInline)await runInline(trainer,trainArgs,'training');
else if(fastWorker)await runFastWorker(trainer,trainArgs);
else run(trainer,trainArgs);
(selected.id==='RL-Linear'?loadModel:loadDeepModel)(JSON.stringify(JSON.parse(readFileSync(model,'utf8')).model));
const evaluator=opponent.id==='AQ-Base-001'?'scripts/evaluate-vs-aq.mjs':selected.id==='RL-Linear'?'scripts/evaluate-linear.mjs':'scripts/evaluate-deep.mjs';
function evaluateIndependent(name) {
  if(fastWorker)return runFastWorker(evaluator,['--model',model,'--purpose','smoke','--blocks','2','--out',dir+'/eval-'+name+'.json']);
  return new Promise((resolve,reject)=>{
    const child=spawn(process.execPath,[evaluator,'--model',model,'--purpose','smoke',
      '--blocks','2','--out',dir+'/eval-'+name+'.json'],{stdio:'inherit'});
    let rejected=false;
    child.once('error',error=>{rejected=true;reject(error);});
    child.once('close',(code,signal)=>{
      if(rejected)return;
      if(code!==0)reject(new Error('Frozen independent evaluation '+name+' failed '+code+'/'+signal));
      else resolve();
    });
  });
}
// Run the two independent smoke evaluations concurrently on distinct Node
// processes, preserving both 4-Match outcomes and the normalized digest gate.
if(fastInline){
  for(const name of ['a','b'])await runInline(evaluator,['--model',model,'--purpose','smoke',
    '--blocks','2','--out',dir+'/eval-'+name+'.json'],'eval-'+name);
} else await Promise.all(['a','b'].map(evaluateIndependent));
const read = name => JSON.parse(readFileSync(`${dir}/${name}`,'utf8'));
const proof=compare(read('eval-a.json'),read('eval-b.json'));
if(modelId==='RL-Linear'&&opponentId==='V2E'&&matches===9&&seed===975020 &&
  (proof.modelSha256!=='47fe72475fff8a5a56de730d3b67d02df6c15a0ec710b9b1145541a874aa4fab'||
   proof.normalizedOutcomeSha256!=='a3fec0779dbd833cb68919f0bb5243057610607e5ceb729f114a1cfd15575466'))
  throw new Error('I975 pre-migration gate mismatch');
proof.modelFileSha256=createHash('sha256').update(readFileSync(model)).digest('hex');
proof.matches=matches; proof.trainingSeed=seed; proof.node=process.version;
proof.modelId=modelId; proof.opponentId=opponentId;
writeFileSync(`${dir}/comparison.json`,JSON.stringify(proof,null,2)+'\n');
console.log(JSON.stringify(proof,null,2));
