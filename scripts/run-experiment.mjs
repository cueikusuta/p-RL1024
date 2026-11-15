import {spawnSync,spawn} from 'node:child_process';
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
const model=`${dir}/training.json`;
const trainArgs=['--matches',String(matches),'--seed',String(seed),'--checkpoint-every','8','--out',model];
if(opponent.id==='AQ-Base-001')trainArgs.unshift('--model',selected.id);
else if(selected.id==='RL-Linear')trainArgs.unshift('--all-phases','true');
run(opponent.id==='AQ-Base-001'?'scripts/train-vs-aq.mjs':selected.trainingEntry,trainArgs);
(selected.id==='RL-Linear'?loadModel:loadDeepModel)(JSON.stringify(JSON.parse(readFileSync(model,'utf8')).model));
const evaluator=opponent.id==='AQ-Base-001'?'scripts/evaluate-vs-aq.mjs':selected.id==='RL-Linear'?'scripts/evaluate-linear.mjs':'scripts/evaluate-deep.mjs';
function evaluateIndependent(name) {
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
await Promise.all(['a','b'].map(evaluateIndependent));
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
