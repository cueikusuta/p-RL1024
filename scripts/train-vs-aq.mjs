import {createHash,randomUUID} from 'node:crypto';
import {readFile,writeFile,mkdir,rename,rm,readdir} from 'node:fs/promises';
import {dirname,resolve,join} from 'node:path';
import {createModel,validateModel,match,mixSeed,weightsSha256} from '../models/aq-runner.mjs';

const opts=new Map(),argv=process.argv.slice(2),allowed=['--model','--matches','--seed','--out','--resume','--checkpoint-every'];
if(argv.length%2)throw new Error('Expected option/value pairs');
for(let i=0;i<argv.length;i+=2){if(!allowed.includes(argv[i])||opts.has(argv[i]))throw new Error('Invalid option');opts.set(argv[i],argv[i+1]);}
const id=opts.get('--model'),matches=Number(opts.get('--matches')),root=Number(opts.get('--seed')),interval=Number(opts.get('--checkpoint-every')??8),out=opts.get('--out');
if(!['RL-Linear','RL-Deep'].includes(id)||!Number.isSafeInteger(matches)||matches<9||matches>256||!Number.isSafeInteger(root)||root<1||root>0xffffffff||interval!==8||!out)throw new Error('Invalid AQ training configuration');
const sha=x=>createHash('sha256').update(x).digest('hex'),h=createHash('sha256');
const files=['provenance/manifest.json','package.json','scripts/train-vs-aq.mjs','scripts/evaluate-vs-aq.mjs'];
for(const x of await readdir('models'))if(x.endsWith('.mjs'))files.push('models/'+x);
for(const path of files.sort()){h.update(path+'\0');h.update(await readFile(path));}
const sourceSha256=h.digest('hex');
let model=createModel(id,root),history=[],decisions=0,updates=0;
if(opts.has('--resume')) {
  const checkpoint=JSON.parse(await readFile(opts.get('--resume'),'utf8')),{checkpointSha256,...state}=checkpoint;
  if(sha(JSON.stringify(state))!==checkpointSha256||state.schema!=='p-rl1024-aq-checkpoint-v1'||state.modelId!==id||state.matches!==matches||state.seed!==root||state.sourceSha256!==sourceSha256||state.opponentSha256!==weightsSha256||!Array.isArray(state.history)||state.history.length<1||state.history.length>=matches||state.history.length%8!==0)throw new Error('Checkpoint integrity or configuration mismatch');
  for(let i=0;i<state.history.length;i++)if(state.history[i].seed!==mixSeed(root,i,0xa975)||state.history[i].studentSeat!==i%2)throw new Error('Checkpoint history mismatch');
  model=validateModel(id,state.model);history=state.history;decisions=state.decisions;updates=state.updates;
}
async function checkpoint() {
  const state={schema:'p-rl1024-aq-checkpoint-v1',modelId:id,matches,seed:root,sourceSha256,opponentSha256:weightsSha256,model,history,decisions,updates};
  const path=resolve(out.replace(/\.json$/,'')+`-m${history.length}.json`),tmp=path+'.tmp-'+randomUUID();
  await mkdir(dirname(path),{recursive:true});
  const body=JSON.stringify(state);
  const data=body.slice(0,-1)+',"checkpointSha256":"'+sha(body)+'"}\n';
  try{await writeFile(tmp,data,{flag:'wx'});await rename(tmp,path);}catch(error){await rm(tmp,{force:true});throw error;}
}
for(let i=history.length;i<matches;i++) {
  const seed=mixSeed(root,i,0xa975),seat=i%2,epsilon=.35+(.08-.35)*i/(matches-1);
  const result=match({id,model,seed,seat,train:true,epsilon}),stats=result.pureRlStats[seat];
  if(!stats?.train||stats.updates<1)throw new Error('Missing student learning updates');
  decisions+=stats.decisions;updates+=stats.updates;
  history.push({seed,studentSeat:seat,winner:result.winner,score:result.score,rounds:result.rounds,decisions:stats.decisions,updates:stats.updates});
  if((i+1)%8===0)await checkpoint();
  console.log(JSON.stringify({model:id,completedMatches:i+1,decisions,updates}));
}
const report={schema:'p-rl1024-aq-training-v1',modelId:id,algorithm:id==='RL-Linear'?'i975-true-online-sarsa-lambda-v1':'terminal-advantage-actor-critic',featureVersion:'pure-rl-sparse-public',phase:id==='RL-Linear'?'full-phase':'battle-only; Basic draft and selection',teacher:'none',trainingOpponent:'AQ-Base-001',opponentModelSha256:weightsSha256,sourceSha256,sourceCommit:process.env.GITHUB_SHA??null,seedRoot:root,completedMatches:matches,trainedDecisions:decisions,updates,hyperparameters:id==='RL-Linear'?{alpha:.025,gamma:1,lambda:.8,epsilonStart:.35,epsilonEnd:.08,checkpointEvery:8}:{alpha:.0005,hidden:16,activation:'tanh',policy:'masked-softmax',reward:'terminal Match +/-1',checkpointEvery:8},modelSha256:sha(JSON.stringify(model)),model,history};
await mkdir(dirname(resolve(out)),{recursive:true});await writeFile(out,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({model:id,completedMatches:matches,modelSha256:report.modelSha256,opponentModelSha256:weightsSha256,sourceSha256}));
