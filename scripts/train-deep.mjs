import {createHash,randomUUID} from 'node:crypto';
import {readFile,writeFile,mkdir,rename,rm,readdir} from 'node:fs/promises';
import {dirname,resolve,relative,join} from 'node:path';
import {SeededRandom} from '../src/battle.js';
import {mixSeed} from './dev/simulator/runner.js';
import {runStrategyMatch} from './dev/simulator/strategy-match.js';
import {createModel,createSession,validateModel} from '../models/deep.mjs';

const opts=new Map(),argv=process.argv.slice(2);
if(argv.length%2)throw new Error('Expected option/value pairs');
for(let i=0;i<argv.length;i+=2){if(!['--matches','--seed','--out','--resume','--checkpoint-every'].includes(argv[i])||opts.has(argv[i]))throw new Error('Invalid option');opts.set(argv[i],argv[i+1]);}
const matches=Number(opts.get('--matches')),root=Number(opts.get('--seed')),interval=Number(opts.get('--checkpoint-every')??8),out=opts.get('--out');
if(!Number.isSafeInteger(matches)||matches<9||matches>256||!Number.isSafeInteger(root)||root<1||root>0xffffffff||!Number.isSafeInteger(interval)||interval!==8||!out)throw new Error('Invalid training configuration');
const hash=x=>createHash('sha256').update(x).digest('hex');
const sourceFiles=[];
async function collect(dir){for(const ent of await readdir(dir,{withFileTypes:true})){const p=join(dir,ent.name);if(ent.isDirectory())await collect(p);else if(ent.isFile()&&/\.(?:js|mjs|cjs|json)$/.test(ent.name))sourceFiles.push(p);}}
for(const dir of ['src','scripts/dev/simulator','data/catalog','models'])await collect(resolve(dir));
for(const p of ['package.json','scripts/train-deep.mjs','scripts/evaluate-deep.mjs'])sourceFiles.push(resolve(p));
const sourceHash=createHash('sha256');for(const p of sourceFiles.sort()){sourceHash.update(relative(process.cwd(),p).replaceAll('\\','/')+'\0');sourceHash.update(await readFile(p));}
const sourceSha256=sourceHash.digest('hex');
let model=createModel(root),past=[],history=[],decisions=0,updates=0;
if(opts.has('--resume')) {
  const checkpoint=JSON.parse(await readFile(opts.get('--resume'),'utf8'));
  const {checkpointSha256,...state}=checkpoint;
  if(hash(JSON.stringify(state))!==checkpointSha256||state.schema!=='p-rl1024-deep-checkpoint-v1'||state.matches!==matches||state.seed!==root||state.sourceSha256!==sourceSha256||!Array.isArray(state.history)||!Array.isArray(state.past)||state.history.length<1||state.history.length>=matches||state.history.length%8!==0||state.past.length!==Math.ceil(state.history.length/8))throw new Error('Checkpoint integrity or configuration mismatch');
  for(let i=0;i<state.history.length;i++)if(state.history[i].seed!==mixSeed(root,i,0x976)||state.history[i].studentSeat!==i%2)throw new Error('Checkpoint history mismatch');
  model=validateModel(state.model);past=state.past.map(validateModel);history=state.history;decisions=state.decisions;updates=state.updates;
}
async function saveCheckpoint() {
  const state={schema:'p-rl1024-deep-checkpoint-v1',matches,seed:root,sourceSha256,model,past,history,decisions,updates};
  const path=resolve(out.replace(/\.json$/,'')+`-m${history.length}.json`),tmp=path+'.tmp-'+randomUUID();
  await mkdir(dirname(path),{recursive:true});
  try{await writeFile(tmp,JSON.stringify({...state,checkpointSha256:hash(JSON.stringify(state))})+'\n',{flag:'wx'});await rename(tmp,path);}catch(error){await rm(tmp,{force:true});throw error;}
}
for(let i=history.length;i<matches;i++) {
  if(i%8===0)past.push(structuredClone(model));
  const seed=mixSeed(root,i,0x976),studentSeat=i%2,opponentIndex=Math.floor(i/8),opponent=past[opponentIndex];
  const sessions=[null,null];
  sessions[studentSeat]=createSession(model,{train:true,rng:new SeededRandom(mixSeed(seed,0xa1,studentSeat))});
  sessions[1-studentSeat]=createSession(opponent,{train:false});
  const result=runStrategyMatch({seed,policies:['pure-rl-linear','pure-rl-linear'],strategyMode:'p2',pureRlSessions:sessions,pureRlAllPhases:false});
  if(!result.completed||result.score[result.winner]!==4)throw new Error('Training Match incomplete');
  const stats=result.pureRlStats[studentSeat];decisions+=stats.decisions;updates+=stats.updates;
  history.push({seed,studentSeat,opponentIndex,winner:result.winner,score:result.score,rounds:result.rounds,decisions:stats.decisions,updates:stats.updates});
  if((i+1)%8===0)await saveCheckpoint();
  console.log(JSON.stringify({completedMatches:i+1,decisions,updates}));
}
const report={schema:'p-rl1024-deep-training-v1',algorithm:'terminal-advantage-actor-critic',featureVersion:'pure-rl-sparse-public',phase:'battle-only; Basic draft and selection',teacher:'none',trainingOpponent:'frozen-self-play',sourceSha256,sourceCommit:process.env.GITHUB_SHA??null,seedRoot:root,completedMatches:matches,trainedDecisions:decisions,updates,architecture:{hidden:16,activation:'tanh',policy:'masked-softmax',value:'tanh'},hyperparameters:{alpha:0.0005,reward:'terminal Match +/-1',checkpointEvery:8},modelSha256:hash(JSON.stringify(model)),model,history};
await mkdir(dirname(resolve(out)),{recursive:true});await writeFile(out,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({schema:report.schema,completedMatches:matches,modelSha256:report.modelSha256,sourceSha256,updates}));
