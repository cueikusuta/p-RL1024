#!/usr/bin/env node
// I975: research-only, zero-teacher non-neural SARSA training.
// By default only battle actions are learned and both seats use fixed Basic
// draft/selection; --all-phases true enables learned draft/selection too.
// The training win count is NOT an independent holdout strength result.
import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir,readdir} from 'node:fs/promises';
import {dirname,resolve,join,relative} from 'node:path';
import {SeededRandom} from '../src/battle.js';
import {mixSeed} from './dev/simulator/runner.js';
import {runStrategyMatch} from './dev/simulator/strategy-match.js';
import {createPureRlModel,createPureRlSession,validatePureRlModel,
  PURE_RL_SCHEMA} from '../src/ai/pure-rl-linear.js';
import {rankActions as rankPureRlActions} from '../models/linear.mjs';
import {rankPureRlPhaseActions} from '../src/ai/pure-rl-phases.js';
import {readPureRlResumeCheckpoint,writePureRlResumeCheckpoint}
  from './dev/simulator/pure-rl-resume.mjs';

const args=new Map();
if((process.argv.length-2)%2)throw new Error('Expected --key value arguments');
for(let i=2;i<process.argv.length;i+=2){
  if(!['--matches','--seed','--out','--epsilon','--epsilon-final',
      '--checkpoint-every','--all-phases','--resume'].includes(process.argv[i])||
    args.has(process.argv[i]))throw new Error('Unknown/duplicate option');
  args.set(process.argv[i],process.argv[i+1]);
}
const matches=Number(args.get('--matches')??16);
const root=Number(args.get('--seed')??975001);
const epsilon=Number(args.get('--epsilon')??0.35);
const finalEpsilon=Number(args.get('--epsilon-final')??epsilon);
const checkpointEvery=Number(args.get('--checkpoint-every')??0);
const allPhases=args.get('--all-phases')==='true';
if(args.has('--all-phases')&& !['true','false'].includes(args.get('--all-phases')))
  throw new Error('Expected --all-phases true or false');
const rankerForSeat=seat=>(model,view)=>['draft','selection'].includes(view.phase)
  ?rankPureRlPhaseActions(model,view,{seat}):rankPureRlActions(model,view);
if(!Number.isSafeInteger(matches)||matches<1||matches>200000||
  !Number.isSafeInteger(root)||root<1||root>0xffffffff||
  !Number.isFinite(epsilon)||epsilon<0||epsilon>1||
  !Number.isFinite(finalEpsilon)||finalEpsilon<0||finalEpsilon>1||
  !Number.isSafeInteger(checkpointEvery)||checkpointEvery<0||
  checkpointEvery>matches||(checkpointEvery>0&&!args.get('--out'))||
  (args.has('--resume')&&(!args.get('--resume')||!args.get('--out'))))
  throw new Error('Invalid bounded research training configuration');
const model=createPureRlModel(),past=[];
let totalWins=0,decisionCount=0,updates=0;
const summaries=[],sourceHash=createHash('sha256');
const sourcePaths=[];
async function collectSources(dir){
  for(const entry of await readdir(dir,{withFileTypes:true})){
    const full=join(dir,entry.name);
    if(entry.isDirectory())await collectSources(full);
    else if(entry.isFile() && /\.(?:js|mjs|cjs|json)$/.test(entry.name))
      sourcePaths.push(full);
  }
}
// Fingerprint every Battle, Match, draft and simulator source plus catalog.
for(const dir of ['src','scripts/dev/simulator','data/catalog'])
  await collectSources(resolve(dir));
sourcePaths.push(resolve('package.json'));
for(const full of sourcePaths.sort()){
  sourceHash.update(relative(process.cwd(),full).replaceAll('\\','/')+'\0');
  sourceHash.update(await readFile(full));
}
const sourceSha256=sourceHash.digest('hex');
const sha256=value=>createHash('sha256').update(value).digest('hex');
const reportFor=()=>({
  schema:'i975-pure-rl-battle-train-result-v1',
  algorithm:'i975-true-online-sarsa-lambda-v1',
  phase:allPhases?'full pure RL policy (draft, selection, battle)'
    :'battle-only; draft/selection fixed Basic for both seats',
  trainingNotHoldout:true,seedRoot:root,completedMatches:summaries.length,
  studentWins:totalWins,trainedDecisions:decisionCount,updates,
  epsilon,epsilonFinal:finalEpsilon,checkpointEvery,
  hyperparameters:{alpha:0.025,gamma:1,lambda:0.8},
  allPhases,sourceSha256,
  modelSha256:sha256(JSON.stringify(model)),
  model:{schema:PURE_RL_SCHEMA,weights:model.weights},
  matches:[...summaries],
});
const checkpointName=(dest,n)=>dest.endsWith('.json')
  ?dest.slice(0,-5)+'-m'+n+'.json':dest+'-m'+n+'.json';
let startAt=0;
if(args.has('--resume')){
  const state=await readPureRlResumeCheckpoint(resolve(args.get('--resume')),{
    algorithm:'i975-true-online-sarsa-lambda-v1',plannedMatches:matches,
    seedRoot:root,sourceSha256,allPhases,epsilon,
    epsilonFinal:finalEpsilon,checkpointEvery
  });
  startAt=state.completedMatches;
  model.weights=state.model.weights;
  past.push(...state.past);
  summaries.push(...state.matches);
  totalWins=state.studentWins;
  decisionCount=state.trainedDecisions;
  updates=state.updates;
  console.log(JSON.stringify({resumedAtMatch:startAt,sourceSha256}));
}
for(let i=startAt;i<matches;i++) {
  // Frozen prior checkpoint opponent; no rule-based score/teacher whatsoever.
  if(i%8===0)past.push(structuredClone(model));
  const seed=mixSeed(root,i,0x975);
  const studentSeat=i%2,otherSeat=1-studentSeat;
  const block=Math.floor(i/8);
  // First two blocks are pinned for Smoke contract compatibility. After that
  // sample across all frozen older snapshots: not just the latest predecessor.
  const opponentCheckpointIndex=block<2?block:
    mixSeed(root,i,0x5e17)%past.length;
  const opponent=past[opponentCheckpointIndex];
  const sessions=[null,null];
  const policyEpsilon=epsilon+(finalEpsilon-epsilon)*
    (matches<=1?0:i/(matches-1));
  sessions[studentSeat]=createPureRlSession(model,{train:true,
    epsilon:policyEpsilon,alpha:0.025,gamma:1,lambda:0.8,
    ranker:allPhases?rankerForSeat(studentSeat):rankPureRlActions,
    rng:new SeededRandom(mixSeed(seed,0xa1,studentSeat))});
  sessions[otherSeat]=createPureRlSession(opponent,{train:false,
    epsilon:0.10,
    ranker:allPhases?rankerForSeat(otherSeat):rankPureRlActions,
    rng:new SeededRandom(mixSeed(seed,0xa1,otherSeat))});
  const result=runStrategyMatch({seed,
    policies:['pure-rl-linear','pure-rl-linear'],
    strategyMode:'p2',pureRlSessions:sessions,
    pureRlAllPhases:allPhases});
  if(!result.completed||result.score[result.winner]!==4)
    throw new Error('Training must complete the authoritative four-win Match');
  totalWins+=Number(result.winner===studentSeat);
  decisionCount+=result.pureRlStats[studentSeat].decisions;
  updates+=result.pureRlStats[studentSeat].updates;
  summaries.push({seed,studentSeat,opponentCheckpointIndex,
    winner:result.winner,score:[...result.score],policyEpsilon,
    won:result.winner===studentSeat,
    rounds:result.rounds,decisions:result.pureRlStats[studentSeat].decisions,
    updates:result.pureRlStats[studentSeat].updates});
  if(checkpointEvery>0&&(i+1)%checkpointEvery===0){
    validatePureRlModel(model);
    const dest=resolve(args.get('--out'));
    await mkdir(dirname(dest),{recursive:true});
    await writePureRlResumeCheckpoint(checkpointName(dest,i+1),
      reportFor(),{plannedMatches:matches,past});
  }
  console.log(JSON.stringify({completedMatches:i+1,studentWins:totalWins,
    trainedDecisions:decisionCount}));
}
validatePureRlModel(model);
const result=reportFor();
const modelSha256=result.modelSha256;
if(args.has('--out')){
  const dest=resolve(args.get('--out'));
  await mkdir(dirname(dest),{recursive:true});
  await writeFile(dest,JSON.stringify(result,null,2)+'\n');
}
console.log(JSON.stringify({schema:result.schema,completedMatches:matches,
  studentWins:totalWins,trainedDecisions:decisionCount,updates,
  modelSha256,sourceSha256,out:args.get('--out')??null}));
