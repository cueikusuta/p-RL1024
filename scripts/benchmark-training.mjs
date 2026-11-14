// Compare identical 24-Match online Actor-Critic training runs.
// Training updates remain strictly sequential; no trajectories are omitted.
import {performance} from 'node:perf_hooks';
import {createHash} from 'node:crypto';
import {SeededRandom} from '../src/battle.js';
import {mixSeed} from './dev/simulator/runner.js';
import {runStrategyMatch} from './dev/simulator/strategy-match.js';
import {createModel as createOriginalModel,createSession as createOriginalSession}
  from '../models/deep-reference.mjs';
import {createModel as createOptimizedModel,createSession as createOptimizedSession}
  from '../models/deep.mjs';

const hash=x=>createHash('sha256').update(x).digest('hex');
const count=24,root=975020;
function training(createModel,createSession,matches=count) {
  const model=createModel(root),past=[],records=[];
  for(let i=0;i<matches;i++) {
    if(i%8===0)past.push(structuredClone(model));
    const seed=mixSeed(root,i,0x976),studentSeat=i%2;
    const opponent=past[Math.floor(i/8)];
    const sessions=[null,null];
    sessions[studentSeat]=createSession(model,{
      train:true,rng:new SeededRandom(mixSeed(seed,0xa1,studentSeat))});
    sessions[1-studentSeat]=createSession(opponent,{train:false});
    const result=runStrategyMatch({
      seed,policies:['pure-rl-linear','pure-rl-linear'],
      strategyMode:'p2',pureRlSessions:sessions,pureRlAllPhases:false});
    if(!result.completed||result.score[result.winner]!==4)
      throw new Error('Training Match incomplete');
    records.push({seed,studentSeat,winner:result.winner,score:result.score,
      rounds:result.rounds,roundWinners:result.roundWinners,
      actions:result.actionCounts,stats:result.pureRlStats});
  }
  return {modelSha256:hash(JSON.stringify(model)),
    learnedKeys:Object.keys(model.weights).length,
    outcomeSha256:hash(JSON.stringify(records))};
}
// Use identical 9-Match warm-up for both paths, outside timed samples.
training(createOriginalModel,createOriginalSession,9);
training(createOptimizedModel,createOptimizedSession,9);
const t1=performance.now();
const original=training(createOriginalModel,createOriginalSession);
const originalTotalMs=performance.now()-t1;
const t2=performance.now();
const optimized=training(createOptimizedModel,createOptimizedSession);
const optimizedTotalMs=performance.now()-t2;
if(JSON.stringify(original)!==JSON.stringify(optimized))
  throw new Error('Optimized online learning changed model weights or Match history');
console.log('TRAINING_BENCHMARK '+JSON.stringify({
  schema:'p-rl1024-trained-actor-critic-v1',
  node:process.version,runner:process.env.RUNNER_NAME??null,matches:count,
  learnedKeys:original.learnedKeys,modelSha256:original.modelSha256,
  outcomeSha256:original.outcomeSha256,
  originalTotalMs,optimizedTotalMs,speedup:originalTotalMs/optimizedTotalMs,
}));
