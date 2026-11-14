// True same-machine comparison: original frozen inference, optimized serial,
// and optimized worker pool on identical, full-four-win Match seeds.
import {performance} from 'node:perf_hooks';
import {createHash} from 'node:crypto';
import {availableParallelism,tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {join} from 'node:path';
import {createSession as createOriginalSession} from '../models/deep-reference.mjs';
import {runStrategyMatch} from './dev/simulator/strategy-match.js';
import {mixSeed} from './dev/simulator/runner.js';
import {frozenDeepMatch,frozenDeepBatch} from './batch-frozen-deep.mjs';

// Using a genuinely learned model is crucial: the original inference path
// revalidated *every learned weight vector* on every action candidate.
const tmp=mkdtempSync(join(tmpdir(),'p-rl1024-perf-'));
let training;
try{
  const path=join(tmp,'training.json');
  const p=spawnSync(process.execPath,['scripts/train-deep.mjs',
    '--matches','9','--seed','975020','--checkpoint-every','8',
    '--out',path],{encoding:'utf8',timeout:180000});
  if(p.error||p.status!==0)throw p.error??new Error('Representative training failed: '+p.stderr);
  training=JSON.parse(readFileSync(path,'utf8'));
}finally{rmSync(tmp,{force:true,recursive:true});}
const model=training.model;
const weightsCount=Object.keys(model.weights).length;
if(training.completedMatches!==9||!weightsCount)throw new Error('Benchmark needs a nonempty learned model');
const originalMatch=seed=>runStrategyMatch({
  seed,policies:['pure-rl-linear','pure-rl-linear'],strategyMode:'p2',
  pureRlSessions:[createOriginalSession(model),createOriginalSession(model)],
  pureRlAllPhases:false,
});
const seeds=Array.from({length:64},(_,i)=>mixSeed(975020,i,0x2b10));
const digest=rows=>createHash('sha256').update(JSON.stringify(rows.map(
  ({seed,score,rounds,roundWinners,actionCounts})=>({
    seed,score,rounds,roundWinners,actions:actionCounts,
  })))).digest('hex');
// Warm both serial paths with the SAME authentic Match.
originalMatch(mixSeed(975020,-1,0x2b10));
frozenDeepMatch(model,mixSeed(975020,-1,0x2b10));
const workers=Math.min(4,availableParallelism()),repeats=3,trials=[];
const sha=results=>digest(results.map((result,i)=>({seed:seeds[i],...result})));
let expectedOutcomeSha256=null;
for(let repeat=0;repeat<repeats;repeat++){
  const tOriginal=performance.now();
  const original=seeds.map(originalMatch);
  const originalTotalMs=performance.now()-tOriginal;
  const tSerial=performance.now();
  const serial=seeds.map(seed=>frozenDeepMatch(model,seed));
  const serialTotalMs=performance.now()-tSerial;
  const tParallel=performance.now();
  const parallel=await frozenDeepBatch(model,seeds,{workers});
  const parallelTotalMs=performance.now()-tParallel;
  const originalSha256=sha(original),serialSha256=sha(serial),parallelSha256=sha(parallel);
  if(originalSha256!==serialSha256||serialSha256!==parallelSha256)
    throw new Error('Frozen parallel results differ from original serial baseline');
  if(expectedOutcomeSha256&&originalSha256!==expectedOutcomeSha256)
    throw new Error('Frozen evaluation outcomes changed between repetitions');
  expectedOutcomeSha256=originalSha256;
  trials.push({
    originalTotalMs,serialTotalMs,parallelTotalMs,
    serialImprovement:originalTotalMs/serialTotalMs,
    workerSpeedup:serialTotalMs/parallelTotalMs,
    baselineToOptimizedSpeedup:originalTotalMs/parallelTotalMs,
  });
}
const median=arr=>[...arr].sort((a,b)=>a-b)[Math.floor(arr.length/2)];
const medianSpeedup=median(trials.map(t=>t.baselineToOptimizedSpeedup));
console.log('ENGINE_BENCHMARK '+JSON.stringify({
  schema:'p-rl1024-trained-engine-throughput-v4',
  node:process.version,runner:process.env.RUNNER_NAME??null,
  availableParallelism:availableParallelism(),fullMatches:64,repeats,workers,
  trainingMatches:9,trainedWeights:weightsCount,trainedModelSha256:training.modelSha256,
  outcomeDigestSha256:expectedOutcomeSha256,
  trials,medianSpeedup,accepted:medianSpeedup>=2,
}));
if(medianSpeedup<2)throw new Error(
  'Frozen evaluation 3-repeat median speedup below 2.00: '+medianSpeedup);
