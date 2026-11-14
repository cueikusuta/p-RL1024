// Compare serial and parallel throughput for the SAME full-Match seeds.
// Neither branch skips a round, a Battle, or a model decision.
import {performance} from 'node:perf_hooks';
import {createHash} from 'node:crypto';
import {availableParallelism} from 'node:os';
import {createModel} from '../models/deep.mjs';
import {createModel as createOriginalModel,createSession as createOriginalSession} from '../models/deep-reference.mjs';
import {runStrategyMatch} from './dev/simulator/strategy-match.js';
import {mixSeed} from './dev/simulator/runner.js';
import {frozenDeepMatch,frozenDeepBatch} from './batch-frozen-deep.mjs';

const model=createModel(975020);
const originalModel=createOriginalModel(975020);
const originalMatch=seed=>runStrategyMatch({seed,policies:['pure-rl-linear','pure-rl-linear'],strategyMode:'p2',pureRlSessions:[createOriginalSession(originalModel),createOriginalSession(originalModel)],pureRlAllPhases:false});
const seeds=Array.from({length:128},(_,i)=>mixSeed(975020,i,0x2b10));
const digest=rows=>createHash('sha256').update(JSON.stringify(rows.map(
  ({seed,score,rounds,roundWinners,actionCounts})=>({
    seed,score,rounds,roundWinners,actions:actionCounts,
  })))).digest('hex');
const totals=rows=>rows.reduce((sum,r)=>sum+(r.phaseProfile?.totalWallMs??0),0);
// Same warm-up in the main isolate; workers warm up by their initial jobs.
frozenDeepMatch(model,mixSeed(975020,-1,0x2b10));
originalMatch(mixSeed(975020,-1,0x2b10));
const tOriginal=performance.now();
const original=seeds.map(originalMatch);
const originalTotalMs=performance.now()-tOriginal;

const t0=performance.now();
const serial=seeds.map(seed=>frozenDeepMatch(model,seed,{profile:false}));
const serialTotalMs=performance.now()-t0;
const workers=Math.min(4,availableParallelism());
const t1=performance.now();
const parallel=await frozenDeepBatch(model,seeds,{workers,profile:false});
const parallelTotalMs=performance.now()-t1;
const originalSha256=digest(original.map((result,i)=>({seed:seeds[i],...result})));
const serialSha256=digest(serial.map((result,i)=>({seed:seeds[i],...result})));
const parallelSha256=digest(parallel.map((result,i)=>({seed:seeds[i],...result})));
if(originalSha256!==serialSha256||serialSha256!==parallelSha256)throw new Error('Frozen parallel results differ from original serial baseline');
const originalEightDigest=digest(serial.slice(0,8).map((result,i)=>({seed:seeds[i],...result})));
if(originalEightDigest!=='b7323a8441e0a518257452f2c14df61952ef7b5bca7b91838a92e0dc82a896c4')
  throw new Error('Frozen baseline outcome digest changed');
const phaseMs={};
for(const r of serial)for(const [key,entry] of Object.entries(r.phaseProfile?.phases??{}))
  phaseMs[key]=(phaseMs[key]??0)+entry.wallMs;
console.log('ENGINE_BENCHMARK '+JSON.stringify({
  schema:'p-rl1024-engine-throughput-v2',node:process.version,
  runner:process.env.RUNNER_NAME??null,availableParallelism:availableParallelism(),
  fullMatches:128,workers,outcomeDigestSha256:serialSha256,
  originalTotalMs,serialTotalMs,parallelTotalMs,
  speedup:serialTotalMs/parallelTotalMs,
  baselineToOptimizedSpeedup:originalTotalMs/parallelTotalMs,
  serialMatchWallMs:totals(serial),parallelMatchWallMs:totals(parallel),
  phaseMs:Object.fromEntries(Object.entries(phaseMs).sort((a,b)=>b[1]-a[1])),
}));
