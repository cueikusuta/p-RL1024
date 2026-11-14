// Reproducible full-four-win engine benchmark. Never shortens the Match.
// The policy and its opponent use the same fixed frozen model and public views.
import {performance} from 'node:perf_hooks';
import {createHash} from 'node:crypto';
import {runStrategyMatch} from './dev/simulator/strategy-match.js';
import {createModel,createSession} from '../models/deep.mjs';
import {mixSeed} from './dev/simulator/runner.js';

const model=createModel(975020);
function play(index,profile=false) {
  const seed=mixSeed(975020,index,0x2b10);
  const begin=performance.now();
  const result=runStrategyMatch({
    seed,policies:['pure-rl-linear','pure-rl-linear'],
    strategyMode:'p2',pureRlSessions:[createSession(model),createSession(model)],
    pureRlAllPhases:false,capturePhaseTimings:profile,
  });
  if(!result.completed||result.score[result.winner]!==4)
    throw new Error('Benchmark must complete an authentic four-win Match');
  return {timeMs:performance.now()-begin,seed,score:result.score,
    rounds:result.rounds,roundWinners:result.roundWinners,
    actions:result.actionCounts,phaseProfile:result.phaseProfile};
}
// Warm-up is not included in timing, but executes a complete Match.
play(-1);
const records=[];
for(let i=0;i<8;i++)records.push(play(i,true));
const times=records.map(r=>r.timeMs).sort((a,b)=>a-b);
const phaseMs={};
for(const r of records)for(const [key,entry] of Object.entries(r.phaseProfile.phases))
  phaseMs[key]=(phaseMs[key]??0)+entry.wallMs;
const durationMs=records.reduce((sum,r)=>sum+r.timeMs,0);
const digest=createHash('sha256').update(JSON.stringify(records.map(
  ({seed,score,rounds,roundWinners,actions})=>({seed,score,rounds,roundWinners,actions})
))).digest('hex');
console.log('ENGINE_BENCHMARK '+JSON.stringify({
  schema:'p-rl1024-engine-benchmark-v1',node:process.version,
  cpu:process.env.RUNNER_NAME??null,fullMatches:8,
  outcomeDigestSha256:digest,medianMatchMs:(times[3]+times[4])/2,
  totalMs:durationMs,matchTimesMs:records.map(r=>r.timeMs),
  phaseMs:Object.fromEntries(Object.entries(phaseMs).sort((a,b)=>b[1]-a[1])),
}));
