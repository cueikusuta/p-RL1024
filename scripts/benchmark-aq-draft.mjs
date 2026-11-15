import {createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
import {writeFileSync,mkdirSync} from 'node:fs';
import {SeededRandom} from '../src/battle.js';
import {createModel as createDeepModel,createSession} from '../models/deep.mjs';
import {createOpponent} from '../models/aq-base.mjs';
import {mixSeed} from '../models/aq-runner.mjs';
import {runStrategyMatch as original} from '../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {runStrategyMatch as optimized} from './optimized/strategy-match.mjs';

const hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
function batch(run,root,count){
  const model=createDeepModel(root),records=[];
  const begin=performance.now();
  for(let i=0;i<count;i++){
    const seed=mixSeed(root,i,0xa975),seat=i%2;
    const session=createSession(model,{train:true,rng:new SeededRandom(mixSeed(seed,0xa1,seat))});
    const policies=['aq-base','aq-base'],sessions=[null,null];
    policies[seat]='pure-rl-linear';sessions[seat]=session;
    const out=run({seed,policies,pureRlSessions:sessions,pureRlAllPhases:false,
      strategyMode:'integrated',v2Weights:{blend:.28,exposure:.75,continuation:.4},
      aqBase:createOpponent()});
    if(!out.completed||out.score[out.winner]!==4)throw new Error('Incomplete Match');
    records.push({seed,studentSeat:seat,winner:out.winner,rounds:out.rounds,
      score:out.score,actions:out.actionCounts,stats:out.pureRlStats});
  }
  return {durationMs:performance.now()-begin,modelSha256:hash(model),
    outcomeDigest:hash(records),count};
}
const count=16,root=975020;
const coldOriginal=batch(original,root,9),coldFast=batch(optimized,root,9);
if(coldOriginal.modelSha256!==coldFast.modelSha256||coldOriginal.outcomeDigest!==coldFast.outcomeDigest)
  throw new Error('Warm-up fork does not reproduce original learned model or full Match history');
const trials=[];
for(let i=0;i<3;i++){
  const a=batch(original,root,count),b=batch(optimized,root,count);
  if(a.modelSha256!==b.modelSha256||a.outcomeDigest!==b.outcomeDigest)
    throw new Error('Optimized draft changed the model SHA or Match outcome digest');
  trials.push({originalMs:a.durationMs,optimizedMs:b.durationMs,speedup:a.durationMs/b.durationMs});
}
const ratios=trials.map(x=>x.speedup).sort((a,b)=>a-b);
const report={schema:'i1-optimized-draft-profile-v1',count,root,node:process.version,
  arch:process.arch,verifiedModelSha256:coldOriginal.modelSha256,
  outcomeDigest:coldOriginal.outcomeDigest,trials,medianSpeedup:ratios[1]};
mkdirSync('artifacts',{recursive:true});
writeFileSync('artifacts/aq-draft-optimization-benchmark.json',JSON.stringify(report,null,2)+'\n');
console.log('AQ_DRAFT_OPTIMIZATION '+JSON.stringify(report));
