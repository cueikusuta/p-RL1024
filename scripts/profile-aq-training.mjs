import {createHash} from 'node:crypto';
import {mkdirSync,writeFileSync} from 'node:fs';
import {performance} from 'node:perf_hooks';
import {Session} from 'node:inspector';
import {SeededRandom} from '../src/battle.js';
import {runStrategyMatch} from './optimized/strategy-match.mjs';
import {createModel,match,mixSeed} from '../models/aq-runner.mjs';
import {createSession} from '../models/deep.mjs';
import {createOpponent} from '../models/aq-base.mjs';

const root=975020,count=16,id='RL-Deep',student=createModel(id,root),
  control=createModel(id,root),phaseTotals=new Map(),decisions=[];
const hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const profiler=new Session();
profiler.connect();
const post=(name,args={})=>new Promise((resolve,reject)=>
  profiler.post(name,args,(error,value)=>error?reject(error):resolve(value)));
await post('Profiler.enable');
await post('Profiler.setSamplingInterval',{interval:1000});
await post('Profiler.start');
const begin=performance.now();
for(let i=0;i<count;i++){
  const seed=mixSeed(root,i,0xa975),seat=i%2;
  const direct=match({id,model:control,seed,seat,train:true});
  const session=createSession(student,{train:true,rng:new SeededRandom(mixSeed(seed,0xa1,seat))});
  const policies=['aq-base','aq-base'],sessions=[null,null];
  policies[seat]='pure-rl-linear';sessions[seat]=session;
  const prof=runStrategyMatch({seed,policies,pureRlSessions:sessions,
    pureRlAllPhases:false,strategyMode:'integrated',
    v2Weights:{blend:.28,exposure:.75,continuation:.4},
    aqBase:createOpponent(),capturePhaseTimings:true});
  if(prof.winner!==direct.winner||JSON.stringify(prof.score)!==JSON.stringify(direct.score)||
      hash(control)!==hash(student))throw new Error('CPU profiling changed training SHA or Match result: '+i);
  for(const [key,value] of Object.entries(prof.phaseProfile.phases)){
    const r=phaseTotals.get(key)||{calls:0,wallMs:0};
    r.calls+=value.calls;r.wallMs+=value.wallMs;
    phaseTotals.set(key,r);
  }
  decisions.push({match:i,seed,rounds:prof.rounds,winner:prof.winner,
    wallMs:prof.phaseProfile.totalWallMs,phases:prof.phaseProfile.phases});
}
const {profile:cpuProfile}=await post('Profiler.stop');
profiler.disconnect();
const byId=new Map(cpuProfile.nodes.map(node=>[node.id,node]));
const sampled=new Map(),byFile=new Map();
for(const id of cpuProfile.samples??[]){
  const frame=byId.get(id)?.callFrame;
  if(!frame)continue;
  const file=(frame.url||'unknown').split('/p-RL1024/').at(-1);
  const name=(frame.functionName||'(anonymous)')+' @ '+file+':'+frame.lineNumber;
  sampled.set(name,(sampled.get(name)||0)+1);
  byFile.set(file,(byFile.get(file)||0)+1);
}
const bySamples=(a,b)=>b[1]-a[1];
const cpuSummary={samples:cpuProfile.samples?.length??0,
  topFunctions:[...sampled].sort(bySamples).slice(0,35),
  topFiles:[...byFile].sort(bySamples).slice(0,20)};
const profile={schema:'p-rl1024-aq-phase-profile-v1',
  matches:count,seedRoot:root,modelSha256:hash(student),
  sourceCommit:process.env.GITHUB_SHA||null,node:process.version,arch:process.arch,
  wallMs:performance.now()-begin,
  phases:Object.fromEntries([...phaseTotals].sort((a,b)=>b[1].wallMs-a[1].wallMs)),
  matchReports:decisions,cpuSummary};
mkdirSync('artifacts',{recursive:true});
writeFileSync('artifacts/aq-phase-profile.json',JSON.stringify(profile,null,2)+'\n');
writeFileSync('artifacts/aq-v8-profile.cpuprofile',JSON.stringify(cpuProfile)+'\n');
console.log('AQ_CPU_SAMPLE '+JSON.stringify(cpuSummary));
console.log('AQ_PHASE_PROFILE '+JSON.stringify({
  modelSha256:profile.modelSha256,wallMs:profile.wallMs,matches:count,
  top:[...phaseTotals].sort((a,b)=>b[1].wallMs-a[1].wallMs).slice(0,15)}));
