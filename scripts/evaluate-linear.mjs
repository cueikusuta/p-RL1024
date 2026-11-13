#!/usr/bin/env node
// I975 research-only, frozen zero-teacher pure RL vs fixed v2-E.
// The actor never receives v2-E ranks/scores; v2-E is a MATCH OPPONENT only.
// Training seeds and evaluation seeds are separated; all results are actual
// authoritative completed four-win Matches, not isolated battle pilots.
import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir,readdir} from 'node:fs/promises';
import {dirname,resolve,join,relative} from 'node:path';
import {performance} from 'node:perf_hooks';
import {cpus} from 'node:os';
import {mixSeed} from './dev/simulator/runner.js';
import {runStrategyMatch} from './dev/simulator/strategy-match.js';
import {validatePureRlModel,createPureRlSession}
  from '../src/ai/pure-rl-linear.js';
import {rankActions as rankPureRlActions} from '../models/linear.mjs';
import {rankPureRlPhaseActions} from '../src/ai/pure-rl-phases.js';

const SCHEMA='i975-frozen-paired-four-win-evaluation-v1';
const WEIGHTS=Object.freeze({blend:0.28,exposure:0.75,continuation:0.40});
const COHORTS=Object.freeze({
  smoke:{root:975441,domain:0x9741,min:1,max:4},
  screen:{root:975442,domain:0x9742,min:500,max:1000},
  confirm:{root:975443,domain:0x9743,min:1000,max:1000}
});
const sha256=value=>createHash('sha256').update(value).digest('hex');
const finite=x=>Number.isFinite(x);
const options=new Map();
if((process.argv.length-2)%2)throw new TypeError('Expected --key value arguments');
for(let i=2;i<process.argv.length;i+=2){
  const key=process.argv[i];
  if(!['--model','--out','--purpose','--blocks'].includes(key)||
    options.has(key))throw new TypeError('Unknown or duplicate argument');
  options.set(key,process.argv[i+1]);
}
const purpose=options.get('--purpose'),cohort=COHORTS[purpose];
const blockText=options.get('--blocks'),blocks=Number(blockText);
if(!cohort||!options.get('--model')||!options.get('--out')||
  !/^[1-9]\d*$/.test(blockText??'')||
  !Number.isSafeInteger(blocks)||blocks<cohort.min||blocks>cohort.max)
  throw new TypeError('Provide model, out, purpose and allowed seed blocks');

const modelPath=resolve(options.get('--model'));
const modelBytes=await readFile(modelPath);
const source=JSON.parse(modelBytes.toString('utf8'));
if(source.schema!=='i975-pure-rl-battle-train-result-v1'||
   source.trainingNotHoldout!==true||source.allPhases!==true||
   source.phase!=='full pure RL policy (draft, selection, battle)'||
   !Number.isInteger(source.completedMatches)||
   source.completedMatches<1||source.completedMatches>200000||
   !Array.isArray(source.matches)||
   source.matches.length!==source.completedMatches||
   !/^[a-f0-9]{64}$/.test(source.sourceSha256??'')||
   !/^[a-f0-9]{64}$/.test(source.modelSha256??''))
  throw new TypeError('A complete independently trained full-phase model is required');
const trainingAlgorithm=source.algorithm??'i975-true-online-sarsa-lambda-v1';
if(!['i975-true-online-sarsa-lambda-v1',
  'i975-episodic-monte-carlo-linear-v1'].includes(trainingAlgorithm))
  throw new TypeError('Unknown pure RL training algorithm');
const model=validatePureRlModel(source.model);
if(sha256(JSON.stringify(model))!==source.modelSha256)
  throw new TypeError('Frozen training model SHA mismatch');
const modelSha256=source.modelSha256;
const frozenSnapshotSha256=sha256(modelBytes);
const trainSeedSet=new Set(source.matches.map(row=>row.seed));
if(trainSeedSet.size!==source.completedMatches||
  [...trainSeedSet].some(n=>!Number.isSafeInteger(n)||n<1||n>0xffffffff))
  throw new TypeError('Corrupt training seed provenance');

const seeds=Array.from({length:blocks},(_,i)=>
  mixSeed(cohort.root,i,cohort.domain));
if(new Set(seeds).size!==blocks||
   seeds.some(s=>trainSeedSet.has(s)))
  throw new Error('Duplicate or training-overlapping evaluation seeds');
const fingerprint=createHash('sha256'),files=[];
async function collect(dir){
  for(const entry of await readdir(dir,{withFileTypes:true})){
    const full=join(dir,entry.name);
    if(entry.isDirectory())await collect(full);
    else if(entry.isFile() && /\.(?:js|mjs|cjs|json)$/.test(entry.name))
      files.push(full);
  }
}
// Capture the Battle / Match / offer / simulator dependencies, not a partial
// hand-curated list that can silently omit outcome-affecting implementations.
for(const dir of ['src','scripts/dev/simulator','data/catalog'])
  await collect(resolve(dir));
files.push(resolve('package.json'));
for(const full of files.sort()){
  fingerprint.update(relative(process.cwd(),full).replaceAll('\\','/')+'\0');
  fingerprint.update(await readFile(full));
}
const evaluationSourceSha256=fingerprint.digest('hex');
const sourceCompatible=source.sourceSha256===evaluationSourceSha256;
// True holdout evaluation must never silently accept a weight file trained
// against a different feature extractor, Match runtime or catalog.
if(purpose!=='smoke'&&!sourceCompatible)
  throw new Error('Formal cohort requires identical training and evaluation source');
const rankerForSeat=seat=>(weights,view)=>
  ['draft','selection'].includes(view.phase)
    ?rankPureRlPhaseActions(weights,view,{seat})
    :rankPureRlActions(weights,view);

const sortedPercentile=(values,p)=>{
  if(!values.length)return null;
  const sorted=[...values].sort((a,b)=>a-b);
  return sorted[Math.ceil(p*sorted.length)-1];
};
const blockWins=[],records=[],rlDecisionMs=[],phaseDurations=new Map();
const pureRlPhaseMs={draft:[],selection:[]};
const matchWallMs=[],matchCpuMs=[],heapEndMiB=[];
const started=performance.now();
let wins=0;
for(let block=0;block<blocks;block++){
  let pairedWins=0;
  for(const rlSeat of [0,1]){
    const actor=createPureRlSession(model,{train:false,epsilon:0,
      ranker:rankerForSeat(rlSeat)});
    const sessions=[null,null];sessions[rlSeat]=actor;
    const policies=rlSeat===0
      ?['pure-rl-linear','strategy-v2-e']
      :['strategy-v2-e','pure-rl-linear'];
    const result=runStrategyMatch({
      seed:seeds[block],policies,strategyMode:'integrated',
      v2Weights:WEIGHTS,pureRlAllPhases:true,pureRlSessions:sessions,
      capturePhaseTimings:true,captureDecisionSamples:true,
      capturePureRlPhaseTimings:true
    });
    if(result.completed!==true||![0,1].includes(result.winner)||
       result.score?.[result.winner]!==4||
       result.pureRlStats?.[rlSeat]?.train!==false||
       result.pureRlStats?.[rlSeat]?.updates!==0||
       result.pureRlStats?.[rlSeat]?.decisions!==result.actionCounts[rlSeat]+2*result.rounds)
      throw new Error('Incomplete or contaminated real four-win Match');
    const samples=result.decisionTimeMs[rlSeat].samples;
    if(!Array.isArray(samples)||!samples.length||
      samples.some(x=>!finite(x)||x<0))
      throw new Error('Missing pure RL decision timing');
    rlDecisionMs.push(...samples);
    const phaseTiming=result.pureRlPhaseTimeMs?.[rlSeat];
    for(const stage of ['draft','selection']){
      const stageSamples=phaseTiming?.[stage]?.samples;
      if(!Array.isArray(stageSamples)||
        stageSamples.length!==result.rounds||
        stageSamples.some(x=>!finite(x)||x<0))
        throw new Error('Missing own-seat pure RL '+stage+' latency samples');
      pureRlPhaseMs[stage].push(...stageSamples);
    }
    matchWallMs.push(result.phaseProfile.totalWallMs);
    matchCpuMs.push(Object.values(result.phaseProfile.cpuMs).reduce((a,b)=>a+b,0));
    heapEndMiB.push(result.phaseProfile.heapUsedMiB);
    for(const [name,row] of Object.entries(result.phaseProfile.phases)){
      const entry=phaseDurations.get(name)??{calls:0,wallMs:0};
      entry.calls+=row.calls;entry.wallMs+=row.wallMs;
      phaseDurations.set(name,entry);
    }
    const won=Number(result.winner===rlSeat);
    pairedWins+=won;wins+=won;
    records.push({block,seed:seeds[block],rlSeat,
      winner:result.winner,won,rounds:result.rounds,
      actionCounts:result.actionCounts,score:result.score});
  }
  blockWins.push(pairedWins);
  if((block+1)%25===0 || block+1===blocks)
    console.log(JSON.stringify({purpose,completeBlocks:block+1,
      completedMatches:2*(block+1),pureRlWins:wins}));
}
const mean=wins/(2*blocks);
// Each paired block, not each seat-Match, is one independent unit.
// Hoeffding's distribution-free two-sided 95% interval remains meaningful
// even for zero observed wins (unlike a zero-width sample-variance interval).
const halfWidth=blocks>=30
  ?Math.sqrt(Math.log(40)/(2*blocks)):null;
const timing={
  elapsedWallMs:performance.now()-started,
  completeMatchesPerSecond:2*blocks/((performance.now()-started)/1000),
  wallP50Ms:sortedPercentile(matchWallMs,.5),
  wallP95Ms:sortedPercentile(matchWallMs,.95),
  cpuP95Ms:sortedPercentile(matchCpuMs,.95),
  heapUsedEndP95MiB:sortedPercentile(heapEndMiB,.95),
  // Per-own-action latency, not end-to-end Match or opponent thinking time.
  rlDecisionCalls:rlDecisionMs.length,
  rlDecisionP50Ms:sortedPercentile(rlDecisionMs,.5),
  rlDecisionP95Ms:sortedPercentile(rlDecisionMs,.95),
  rlDecisionP99Ms:sortedPercentile(rlDecisionMs,.99),
  pureRlDraftCalls:pureRlPhaseMs.draft.length,
  pureRlDraftP95Ms:sortedPercentile(pureRlPhaseMs.draft,.95),
  pureRlDraftP99Ms:sortedPercentile(pureRlPhaseMs.draft,.99),
  pureRlSelectionCalls:pureRlPhaseMs.selection.length,
  pureRlSelectionP95Ms:sortedPercentile(pureRlPhaseMs.selection,.95),
  pureRlSelectionP99Ms:sortedPercentile(pureRlPhaseMs.selection,.99),
  // Phase durations include both contestants. Not RL-only timings.
  bothPoliciesPhases:Object.fromEntries([...phaseDurations.entries()]
    .sort(([a],[b])=>a.localeCompare(b)))
};
const report={
  schema:SCHEMA,purpose,
  formalEligible:purpose!=='smoke'&&2*blocks>=1000&&sourceCompatible,
  trainingAlgorithm,sourceCompatible,
  runtime:{nodeVersion:process.version,platform:process.platform,
    arch:process.arch,logicalCpus:cpus().length,
    cpuModel:cpus()[0]?.model??'unknown',workerThreads:0},
  comparison:'full-phase pure RL frozen model vs fixed strategy-v2-e',
  v2Weights:WEIGHTS,
  modelSha256,frozenSnapshotSha256,
  trainingSourceSha256:source.sourceSha256,
  trainingCompletedMatches:source.completedMatches,
  evaluationSourceSha256,
  seedRoot:cohort.root,seedDomain:cohort.domain,
  seedDigestSha256:sha256(JSON.stringify(seeds)),
  blocks,completedMatches:records.length,
  pureRlWins:wins,pureRlWinRate:mean,
  pairedBlockWins:blockWins,
  pairedHoeffding95:halfWidth===null?null:[
    Math.max(0,mean-halfWidth),Math.min(1,mean+halfWidth)],
  outcomeDigestSha256:sha256(JSON.stringify(records)),
  // Timing is noisy; kept OUTSIDE outcome digest/reproducibility proof.
  timing,records
};
const out=resolve(options.get('--out'));
await mkdir(dirname(out),{recursive:true});
await writeFile(out,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({schema:SCHEMA,purpose,
  completedMatches:report.completedMatches,pureRlWins:wins,
  modelSha256,trainingAlgorithm,
  outcomeDigestSha256:report.outcomeDigestSha256,
  pureRlDraftP95Ms:timing.pureRlDraftP95Ms,
  pureRlSelectionP95Ms:timing.pureRlSelectionP95Ms,
  rlDecisionP95Ms:timing.rlDecisionP95Ms,
  completeMatchesPerSecond:timing.completeMatchesPerSecond,
  out}));
