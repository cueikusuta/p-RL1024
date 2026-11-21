// Direct neural-vs-neural evaluation with identical Match seeds,
// both seat assignments and an authoritative 4-win Match per result.
// Independent of Stage0 baseline; demonstrates actual relative strength.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {PersistentRNADInference,RNADMixedPolicy} from './infer.mjs';
import {RnadSwitchLoopGuard,RNAD_SWITCH_GUARD} from './switch-loop-guard.mjs';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA}
  from '../../src/ml/rnad-phase-features.js';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA_V2}
  from '../../src/ml/rnad-phase-features-v2.js';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA_V3} from '../../src/ml/rnad-phase-features-v3.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};
import {pairedHoeffding95} from './aggregate-eval.mjs';

const sha256=file=>createHash('sha256').update(readFileSync(file)).digest('hex');
const createPolicy=(path,selectionMode='sample',guard='none')=>{
  const policy=new RNADMixedPolicy({
    model:new PersistentRNADInference(path),
    magnet:new PersistentRNADInference(path),epsilon:0,selectionMode
  });
  return guard===RNAD_SWITCH_GUARD?new RnadSwitchLoopGuard(policy):policy;
};

export function directPairedNeuralEval({
  teacherPath,rnadPath,seed=988501,pairs=64,selectionMode='sample',
  teacherSwitchGuard='none',rnadSwitchGuard='none'}={}){
  if(typeof teacherPath!=='string'||!teacherPath||
    typeof rnadPath!=='string'||!rnadPath||
    !Number.isSafeInteger(seed)||seed<1||
    !Number.isSafeInteger(pairs)||pairs<1||pairs>100||
    seed+pairs-1>0xffffffff||
    !['sample','greedy'].includes(selectionMode)||
    !['none',RNAD_SWITCH_GUARD].includes(teacherSwitchGuard)||
    !['none',RNAD_SWITCH_GUARD].includes(rnadSwitchGuard))
    throw new TypeError('Invalid bounded neural-versus-neural evaluation');
  const teacherSha=sha256(teacherPath),rnadSha=sha256(rnadPath);
  if(teacherSha===rnadSha&&teacherSwitchGuard===rnadSwitchGuard)
    throw new Error('Expected different frozen models or deployment policies');
  const family=new PersistentRNADInference(teacherPath).featureSchema;
  if(family!==new PersistentRNADInference(rnadPath).featureSchema)
    throw new Error('Cannot directly compare different actor feature schemas');
  const threePhase=[RNAD_THREE_PHASE_FEATURE_SCHEMA,
    RNAD_THREE_PHASE_FEATURE_SCHEMA_V2,RNAD_THREE_PHASE_FEATURE_SCHEMA_V3].includes(family);
  const rows=[];
  let wins=0;
  for(let offset=0;offset<pairs;offset++){
    const sampleSeed=seed+offset;
    for(const rnadSeat of [0,1]){
      // Do not reuse GRU memory between matches or models.
      const teacher=createPolicy(teacherPath,selectionMode,teacherSwitchGuard);
      const rnad=createPolicy(rnadPath,selectionMode,rnadSwitchGuard);
      const version=threePhase?
        ['RNAD-002-teacher','RNAD-002-selfplay']:
        ['RNAD-001-teacher','RNAD-001-selfplay'];
      const perSeatPolicy=ctx=>ctx.seat===rnadSeat?
        rnad.choose(ctx):teacher.choose(ctx);
      // The guard's state only advances after the trusted Match collector
      // consumes its one-use accepted-action receipt; never on proposal.
      perSeatPolicy.accepted=ctx=>{
        (ctx.seat===rnadSeat?rnad:teacher).accepted?.(ctx);
      };
      const result=runRnadAcceptanceSmoke({
        includeAcceptedHistory:family===RNAD_THREE_PHASE_FEATURE_SCHEMA_V3,
        seed:sampleSeed,matchId:'direct-neural-'+sampleSeed+'-'+rnadSeat,
        experimentId:threePhase?'RNAD-002-direct-heldout':'RNAD-001-direct-heldout',
        sourceSha:verifiedRunningSourceSha(),
        engineSnapshotSha:provenance.sourceCommit,
        policyVersions:[0,1].map(seat=>{
          const guarded=seat===rnadSeat?rnadSwitchGuard:teacherSwitchGuard;
          return (seat===rnadSeat?version[1]:version[0])+
            (guarded===RNAD_SWITCH_GUARD?'-'+RNAD_SWITCH_GUARD:'');
        }),
        phasePolicies:threePhase?[perSeatPolicy,perSeatPolicy]:null,
        policy:perSeatPolicy
      });
      if(!result.completed||!Array.isArray(result.score)||
        result.score[result.winner]!==4||
        !result.trajectories?.every(t=>t.completed))
        throw new Error('Incomplete authoritative neural head-to-head Match');
      const rnadWon=result.winner===rnadSeat;
      if(rnadWon)wins++;
      rows.push({seed:sampleSeed,rnadSeat,won:rnadWon,
        score:result.score,rounds:result.rounds});
    }
  }
  const matches=2*pairs;
  const ci95=pairedHoeffding95(wins,pairs);
  const guarded=teacherSwitchGuard!== 'none'||rnadSwitchGuard!=='none';
  return {schema:guarded?'rnad-direct-switch-guard-paired-v1':
      'rnad-direct-neural-paired-match-v1',
    ...(guarded?{deployment_policy:{
      teacher_switch_guard:teacherSwitchGuard,
      rnad_switch_guard:rnadSwitchGuard
    }}:{}),
    source_sha:verifiedRunningSourceSha(),engine_sha:provenance.sourceCommit,
    teacher_model_sha256:teacherSha,rnad_model_sha256:rnadSha,
    model_feature_schema:family,inference_selection_mode:selectionMode,
    seed,pairs,matches,wins,win_rate:wins/matches,
    paired_seed_hoeffding_95pct:ci95,
    improvement_statistically_established:ci95[0]>.5,
    is_champion:false,rows};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const argv=process.argv.slice(2);
  if(argv.length<10||argv.length>16||argv.length%2!==0||
    argv.some((v,i)=>i%2===0&&!v.startsWith('--')))
    throw new Error('Usage: --teacher FILE --rnad FILE --seed N --pairs N --out FILE');
  const params={};
  for(let i=0;i<argv.length;i+=2){
    const key=argv[i].slice(2);
    if(!['teacher','rnad','seed','pairs','out','selection-mode',
      'guard-teacher','guard-rnad'].includes(key)||
      Object.hasOwn(params,key))throw new Error('Unknown duplicate CLI argument');
    params[key]=argv[i+1];
  }
  if(!Object.hasOwn(params,'out')||!Object.hasOwn(params,'teacher')||
    !Object.hasOwn(params,'rnad'))throw new Error('Missing required input');
  const result=directPairedNeuralEval({
    teacherPath:params.teacher,rnadPath:params.rnad,
    seed:Number(params.seed),pairs:Number(params.pairs),
    selectionMode:params['selection-mode']??'sample',
    teacherSwitchGuard:params['guard-teacher']??'none',
    rnadSwitchGuard:params['guard-rnad']??'none'});
  writeFileSync(params.out,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({matches:result.matches,wins:result.wins,
    win_rate:result.win_rate,ci95:result.paired_seed_hoeffding_95pct,
    significant:result.improvement_statistically_established}));
}
