// Optional RNAD-001 fixed-Policy teacher, entirely separate from pure selfplay.
// Only a real accepted four-win Match may produce imitation examples.
import {writeFileSync} from 'node:fs';
import {runRnadAcceptanceSmoke,stage0FixedPolicy,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {encodeRnadDecision,RNAD_FEATURE_SCHEMA}
  from '../../src/ml/rnad-features.js';
import {encodeRnadThreePhaseDecision,RNAD_THREE_PHASE_FEATURE_SCHEMA}
  from '../../src/ml/rnad-phase-features.js';
import {encodeRnadThreePhaseDecisionV2,RNAD_THREE_PHASE_FEATURE_SCHEMA_V2}
  from '../../src/ml/rnad-phase-features-v2.js';
import {encodeRnadThreePhaseDecisionV4,RNAD_THREE_PHASE_FEATURE_SCHEMA_V4}
  from '../../src/ml/rnad-phase-features-v4.js';
import {chooseTeam,chooseSelection} from '../dev/simulator/draft-ai.js';
import {playableSets} from '../../src/production-runtime.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};
import provenance from '../../provenance/manifest.json' with {type:'json'};
import {createPublicAQTeacher,AQ_TEACHER_ID,AQ_STRATEGY_SELECTION_TEACHER_ID,AQ_WEIGHTS_SHA256}
  from './aq-teacher-policy.mjs';

function stage0ThreePhaseTeacher({observation,legalActions,rng}){
  if(observation.phase==='battle')
    return stage0FixedPolicy({observation,legalActions});
  const action=observation.phase==='draft'?
    chooseTeam(observation,{data,rng}):
    chooseSelection(observation,{data,sets:playableSets,rng});
  if(!legalActions.some(candidate=>
      JSON.stringify(candidate)===JSON.stringify(action)))
    throw new Error('Trusted Basic phase action not in authoritative legal set');
  return {action,behaviorLogProbability:0,
    referenceLogProbability:-Math.log(legalActions.length),predictedValue:0};
}
export function createTeacherMatch(seed,{threePhase=false,teacher='basic-stage0',
  featuresV2=false,featuresV4=false}={}){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
    typeof threePhase!=='boolean'||typeof featuresV2!=='boolean'||
    typeof featuresV4!=='boolean'||
    ((featuresV2||featuresV4)&&!threePhase)||(featuresV2&&featuresV4)||
    !['basic-stage0','aq-public-belief','aq-public-strategy-selection'].includes(teacher)||
    (teacher!=='basic-stage0'&&!threePhase))
    throw new TypeError('Invalid teacher seed');
  const aq=teacher==='basic-stage0'?null:createPublicAQTeacher({
    selectionMode:teacher==='aq-public-strategy-selection'?'strategy-public':'tactical'});
  const teacherId=teacher==='aq-public-strategy-selection'?
    AQ_STRATEGY_SELECTION_TEACHER_ID:AQ_TEACHER_ID;
  const effectivePolicy=aq??stage0FixedPolicy;
  const match=runRnadAcceptanceSmoke({
    seed,matchId:'rnad-stage0-teacher-'+seed,
    experimentId:'RNAD-001-optional-stage0-imitation',
    sourceSha:verifiedRunningSourceSha(),
    engineSnapshotSha:provenance.sourceCommit,
    policyVersion:aq?teacherId:
      threePhase?'basic-stage0-3phase-teacher':'stage0-fixed-teacher',
    phasePolicy:aq??(threePhase?stage0ThreePhaseTeacher:null),
    includeAcceptedHistory:featuresV4,
    policy:effectivePolicy});
  if(!match.completed||!match.jointTimeline||!match.trajectories||
     ![0,1].includes(match.winner)||match.score[match.winner]!==4)
    throw new Error('Teacher data requires a complete authoritative Match');
  const examples=match.jointTimeline.map(({seat,index})=>{
    const record=match.trajectories[seat].decisions[index];
    if(record.player_seat!==seat||
      (!threePhase&&record.phase!=='battle')||
      !['draft','selection','battle'].includes(record.phase))
      throw new Error('Invalid seat-local accepted teacher event');
    const f=featuresV4?
      encodeRnadThreePhaseDecisionV4(record.observation,record.legal_actions):
      featuresV2?
      encodeRnadThreePhaseDecisionV2(record.observation,record.legal_actions):
      threePhase?
      encodeRnadThreePhaseDecision(record.observation,record.legal_actions):
      encodeRnadDecision(record.observation,record.legal_actions);
    const choice=record.legal_actions.findIndex(x=>
      JSON.stringify(x)===JSON.stringify(record.accepted_action));
    if(choice<0||choice>=f.actions.length)
      throw new Error('Accepted teacher action is not legal');
    return {actor:seat,phase:record.phase,
      state:f.state,actions:f.actions,
      choice,terminal_target:seat===match.winner?1:-1};
  });
  if(examples.length!==match.actionCounts[0]+match.actionCounts[1]+
    (threePhase?4*match.rounds:0)||
    (threePhase&&['draft','selection'].some(stage=>
      examples.filter(e=>e.phase===stage).length!==2*match.rounds))||
     !examples.some(e=>e.actor===0)||!examples.some(e=>e.actor===1))
    throw new Error('Teacher match acceptance count mismatch');
  return {schema:teacher==='aq-public-strategy-selection'?
      'rnad-aq-public-strategy-selection-three-phase-teacher-v1':
      aq?'rnad-aq-public-belief-three-phase-teacher-v1':
      threePhase?'rnad-basic-stage0-three-phase-teacher-v1':
      'rnad-fixed-stage0-teacher-match-v1',
    feature_schema:featuresV4?RNAD_THREE_PHASE_FEATURE_SCHEMA_V4:
      featuresV2?RNAD_THREE_PHASE_FEATURE_SCHEMA_V2:
      threePhase?RNAD_THREE_PHASE_FEATURE_SCHEMA:RNAD_FEATURE_SCHEMA,
    source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,
    seed,completed:true,winner:match.winner,score:match.score,
    teacher:aq?teacherId:
      threePhase?'basic-stage0-3phase':'stage0-fixed-untrained',
    ...(aq?{teacher_model_sha256:AQ_WEIGHTS_SHA256}:{}),
    rounds:match.rounds,action_counts:match.actionCounts,examples};
}
if(process.argv[1]?.endsWith('/export-teacher.mjs')){
  const args=process.argv.slice(2);
  if(args.length%2!==0)throw new Error('Invalid teacher CLI');
  const inputs={};
  for(let i=0;i<args.length;i+=2){
    const k=args[i]?.slice(2);
    if(!args[i]?.startsWith('--')||!['seed','out','three-phase','teacher','features-v2','features-v4'].includes(k)||
      Object.hasOwn(inputs,k))throw new Error('Invalid teacher CLI option');
    inputs[k]=args[i+1];
  }
  if(!inputs.seed||!inputs.out||
    (inputs['three-phase']!==undefined&&inputs['three-phase']!=='true')||
    (inputs['features-v2']!==undefined&&inputs['features-v2']!=='true')||
    (inputs['features-v4']!==undefined&&inputs['features-v4']!=='true'))
    throw new Error('Usage: --seed N --out FILE [--three-phase true]');
  const trace=createTeacherMatch(Number(inputs.seed),
    {threePhase:inputs['three-phase']==='true',
     teacher:inputs.teacher??'basic-stage0',
     featuresV2:inputs['features-v2']==='true',
     featuresV4:inputs['features-v4']==='true'});
  writeFileSync(inputs.out,JSON.stringify(trace)+'\n',{flag:'wx'});
  process.stdout.write(JSON.stringify({seed:trace.seed,
    score:trace.score,examples:trace.examples.length})+'\n');
}
