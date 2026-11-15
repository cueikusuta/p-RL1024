// Compare actual public AQ/Strategy/Tactical teacher to immutable Basic
// controls before collecting supervised labels from its accepted Match actions.
import {writeFileSync} from 'node:fs';
import {runPhaseMatch} from '../dev/simulator/phase-match.js';
import {createBuiltinPhaseAgents} from '../dev/simulator/phase-agents.js';
import {projectRnadObservation,rnadLegalActions}
  from '../../src/ml/legal-action-adapter.js';
import {createPublicAQTeacher,AQ_WEIGHTS_SHA256,AQ_TEACHER_ID}
  from './aq-teacher-policy.mjs';
import {pairedHoeffding95} from './aggregate-eval.mjs';
import {verifiedRunningSourceSha,runRnadAcceptanceSmoke}
  from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};
import {pathToFileURL} from 'node:url';

export function benchmarkAQTeacher({seed=994601,pairs=32}={}){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
    !Number.isSafeInteger(pairs)||pairs<1||pairs>100||seed+pairs-1>0xffffffff)
    throw new TypeError('Invalid AQ teacher baseline pilot');
  const rows=[];let wins=0;
  for(let k=0;k<pairs;k++)for(const candidateSeat of [0,1]){
    const choose=createPublicAQTeacher();
    const teacherHandler=(phase)=>({
      id:AQ_TEACHER_ID+'-'+phase,
      select:({view,seat,round})=>{
        const projected=projectRnadObservation(view,{seat,
          context:null});
        if(projected.phase!==phase)
          throw new TypeError('Wrong frozen teacher phase');
        return choose({observation:projected,
          legalActions:rnadLegalActions(view),seat,round}).action;
      },
      accepted:({action,round})=>{
        choose.accepted?.({seat:candidateSeat,round,action});
      }
    });
    const candidate=createBuiltinPhaseAgents({
      draft:teacherHandler('draft'),
      selection:teacherHandler('selection'),
      battle:teacherHandler('battle')});
    const control=createBuiltinPhaseAgents({
      draft:'basic',selection:'basic',battle:'basic'});
    const result=runPhaseMatch({seed:seed+k,
      agents:[0,1].map(s=>s===candidateSeat?candidate:control)});
    if(!result.completed||result.score[result.winner]!==4)
      throw new Error('Incomplete AQ versus Basic four-win Match');
    const won=result.winner===candidateSeat;
    wins+=Number(won);
    rows.push({seed:seed+k,candidateSeat,won,
      rounds:result.rounds,score:result.score});
  }
  return {schema:'rnad-aq-public-teacher-paired-v1',
    source_sha:verifiedRunningSourceSha(),engine_sha:provenance.sourceCommit,
    teacher_id:AQ_TEACHER_ID,
    frozen_aq_model_sha256:AQ_WEIGHTS_SHA256,
    control:'basic-draft-selection-battle',
    seed,pairs,matches:2*pairs,wins,win_rate:wins/(2*pairs),
    independent_seed_hoeffding_ci95:pairedHoeffding95(wins,pairs),
    rows,is_champion:false};
}
export function verifyTeacherAcceptedMatch(seed=994599){
 const choose=createPublicAQTeacher();
 const match=runRnadAcceptanceSmoke({
   seed,matchId:'rnad-aq-teacher-smoke-'+seed,
   experimentId:'rnad-aq-teacher-authority-smoke',
   sourceSha:verifiedRunningSourceSha(),
   engineSnapshotSha:provenance.sourceCommit,
   phasePolicy:choose,policy:choose,
   policyVersion:AQ_TEACHER_ID});
 if(!match.completed||match.score[match.winner]!==4||
    match.jointTimeline.length!==match.actionCounts[0]+match.actionCounts[1]+
      4*match.rounds)
   throw new Error('AQ teacher lacked all-phase accepted authority receipts');
 return {seed,rounds:match.rounds,accepted:match.jointTimeline.length,
    completed:true};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2);
  if(args.length!==6||args[0]!=='--seed'||args[2]!=='--pairs'||args[4]!=='--out')
    throw new Error('Usage --seed N --pairs N --out FILE');
  console.log(JSON.stringify(verifyTeacherAcceptedMatch()));
  const report=benchmarkAQTeacher({seed:Number(args[1]),pairs:Number(args[3])});
  writeFileSync(args[5],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({wins:report.wins,matches:report.matches,
    win_rate:report.win_rate,ci95:report.independent_seed_hoeffding_ci95}));
}
