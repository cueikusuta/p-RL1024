// Optional RNAD-001 fixed-Policy teacher, entirely separate from pure selfplay.
// Only a real accepted four-win Match may produce imitation examples.
import {writeFileSync} from 'node:fs';
import {runRnadAcceptanceSmoke,stage0FixedPolicy,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {encodeRnadDecision,RNAD_FEATURE_SCHEMA}
  from '../../src/ml/rnad-features.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

export function createTeacherMatch(seed){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff)
    throw new TypeError('Invalid teacher seed');
  const match=runRnadAcceptanceSmoke({
    seed,matchId:'rnad-stage0-teacher-'+seed,
    experimentId:'RNAD-001-optional-stage0-imitation',
    sourceSha:verifiedRunningSourceSha(),
    engineSnapshotSha:provenance.sourceCommit,
    policyVersion:'stage0-fixed-teacher',
    policy:stage0FixedPolicy});
  if(!match.completed||!match.jointTimeline||!match.trajectories||
     ![0,1].includes(match.winner)||match.score[match.winner]!==4)
    throw new Error('Teacher data requires a complete authoritative Match');
  const examples=match.jointTimeline.map(({seat,index})=>{
    const record=match.trajectories[seat].decisions[index];
    if(record.player_seat!==seat||record.phase!=='battle')
      throw new Error('Invalid seat-local accepted teacher event');
    const f=encodeRnadDecision(record.observation,record.legal_actions);
    const choice=record.legal_actions.findIndex(x=>
      JSON.stringify(x)===JSON.stringify(record.accepted_action));
    if(choice<0||choice>=f.actions.length)
      throw new Error('Accepted teacher action is not legal');
    return {actor:seat,state:f.state,actions:f.actions,
      choice,terminal_target:seat===match.winner?1:-1};
  });
  if(examples.length!==match.actionCounts[0]+match.actionCounts[1]||
     !examples.some(e=>e.actor===0)||!examples.some(e=>e.actor===1))
    throw new Error('Teacher match acceptance count mismatch');
  return {schema:'rnad-fixed-stage0-teacher-match-v1',
    feature_schema:RNAD_FEATURE_SCHEMA,
    source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,
    seed,completed:true,winner:match.winner,score:match.score,
    teacher:'stage0-fixed-untrained',examples};
}
if(process.argv[1]?.endsWith('/export-teacher.mjs')){
  const args=process.argv.slice(2);
  if(args.length!==4||args[0]!=='--seed'||args[2]!=='--out')
    throw new Error('Usage: --seed N --out FILE');
  const trace=createTeacherMatch(Number(args[1]));
  writeFileSync(args[3],JSON.stringify(trace)+'\n',{flag:'wx'});
  process.stdout.write(JSON.stringify({seed:trace.seed,
    score:trace.score,examples:trace.examples.length})+'\n');
}
