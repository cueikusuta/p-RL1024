// Research-only three-phase FULL Match acceptance smoke.
// It composes a frozen Deep Sets draft policy, the SHA-pinned A4096 selection
// policy and the built-in BASIC battle policy. NOT a trained three-phase AI,
// imitation gate, AQ victory test, league entry or Champion promotion.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {DeepSetsDraftPolicy} from './draft-deepsets-infer.mjs';
import {loadVerifiedASelection,checkASelectionParity,selectFrozenAAction}
  from './selection-a-public-infer.mjs';
import {encodeRichSelectionActor} from './export-selection-rich.mjs';
import {projectRnadObservation,rnadLegalActions}
  from '../../src/ml/legal-action-adapter.js';
import {createBuiltinPhaseAgents} from '../dev/simulator/phase-agents.js';
import {runPhaseMatch} from '../dev/simulator/phase-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const DRAFT_SOURCE='022aa9cd5b4ac37514df1d845ea00c11c3c13a5c';
const A_SOURCE='a7224b05eeef49650794b24546576119dffed951';
export const INTEGRATION_SMOKE_SCHEMA='rnad-i5-hybrid-threephase-a4096-acceptance-smoke-v1';

export function createHybridPhaseAgents({draftPath,draftSha,selectionPath,selectionSha}){
  const draft=new DeepSetsDraftPolicy(draftPath,{sha256:draftSha});
  const selection=loadVerifiedASelection(readFileSync(selectionPath),selectionSha);
  const parity=checkASelectionParity(selection);
  if(draft.trainingSourceSha!==DRAFT_SOURCE||
     selection.source_sha!==A_SOURCE||
     selection.engine_sha!==provenance.sourceCommit||
     selection.source_model_filename!=='selection-A-4096-e60-s90252.pt')
    throw new Error('Incompatible registered component source/engine/model');
  const basic=createBuiltinPhaseAgents({draft:'strategy',selection:'strategy',battle:'basic'});
  const agents={
    draft:{
      id:'draft-deepsets256-frozen',
      select({view,seat,context}){
        if(context.current?.draft!==null||view.phase!=='draft')
          throw new Error('Draft must start before accepted own decisions');
        const obs=projectRnadObservation(view,{seat});
        const legal=rnadLegalActions(view);
        const choice=draft.choose(obs,legal);
        if(!Number.isInteger(choice.index)||choice.index<0||
           choice.index>=legal.length)
          throw new Error('Deep Sets picked non-authoritative draft');
        return {action:legal[choice.index],
          handoff:{kind:'accepted-draft-proposal',picked:legal[choice.index]}};
      },
    },
    selection:{
      id:'selection-a4096-frozen',
      select({view,seat,context}){
        if(view.phase!=='selection'||!Array.isArray(context.current?.draft?.ids)||
           context.current.draft.ids.length!==6||
           !same(context.handoff.draft?.picked,context.current.draft.ids))
          throw new Error('Missing authoritative accepted draft handoff');
        const obs=projectRnadObservation(view,{seat});
        const legal=rnadLegalActions(view);
        const choice=selectFrozenAAction(selection,encodeRichSelectionActor(obs,legal));
        if(!Number.isInteger(choice.index)||choice.index<0||
           choice.index>=legal.length||
           legal[choice.index].some(id=>!context.current.draft.ids.includes(id)))
          throw new Error('A4096 picked a non-authoritative own selection');
        return {action:legal[choice.index],
          handoff:{kind:'accepted-selection-proposal',picked:legal[choice.index]}};
      },
    },
    battle:{
      id:'battle-basic-fixed',
      select(args){
        const ctx=args.context;
        if(!Array.isArray(ctx.current?.selection?.ids)||
           ctx.current.selection.ids.length!==3||
           !same(ctx.handoff.selection?.picked,ctx.current.selection.ids)||
           !ctx.current.selection.ids.every(id=>ctx.current.draft.ids.includes(id)))
          throw new Error('Missing authoritative accepted selection handoff');
        return basic.battle.select(args);
      },
    },
  };
  return {agents,source_sha:{draft:DRAFT_SOURCE,selection:A_SOURCE},
    model_sha256:{draft:draft.modelSha256,selection:selection.source_model_sha256},
    model_json_sha256:selectionSha,python_js_parity_max_error:parity};
}

export function smokeHybridThreePhase({draftPath,draftSha,selectionPath,selectionSha,
  seedStart=2501001}={}){
  if(!Number.isSafeInteger(seedStart)||seedStart<1||seedStart+1>0xffffffff)
    throw new TypeError('Invalid bounded smoke seed');
  const config={draftPath,draftSha,selectionPath,selectionSha};
  const preview=createHybridPhaseAgents(config);
  const rows=[];
  for(const seed of [seedStart,seedStart+1]){
    // Construct independent per-seat model sessions; do not share mutable game
    // memory, handoffs or accepted callbacks across players/matches.
    const agents=[0,1].map(()=>createHybridPhaseAgents(config).agents);
    const result=runPhaseMatch({seed,agents});
    if(!result.completed||![0,1].includes(result.winner)||
       result.score[result.winner]!==4||
       result.rounds<4||result.rounds>64||
       result.agentIds.some(ids=>
         ids.draft!=='draft-deepsets256-frozen'||
         ids.selection!=='selection-a4096-frozen'||
         ids.battle!=='battle-basic-fixed'))
      throw new Error('Three-phase hybrid authority-accepted four-win Match failed');
    rows.push({seed,completed:true,score:result.score,winner:result.winner,
      rounds:result.rounds,actionCounts:result.actionCounts,agentIds:result.agentIds});
  }
  return {schema:INTEGRATION_SMOKE_SCHEMA,policy_id:'draft256-a4096-basic-readonly',
    integrated_execution:true,trained_three_phase_policy:false,
    same_teacher_source_for_all_phases:false,complete_four_win_matches:rows.length,
    fresh_seat_scoped_authoritative_handoffs:true,
    ...Object.fromEntries(Object.entries(preview).filter(([key])=>key!=='agents')),
    selection_teacher_top1_not_game_win_rate:true,
    league_entry_gate_passed:false,league_started:false,champion_promoted:false,
    strength_established:false,rows};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2),options=new Map();
  if(args.length!==12)throw new Error(
    'Use --draft FILE --draft-sha SHA --selection FILE --selection-sha SHA --seed N --out FILE');
  const names=new Set(['--draft','--draft-sha','--selection','--selection-sha','--seed','--out']);
  for(let i=0;i<args.length;i+=2){
    if(!names.has(args[i])||options.has(args[i])||!args[i+1])
      throw new TypeError('Bad/repeated hybrid smoke argument');
    options.set(args[i],args[i+1]);
  }
  if(options.size!==6)throw new Error('Incomplete hybrid smoke arguments');
  const report=smokeHybridThreePhase({
    draftPath:options.get('--draft'),draftSha:options.get('--draft-sha'),
    selectionPath:options.get('--selection'),selectionSha:options.get('--selection-sha'),
    seedStart:Number(options.get('--seed'))});
  writeFileSync(options.get('--out'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({schema:report.schema,completions:report.complete_four_win_matches,
    policy_id:report.policy_id,league_entry_gate_passed:report.league_entry_gate_passed}));
}
