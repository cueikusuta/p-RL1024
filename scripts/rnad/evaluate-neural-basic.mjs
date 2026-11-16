// Independent, both-seat real four-win Match comparison of a 3-phase
// neural policy against immutable Basic draft/selection/battle.
import {writeFileSync,readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runPhaseMatch} from '../dev/simulator/phase-match.js';
import {createBuiltinPhaseAgents} from '../dev/simulator/phase-agents.js';
import {projectRnadObservation,rnadLegalActions}
  from '../../src/ml/legal-action-adapter.js';
import {PersistentRNADInference,RNADMixedPolicy} from './infer.mjs';
import {RnadSwitchLoopGuard,RNAD_SWITCH_GUARD} from './switch-loop-guard.mjs';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA}
  from '../../src/ml/rnad-phase-features.js';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA_V2}
  from '../../src/ml/rnad-phase-features-v2.js';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA_V3}
  from '../../src/ml/rnad-phase-features-v3.js';
import {pairedHoeffding95} from './aggregate-eval.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

export function greedyLegalIndex(logits){
  if(!Array.isArray(logits)||logits.length<1||
    logits.some(x=>typeof x!=='number'||!Number.isFinite(x)))
    throw new TypeError('Greedy mode requires finite bounded logits');
  let best=0;
  for(let i=1;i<logits.length;i++)if(logits[i]>logits[best])best=i;
  return best;
}
export function evaluateThreePhaseAgainstBasic({
  modelPath,seed=997001,pairs=128,selectionMode='sample',
  switchGuard='none'}={}){
  if(typeof modelPath!=='string'||!modelPath||
    !Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
    !Number.isSafeInteger(pairs)||pairs<1||pairs>128||
    seed+pairs-1>0xffffffff||
    !['sample','greedy'].includes(selectionMode)||
    !['none',RNAD_SWITCH_GUARD].includes(switchGuard))
    throw new TypeError('Invalid independent Basic opponent evaluation');
  const modelSha256=createHash('sha256').update(readFileSync(modelPath)).digest('hex');
  let wins=0;const rows=[];
  for(let i=0;i<pairs;i++)for(const candidateSeat of [0,1]){
    const model=new PersistentRNADInference(modelPath);
    const ai=switchGuard===RNAD_SWITCH_GUARD?
      new RnadSwitchLoopGuard(new RNADMixedPolicy({
        model,magnet:new PersistentRNADInference(modelPath),
        epsilon:0,selectionMode
      })):selectionMode==='sample'?new RNADMixedPolicy({
        model,magnet:new PersistentRNADInference(modelPath),epsilon:0}):null;
    const useHistory=model.featureSchema===RNAD_THREE_PHASE_FEATURE_SCHEMA_V3;
    if(![RNAD_THREE_PHASE_FEATURE_SCHEMA,RNAD_THREE_PHASE_FEATURE_SCHEMA_V2,
      RNAD_THREE_PHASE_FEATURE_SCHEMA_V3].includes(model.featureSchema))
      throw new Error('Requires separate true three-phase neural actor');
    const acceptedHistory={lastKind:'none',consecutiveSwitches:0,
      lastSwitchTarget:-1,lastActive:-1};
    function agent(phase){
      let pendingOwn=null;
      return {id:'RNAD-002-frozen-'+phase+
          (switchGuard===RNAD_SWITCH_GUARD?'-'+switchGuard:''),
        select({view,seat,rng,round}){
          const observation=projectRnadObservation(view,{seat});
          if(useHistory)observation.acceptedHistory={...acceptedHistory};
          pendingOwn={active:observation.self?.active??-1,
            requestType:observation.requestType??null};
          if(observation.phase!==phase)
            throw new Error('Wrong projected neural phase');
          const legalActions=rnadLegalActions(view);
          if(switchGuard===RNAD_SWITCH_GUARD)
            return ai.choose({observation,legalActions,seat,rng,round}).action;
          if(selectionMode==='greedy'){
            const result=model.inferProjected(observation,legalActions,seat);
            return legalActions[greedyLegalIndex(result.logits)];
          }
          return ai.choose({observation,legalActions,seat,rng}).action;
        },
        ...((switchGuard===RNAD_SWITCH_GUARD||useHistory)?{
          accepted({seat,round,action}){
            // Called after authority acceptance, not from the proposal.
            if(useHistory){
              if(phase==='battle'&&pendingOwn?.requestType==='turn'){
                acceptedHistory.lastActive=Number.isSafeInteger(pendingOwn.active)?
                  pendingOwn.active:-1;
                if(action?.kind==='switch'){
                  acceptedHistory.lastKind='switch';
                  acceptedHistory.consecutiveSwitches=
                    Math.min(3000,acceptedHistory.consecutiveSwitches+1);
                  acceptedHistory.lastSwitchTarget=Number.isSafeInteger(action.index)?
                    action.index:-1;
                }else{
                  acceptedHistory.lastKind='move';
                  acceptedHistory.consecutiveSwitches=0;
                  acceptedHistory.lastSwitchTarget=-1;
                }
              }else{
                acceptedHistory.lastKind=phase==='draft'?'draft':
                  phase==='selection'?'selection':'none';
                acceptedHistory.consecutiveSwitches=0;
                acceptedHistory.lastSwitchTarget=-1;
                acceptedHistory.lastActive=-1;
              }
            }
            pendingOwn=null;
            ai?.accepted?.({seat,round,phase,action});
          }
        }:{})};
    }
    const player=createBuiltinPhaseAgents({
      draft:agent('draft'),selection:agent('selection'),battle:agent('battle')});
    const basic=createBuiltinPhaseAgents({draft:'basic',selection:'basic',battle:'basic'});
    const outcome=runPhaseMatch({seed:seed+i,
      agents:[0,1].map(seat=>seat===candidateSeat?player:basic)});
    if(!outcome.completed||outcome.score[outcome.winner]!==4)
      throw new Error('Unfinished authoritative 4-win Basic evaluation');
    const won=outcome.winner===candidateSeat;
    wins+=Number(won);
    rows.push({seed:seed+i,candidateSeat,won,rounds:outcome.rounds,
      score:outcome.score});
  }
  return {schema:switchGuard===RNAD_SWITCH_GUARD?
      'rnad-three-phase-switch-guard-vs-basic-paired-v1':
      'rnad-three-phase-vs-basic-paired-v1',
    ...(switchGuard===RNAD_SWITCH_GUARD?
      {deployment_policy:{switch_guard:switchGuard}}:{}),
    model_sha256:modelSha256,source_sha:verifiedRunningSourceSha(),
    inference_selection_mode:selectionMode,
    engine_sha:provenance.sourceCommit,opponent_id:'basic-all-phases',
    pairs,matches:2*pairs,wins,win_rate:wins/(2*pairs),
    ci95:pairedHoeffding95(wins,pairs),rows,champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const a=process.argv.slice(2),opts=new Map();
  const allowed=new Set(['--model','--seed','--pairs','--out','--selection-mode',
    '--switch-guard']);
  if(![8,10,12].includes(a.length))throw new Error('Invalid paired evaluation CLI');
  for(let i=0;i<a.length;i+=2){
    if(!allowed.has(a[i])||opts.has(a[i])||typeof a[i+1]!=='string')
      throw new Error('Invalid or duplicate paired evaluation option');
    opts.set(a[i],a[i+1]);
  }
  if(!opts.has('--model')||!opts.has('--seed')||
    !opts.has('--pairs')||!opts.has('--out'))
    throw new Error('Required --model, --seed, --pairs and --out');
  const report=evaluateThreePhaseAgainstBasic({
    modelPath:opts.get('--model'),
    seed:Number(opts.get('--seed')),pairs:Number(opts.get('--pairs')),
    selectionMode:opts.get('--selection-mode')??'sample',
    switchGuard:opts.get('--switch-guard')??'none'});
  writeFileSync(opts.get('--out'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({matches:report.matches,wins:report.wins,
    win_rate:report.win_rate,ci95:report.ci95}));
}
