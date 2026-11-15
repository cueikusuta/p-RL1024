// True four-win Match phase ablation of ONE frozen RNAD-002 GRU model.
// The same weights are used for every profile and recurrent memory advances
// on accepted own draft/selection observations even when another agent chooses.
// No opponent hidden data, no learning or Champion promotion.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runPhaseMatch} from '../dev/simulator/phase-match.js';
import {createBuiltinPhaseAgents} from '../dev/simulator/phase-agents.js';
import {projectRnadObservation,rnadLegalActions} from '../../src/ml/legal-action-adapter.js';
import {PersistentRNADInference,argmaxRnad} from './infer.mjs';
import {pairedHoeffding95} from './aggregate-eval.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

export const PHASE_ABLATION_PROFILES=Object.freeze({
  neural_three:Object.freeze({draft:'neural',selection:'neural',battle:'neural'}),
  basic_then_neural:Object.freeze({draft:'basic',selection:'basic',battle:'neural'}),
  strategy_tactical_then_neural:Object.freeze({
    draft:'strategy',selection:'tactical',battle:'neural'})
});
function buildCandidate(modelPath,profile){
  const model=new PersistentRNADInference(modelPath);
  if(model.featureSchema!=='rnad-three-phase-features-v1')
    throw new TypeError('Requires frozen original RNAD-002 three-phase model');
  const builtins=createBuiltinPhaseAgents({
    draft:profile.draft==='neural'?'basic':profile.draft,
    selection:profile.selection==='neural'?'basic':profile.selection,
    battle:'basic'});
  const out={};
  for(const phase of ['draft','selection','battle']){
    out[phase]={id:'rnad-ablation-'+phase,
      select(args){
        const {view,seat}=args;
        const obs=projectRnadObservation(view,{seat});
        if(obs.phase!==phase)
          throw new Error('Wrong phase at RNAD ablation Actor');
        const legal=rnadLegalActions(view);
        // Recurrence always consumes every accepted own phase observation
        // so Basic-phase and all-neural share a comparable memory schedule.
        const logits=model.inferProjected(obs,legal,seat).logits;
        if(profile[phase]==='neural')
          return legal[argmaxRnad(logits)];
        const decision=builtins[phase].select(args);
        if(!legal.some(x=>JSON.stringify(x)===JSON.stringify(decision)))
          throw new Error('Built-in phase policy returned illegal Match action');
        return decision;
      }};
  }
  return out;
}
export function runPhaseAblation({modelPath,seed=997301,pairs=128}={}){
  if(!modelPath||typeof modelPath!=='string'||
     !Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
     !Number.isSafeInteger(pairs)||pairs<1||pairs>128||
     seed+pairs-1>0xffffffff)
    throw new TypeError('Invalid frozen 3-phase ablation bounds');
  const modelSha256=createHash('sha256')
    .update(readFileSync(modelPath)).digest('hex');
  const profiles={};
  for(const [id,profile] of Object.entries(PHASE_ABLATION_PROFILES)){
    let wins=0;const rows=[];
    for(let k=0;k<pairs;k++)for(const candidateSeat of [0,1]){
      const candidate=buildCandidate(modelPath,profile);
      const baseline=createBuiltinPhaseAgents({
        draft:'basic',selection:'basic',battle:'basic'});
      const result=runPhaseMatch({seed:seed+k,
        agents:[0,1].map(seat=>seat===candidateSeat?candidate:baseline)});
      if(!result.completed||result.score[result.winner]!==4||
        ![0,1].includes(result.winner))
        throw new Error('Unaccepted 4-win phase-ablation evaluation');
      const won=result.winner===candidateSeat;
      wins+=Number(won);
      rows.push({seed:seed+k,candidateSeat,won,
        rounds:result.rounds,score:result.score});
    }
    profiles[id]={agents:profile,wins,matches:2*pairs,
      win_rate:wins/(2*pairs),ci95:pairedHoeffding95(wins,pairs),rows};
    console.log(JSON.stringify({id,wins,matches:2*pairs,
      win_rate:wins/(2*pairs)}));
  }
  return {schema:'rnad-002-frozen-512-phase-ablation-v1',
    source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,model_sha256:modelSha256,
    seed,pairs,opponent:'basic-all-phases',
    profiles,is_champion:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const argv=process.argv.slice(2);
  if(argv.length!==8||argv[0]!=='--model'||argv[2]!=='--seed'||
     argv[4]!=='--pairs'||argv[6]!=='--out')
    throw new Error('Usage --model FILE --seed N --pairs N --out FILE');
  const report=runPhaseAblation({modelPath:argv[1],
    seed:Number(argv[3]),pairs:Number(argv[5])});
  writeFileSync(argv[7],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
}
