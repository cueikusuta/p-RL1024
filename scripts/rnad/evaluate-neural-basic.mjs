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
import {RNAD_THREE_PHASE_FEATURE_SCHEMA}
  from '../../src/ml/rnad-phase-features.js';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA_V2}
  from '../../src/ml/rnad-phase-features-v2.js';
import {pairedHoeffding95} from './aggregate-eval.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

export function evaluateThreePhaseAgainstBasic({modelPath,seed=997001,pairs=128}={}){
  if(typeof modelPath!=='string'||!modelPath||
    !Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
    !Number.isSafeInteger(pairs)||pairs<1||pairs>128||
    seed+pairs-1>0xffffffff)
    throw new TypeError('Invalid independent Basic opponent evaluation');
  const modelSha256=createHash('sha256').update(readFileSync(modelPath)).digest('hex');
  let wins=0;const rows=[];
  for(let i=0;i<pairs;i++)for(const candidateSeat of [0,1]){
    const ai=new RNADMixedPolicy({
      model:new PersistentRNADInference(modelPath),
      magnet:new PersistentRNADInference(modelPath),epsilon:0});
    if(![RNAD_THREE_PHASE_FEATURE_SCHEMA,RNAD_THREE_PHASE_FEATURE_SCHEMA_V2]
      .includes(ai.model.featureSchema))
      throw new Error('Requires separate true three-phase neural actor');
    function agent(phase){
      return {id:'RNAD-002-frozen-'+phase,
        select({view,seat,rng}){
          const observation=projectRnadObservation(view,{seat});
          if(observation.phase!==phase)
            throw new Error('Wrong projected neural phase');
          return ai.choose({observation,legalActions:rnadLegalActions(view),seat,rng}).action;
        }};
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
  return {schema:'rnad-three-phase-vs-basic-paired-v1',
    model_sha256:modelSha256,source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,opponent_id:'basic-all-phases',
    pairs,matches:2*pairs,wins,win_rate:wins/(2*pairs),
    ci95:pairedHoeffding95(wins,pairs),rows,champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const a=process.argv.slice(2);
  if(a.length!==8||a[0]!=='--model'||a[2]!=='--seed'||
    a[4]!=='--pairs'||a[6]!=='--out')throw new Error('Usage --model P --seed N --pairs N --out P');
  const report=evaluateThreePhaseAgainstBasic({modelPath:a[1],
    seed:Number(a[3]),pairs:Number(a[5])});
  writeFileSync(a[7],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({matches:report.matches,wins:report.wins,
    win_rate:report.win_rate,ci95:report.ci95}));
}
