// RNAD-001 trusted two-seat full-Match rollout exporter. Only Match-issued
// accepted receipts can enter the training file; Actor receives no opponent
// hidden state and never sees the opponent's uncommitted simultaneous action.
import {writeFileSync} from 'node:fs';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {PersistentRNADInference,RNADMixedPolicy} from './infer.mjs';
import {encodeRnadDecision,RNAD_FEATURE_SCHEMA} from '../../src/ml/rnad-features.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

function argsMap(argv){
  const r={};
  for(let i=0;i<argv.length;i+=2){
    const k=argv[i],v=argv[i+1];
    if(!k?.startsWith('--')||v===undefined)throw new Error('Invalid RNAD CLI argument');
    r[k.slice(2)]=v;
  }
  return r;
}
export function createLearningTrajectory({seed,modelPath,magnetPath,
  matchId='rnad-match-'+seed,epsilon=.02}={}){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff)
    throw new TypeError('Invalid bounded RNAD seed');
  const neural=new RNADMixedPolicy({
    model:new PersistentRNADInference(modelPath),
    magnet:new PersistentRNADInference(magnetPath),epsilon
  });
  const result=runRnadAcceptanceSmoke({
    seed,matchId,experimentId:'RNAD-001',
    sourceSha:verifiedRunningSourceSha(),
    engineSnapshotSha:provenance.sourceCommit,
    policyVersion:'RNAD-001-neural-learner',
    opponentPolicyVersion:'RNAD-001-neural-learner',
    policy:args=>neural.choose(args)
  });
  if(!result.completed||!result.jointTimeline||
    result.jointTimeline.length===0||result.trajectories.length!==2)
    throw new Error('Only completed accepted Matches may enter learning');
  const events=result.jointTimeline.map(({seat,index})=>{
    const row=result.trajectories[seat].decisions[index];
    if(row.player_seat!==seat||row.phase!=='battle')
      throw new Error('Invalid authority timeline/seat boundary');
    const encoded=encodeRnadDecision(row.observation,row.legal_actions);
    const actionIndex=row.legal_actions.findIndex(a=>
      JSON.stringify(a)===JSON.stringify(row.accepted_action));
    if(actionIndex<0)throw new Error('Missing accepted legal action');
    return {actor:seat,round:row.round,turn:row.observation.turn,
      state:encoded.state,actions:encoded.actions,action_index:actionIndex,
      behavior_prob:Math.exp(row.behavior_log_probability),
      magnet_log_prob:row.reference_log_probability,
      reward:[0,0]};
  });
  const payoff0=result.winner===0?1:-1;
  events.at(-1).reward=[payoff0,-payoff0];
  return {schema:'rnad-accepted-training-match-v1',
    feature_schema:RNAD_FEATURE_SCHEMA,match_id:matchId,
    source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,seed,completed:true,
    winner:result.winner,score:result.score,rounds:result.rounds,
    action_counts:result.actionCounts,events};
}
if(process.argv[1]?.endsWith('/export-trajectory.mjs')){
  const a=argsMap(process.argv.slice(2));
  if(!a.model||!a.magnet||!a.out||!a.seed)
    throw new Error('Usage: --model MODEL.json --magnet REF.json --seed N --out FILE');
  const seed=Number(a.seed);
  const match=createLearningTrajectory({seed,modelPath:a.model,magnetPath:a.magnet});
  writeFileSync(a.out,JSON.stringify(match)+'\n',{flag:'wx'});
  process.stderr.write(JSON.stringify({seed,winner:match.winner,score:match.score,
    rounds:match.rounds,events:match.events.length})+'\n');
}
