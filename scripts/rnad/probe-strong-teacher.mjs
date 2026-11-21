// Separate experiment: benchmark authorized built-in phase agents before
// treating any combination as a "strong" source for RNAD supervised pretraining.
// No trainer data crosses the opponent-information boundary.
import {writeFileSync} from 'node:fs';
import {runPhaseMatch} from '../dev/simulator/phase-match.js';
import {createBuiltinPhaseAgents} from '../dev/simulator/phase-agents.js';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};
import {pairedHoeffding95} from './aggregate-eval.mjs';
import {pathToFileURL} from 'node:url';

export const CANDIDATE_PROFILES=Object.freeze({
  strategy_tactical_intermediate: Object.freeze({draft:'strategy',selection:'tactical',battle:'intermediate'}),
  strategy_strategy_intermediate: Object.freeze({draft:'strategy',selection:'strategy',battle:'intermediate'}),
  basic_basic_intermediate: Object.freeze({draft:'basic',selection:'basic',battle:'intermediate'})
});
const OPPONENT=Object.freeze({draft:'basic',selection:'basic',battle:'basic'});
export function probeBuiltInTeachers({seed=994501,pairs=32}={}){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
    !Number.isSafeInteger(pairs)||pairs<1||pairs>100||seed+pairs-1>0xffffffff)
    throw new TypeError('Invalid bounded held-out phase profile study');
  const results={};
  for(const [id,profile] of Object.entries(CANDIDATE_PROFILES)){
    let wins=0;
    const rows=[];
    for(let i=0;i<pairs;i++){
      const sampleSeed=seed+i;
      for(const candidateSeat of [0,1]){
        const agents=[0,1].map(seat=>createBuiltinPhaseAgents(
          seat===candidateSeat?profile:OPPONENT));
        const result=runPhaseMatch({seed:sampleSeed,agents});
        if(!result.completed||result.score[result.winner]!==4||
          ![0,1].includes(result.winner))
          throw new Error('Incomplete authoritative candidate Match');
        const won=result.winner===candidateSeat;
        wins+=Number(won);
        rows.push({seed:sampleSeed,candidateSeat,won,rounds:result.rounds,
          score:result.score});
      }
    }
    results[id]={profile,pairs,matches:2*pairs,wins,
      win_rate:wins/(2*pairs),paired_seed_hoeffding_95pct:
      pairedHoeffding95(wins,pairs),rows};
    console.log(JSON.stringify({profile:id,matches:2*pairs,wins,
      win_rate:wins/(2*pairs)}));
  }
  return {schema:'rnad-built-in-teacher-probe-v1',source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,seed,pairs,
    opponent:OPPONENT,results,is_champion:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2);
  if(args.length!==6||args[0]!=='--seed'||args[2]!=='--pairs'||args[4]!=='--out')
    throw new Error('Usage: --seed N --pairs N --out FILE');
  const report=probeBuiltInTeachers({seed:Number(args[1]),pairs:Number(args[3])});
  writeFileSync(args[5],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
}
