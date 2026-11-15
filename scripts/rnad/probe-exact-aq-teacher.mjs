// True frozen AQ-Base policy with its original public matchup/belief, V2E
// baseline, tactical plan and anti-stall path (NOT a simplified adapter).
// This uses the unmodified provenance-frozen strategy Match engine.
import {writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent} from '../../models/aq-base.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};
import {pairedHoeffding95} from './aggregate-eval.mjs';
export function exactAQBaseline({seed=994701,pairs=64}={}){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
    !Number.isSafeInteger(pairs)||pairs<1||pairs>100||
    seed+pairs-1>0xffffffff)
    throw new TypeError('Invalid frozen AQ-Base research bounds');
  const aq=createOpponent();const rows=[];let wins=0;
  for(let offset=0;offset<pairs;offset++)
    for(const seat of [0,1]){
      const result=runStrategyMatch({seed:seed+offset,
        policies:[0,1].map(i=>i===seat?'aq-base':'basic'),
        strategyMode:'integrated',v2Weights:{
          blend:.28,exposure:.75,continuation:.4},
        aqBase:aq});
      if(!result.completed||result.score[result.winner]!==4||
        ![0,1].includes(result.winner))
        throw new Error('Incomplete original AQ-Base full Match');
      const won=result.winner===seat;
      wins+=Number(won);
      rows.push({seed:seed+offset,candidateSeat:seat,won,
        rounds:result.rounds,score:result.score});
    }
  return {schema:'rnad-exact-frozen-aq-vs-basic-match-v1',
    source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,
    aq_model_sha256:aq.modelSha256,aq_report_sha256:aq.reportSha256,
    opponent_id:'basic-all-phases',seed,pairs,matches:2*pairs,
    wins,win_rate:wins/(2*pairs),ci95:pairedHoeffding95(wins,pairs),
    rows,is_champion:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2);
  if(args.length!==6||args[0]!=='--seed'||args[2]!=='--pairs'||args[4]!=='--out')
    throw new Error('Usage: --seed N --pairs N --out FILE');
  const report=exactAQBaseline({seed:Number(args[1]),pairs:Number(args[3])});
  writeFileSync(args[5],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({matches:report.matches,wins:report.wins,
    win_rate:report.win_rate,ci95:report.ci95}));
}
