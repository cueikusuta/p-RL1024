// RNAD-001 paired, unseen-seed *full four-win Match* evaluation.
// No automatic Champion promotion; N>=1000 and 95% CI are necessary,
// not sufficient. Opponent uses the frozen deterministic Stage0 baseline.
import {readFileSync,writeFileSync} from 'node:fs';
import {runRnadAcceptanceSmoke,stage0FixedPolicy,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {PersistentRNADInference,RNADMixedPolicy} from './infer.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const wilson=(successes,total)=>{
  if(!total)return [0,1];
  const z=1.959963984540054,p=successes/total,z2=z*z;
  const center=(p+z2/(2*total))/(1+z2/total);
  const radius=z*Math.sqrt(p*(1-p)/total+z2/(4*total*total))/(1+z2/total);
  return [Math.max(0,center-radius),Math.min(1,center+radius)];
};
export function evaluatePaired({modelPath,seed=987613,pairs=1}={}){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
    !Number.isSafeInteger(pairs)||pairs<1||pairs>100||
    typeof modelPath!=='string'||!modelPath)
    throw new TypeError('Invalid bounded RNAD evaluation');
  // Must be generated from a DIFFERENT, held-out seed range than training.
  let wins=0,rounds=0;
  const rows=[];
  for(let i=0;i<pairs;i++){
    const gameSeed=((seed+i-1)%0xffffffff)+1;
    for(const candidateSeat of [0,1]){
      const net=new RNADMixedPolicy({
        model:new PersistentRNADInference(modelPath),
        magnet:new PersistentRNADInference(modelPath),
        epsilon:0});
      const outcome=runRnadAcceptanceSmoke({
        seed:gameSeed,matchId:'eval-'+gameSeed+'-'+candidateSeat,
        experimentId:'RNAD-001-heldout',
        sourceSha:verifiedRunningSourceSha(),
        engineSnapshotSha:provenance.sourceCommit,
        policyVersion:'RNAD-001-eval',
        opponentPolicyVersion:'fixed-stage0',
        policyVersions:[0,1].map(seat=>seat===candidateSeat?
          'RNAD-001-eval':'fixed-stage0'),
        policy:ctx=>ctx.seat===candidateSeat?net.choose(ctx):
          stage0FixedPolicy(ctx)
      });
      if(!outcome.completed)throw new Error('Partial Match in paired evaluation');
      const won=outcome.winner===candidateSeat;
      if(won)wins++;
      rounds+=outcome.rounds;
      rows.push({seed:gameSeed,candidateSeat,won,score:outcome.score,
        rounds:outcome.rounds});
    }
  }
  const matches=rows.length;
  const [ciLower,ciUpper]=wilson(wins,matches);
  return {schema:'rnad-paired-full-match-eval-v1',
    source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,seed,pairs,matches,
    wins,win_rate:wins/matches,ci95:[ciLower,ciUpper],
    rounds,rows,is_champion:false,
    eligible_for_promotion_consideration:matches>=1000&&ciLower>.5};
}
if(process.argv[1]?.endsWith('/evaluate.mjs')){
  const arg={};
  for(let i=2;i<process.argv.length;i+=2){
    if(!process.argv[i]?.startsWith('--'))throw new Error('Invalid evaluation CLI');
    arg[process.argv[i].slice(2)]=process.argv[i+1];
  }
  if(!arg.model||!arg.out)throw new Error('Usage: --model PATH --seed N --pairs 1..100 --out PATH');
  const report=evaluatePaired({modelPath:arg.model,seed:Number(arg.seed??987613),
    pairs:Number(arg.pairs??1)});
  writeFileSync(arg.out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({matches:report.matches,wins:report.wins,
    win_rate:report.win_rate,ci95:report.ci95}));
}
