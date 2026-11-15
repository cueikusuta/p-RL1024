// RNAD-001 held-out full-Match Champion *eligibility* gate. The file
// is evidence only. Independent reviewer approval is still mandatory.
import {readFileSync,writeFileSync} from 'node:fs';
const sha=x=>typeof x==='string'&&/^[0-9a-f]{40,64}$/.test(x);
export function wilson(successes,total){
  if(!Number.isInteger(total)||total<1||!Number.isInteger(successes)||
    successes<0||successes>total)throw new Error('Invalid binomial counts');
  const z=1.959963984540054,p=successes/total,z2=z*z;
  const norm=1+z2/total;
  const center=(p+z2/(2*total))/norm;
  const radius=z*Math.sqrt(p*(1-p)/total+z2/(4*total*total))/norm;
  return [Math.max(0,center-radius),Math.min(1,center+radius)];
}
// Hoeffding bound on INDEPENDENT SEED BLOCKS, treating each swapped-seat
// seed as one bounded [0,1] pair score. Wilson on 2N correlated matches
// would be overconfident and must not authorize Champion promotion.
export function pairedHoeffding95(wins,matchedPairs){
  if(!Number.isInteger(matchedPairs)||matchedPairs<1||
    !Number.isInteger(wins)||wins<0||wins>2*matchedPairs)
    throw new TypeError('Invalid paired statistics');
  const score=wins/(2*matchedPairs);
  const error=Math.sqrt(Math.log(40)/(2*matchedPairs));
  return [Math.max(0,score-error),Math.min(1,score+error)];
}
export function aggregateIndependentPaired(reports,{minMatches=1000}={}){
  if(!Array.isArray(reports)||!reports.length||!Number.isInteger(minMatches)||
    minMatches<1000)throw new TypeError('Insufficient Champion evidence inputs');
  let model=null,engine=null,opponent=null,wins=0;
  const played=new Set(),bySeed=new Map();
  for(const report of reports){
    if(report?.schema!=='rnad-paired-full-match-eval-v1'||
      !sha(report.model_sha256)||!sha(report.engine_sha)||
      report.opponent_id!=='stage0-fixed-untrained'||
      !Array.isArray(report.rows)||report.rows.length!==report.matches||
      report.rows.length!==report.pairs*2)
      throw new TypeError('Malformed paired full-Match report');
    if(model!==null&&(model!==report.model_sha256||engine!==report.engine_sha||
        opponent!==report.opponent_id))
      throw new Error('Mixed model/engine/opponent is not a paired comparison');
    model=report.model_sha256;engine=report.engine_sha;
    opponent=report.opponent_id;
    for(const row of report.rows){
      if(!Number.isInteger(row.seed)||row.seed<1||row.seed>0xffffffff||
        ![0,1].includes(row.candidateSeat)||
        typeof row.won!=='boolean'||!Array.isArray(row.score)||
        row.score.length!==2||!row.score.every(x=>Number.isInteger(x)&&x>=0&&x<=4)||
        row.score[row.won?row.candidateSeat:1-row.candidateSeat]!==4)
        throw new TypeError('Invalid held-out authoritative outcome');
      const key=row.seed+':'+row.candidateSeat;
      if(played.has(key))throw new Error('Reused seed/seat in held-out evaluation');
      played.add(key);
      const seats=bySeed.get(row.seed)??new Set();
      seats.add(row.candidateSeat);bySeed.set(row.seed,seats);
      wins+=Number(row.won);
    }
  }
  if([...bySeed.values()].some(set=>set.size!==2))
    throw new Error('Unpaired seed detected');
  const matches=played.size;
  const ci95=pairedHoeffding95(wins,bySeed.size);
  return {schema:'rnad-champion-eligibility-v1',model_sha256:model,
    engine_sha:engine,opponent_id:opponent,
    unique_paired_seeds:bySeed.size,matches,wins,
    win_rate:wins/matches,ci95,ci_method:'independent-seed-block-Hoeffding',
    diagnostic_wilson_ci95:wilson(wins,matches),
    eligible_for_independent_champion_review:matches>=minMatches&&ci95[0]>.5,
    champion_promoted:false};
}
if(process.argv[1]?.endsWith('/aggregate-eval.mjs')){
  const args=process.argv.slice(2);
  if(args.length<3||args[0]!=='--out')
    throw new Error('Usage: --out report.json paired-1.json paired-2.json ...');
  const result=aggregateIndependentPaired(args.slice(2).map(p=>
    JSON.parse(readFileSync(p,'utf8'))));
  writeFileSync(args[1],JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(result));
}
