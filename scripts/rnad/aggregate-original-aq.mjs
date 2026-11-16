// Strict independent-SEED, both-seat aggregation for the ORIGINAL AQ-Base
// strategy Match. One seed contributes both seat assignments; 2N games
// are not incorrectly counted as 2N independent seeds for confidence.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {pairedHoeffding95} from './aggregate-eval.mjs';
import {FULL_AQ_EVALUATION_SCHEMA} from './evaluate-vs-aq-full.mjs';
import {weightsSha256 as AQ_BASE_SHA} from '../../models/aq-base.mjs';

const hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const commit=x=>typeof x==='string'&&/^[a-f0-9]{40}$/.test(x);
export function aggregateOriginalAQReports(reports,{seedStart,seedCount}={}){
  if(!Array.isArray(reports)||reports.length<1||
     !Number.isSafeInteger(seedStart)||seedStart<1||
     !Number.isSafeInteger(seedCount)||seedCount<1||seedCount>2048||
     seedStart+seedCount-1>0xffffffff)
    throw new TypeError('Invalid bounded original AQ aggregate seed schedule');
  let reference=null;
  const bySeed=new Map(),collected=new Set();
  let wins=0,matches=0;
  for(const r of reports){
    if(r?.schema!==FULL_AQ_EVALUATION_SCHEMA||
       !hash(r.aq_base_model_sha256)||r.aq_base_model_sha256!==AQ_BASE_SHA||
       !hash(r.model_sha256)||!commit(r.source_sha)||
       !commit(r.engine_sha)||r.evaluation_policy!=='greedy-no-training'||
       r.champion_promoted!==false||
       !['none','anti-loop-v1'].includes(r.switch_guard)||
       !['neural-all','aq-structure-neural-battle'].includes(r.phase_mode)||
       !Number.isSafeInteger(r.seed)||r.seed<seedStart||
       !Number.isSafeInteger(r.pairs)||r.pairs<1||
       r.seed+r.pairs>seedStart+seedCount||
       r.matches!==2*r.pairs||
       !Array.isArray(r.rows)||r.rows.length!==r.matches)
      throw new Error('Malformed original AQ source-bound shard');
    const identity=[r.source_sha,r.engine_sha,r.model_sha256,
      r.aq_base_model_sha256,r.phase_mode,r.switch_guard].join(':');
    if(reference===null)reference=identity;
    else if(reference!==identity)
      throw new Error('Mixed source, model, engine, phase or guard in AQ study');
    let shardWins=0;
    for(const row of r.rows){
      if(!Number.isSafeInteger(row.seed)||row.seed<r.seed||
         row.seed>=r.seed+r.pairs||
         ![0,1].includes(row.candidateSeat)||
         typeof row.won!=='boolean'||
         !Number.isSafeInteger(row.rounds)||row.rounds<4||row.rounds>64||
         !Number.isSafeInteger(row.accepted_decisions)||
         row.accepted_decisions<2*row.rounds||
         !Array.isArray(row.score)||row.score.length!==2||
         !row.score.every(x=>Number.isSafeInteger(x)&&x>=0&&x<=4)||
         row.score[row.won?row.candidateSeat:1-row.candidateSeat]!==4||
         row.score[row.won?1-row.candidateSeat:row.candidateSeat]>=4)
        throw new Error('Unverified 4-win accepted-seat Match row');
      const key=row.seed+':'+row.candidateSeat;
      if(collected.has(key))throw new Error('Duplicate original AQ seed/seat');
      collected.add(key);
      const seats=bySeed.get(row.seed)??new Set();
      seats.add(row.candidateSeat);
      bySeed.set(row.seed,seats);
      shardWins+=Number(row.won);
    }
    if(shardWins!==r.wins||r.win_rate!==shardWins/r.matches)
      throw new Error('Sharded original AQ win counter mismatch');
    wins+=shardWins;matches+=r.matches;
  }
  if(bySeed.size!==seedCount||matches!==2*seedCount||
     [...bySeed.values()].some(s=>s.size!==2)||
     Array.from({length:seedCount},(_,i)=>seedStart+i)
       .some(seed=>!bySeed.has(seed)))
    throw new Error('Missing seed, swapped seat or complete AQ Match shard');
  const ci95=pairedHoeffding95(wins,seedCount);
  const [sourceSha,engineSha,modelSha,opponentSha,phaseMode,guard]=reference.split(':');
  return {schema:'rnad-original-aq-independent-seed-aggregate-v1',
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    source_sha:sourceSha,engine_sha:engineSha,model_sha256:modelSha,
    aq_base_model_sha256:opponentSha,phase_mode:phaseMode,switch_guard:guard,
    seed_start:seedStart,independent_paired_seeds:seedCount,
    completed_matches:matches,wins,win_rate:wins/matches,
    paired_hoeffding_95pct:ci95,
    // This flag is only eligibility for a separate independent ChatGPT
    // review. The app's Champion registry is never written here.
    eligible_for_independent_champion_review:
      matches>=1024&&ci95[0]>.5,
    champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2);
  if(args.length<8||args[0]!=='--seed-start'||
     args[2]!=='--seed-count'||args[4]!=='--out')
    throw new Error('Usage: --seed-start N --seed-count N --out SUMMARY FILE ...');
  const report=aggregateOriginalAQReports(args.slice(6).map(path=>
    JSON.parse(readFileSync(path,'utf8'))),{
    seedStart:Number(args[1]),seedCount:Number(args[3])});
  writeFileSync(args[5],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(report));
}
