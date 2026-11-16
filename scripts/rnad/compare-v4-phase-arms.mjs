// Direct AQ-Base paired phase-ablation of fixed v4/v3 model bytes.
// The unit is a seed (both seat placements), not an independent row.
// This is a screen for non-inferiority/improvement, never auto promotion.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {COMPOSITE_SCHEMA,COMPOSITE_MODES}
  from './evaluate-v4-composite-vs-aq.mjs';
import {weightsSha256 as AQ_SHA} from '../../models/aq-base.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const exact=(x)=>typeof x==='string'&&/^[0-9a-f]{64}$/.test(x);
const git=(x)=>typeof x==='string'&&/^[0-9a-f]{40}$/.test(x);
const signature=x=>[x.source_sha,x.engine_sha,x.aq_base_model_sha256,
  x.v4_model_sha256,x.v3_model_sha256,x.seed,x.pairs].join(':');

function verify(r){
  if(r?.schema!==COMPOSITE_SCHEMA||r.champion_promoted!==false||
    r.policy!=='frozen-greedy-v4-v3-anti-loop-v1'||
    !Object.hasOwn(COMPOSITE_MODES,r.mode)||
    !git(r.source_sha)||r.engine_sha!==provenance.sourceCommit||
    r.aq_base_model_sha256!==AQ_SHA||
    !exact(r.v4_model_sha256)||!exact(r.v3_model_sha256)||
    !Number.isSafeInteger(r.seed)||r.seed<1||
    !Number.isSafeInteger(r.pairs)||r.pairs<1||r.pairs>100||
    r.seed+r.pairs-1>0xffffffff||
    r.matches!==2*r.pairs||!Array.isArray(r.rows)||
    r.rows.length!==r.matches||!Number.isSafeInteger(r.wins)||
    r.win_rate!==r.wins/r.matches)
    throw new Error('Invalid exact-AQ v4/v3 phase-ablation report');
  const seen=new Map(),count=new Map(),wins=r.rows.filter(x=>x.won).length;
  if(wins!==r.wins)throw new Error('Forged exact AQ win counter');
  for(const x of r.rows){
    if(!Number.isSafeInteger(x.seed)||x.seed<r.seed||
       x.seed>=r.seed+r.pairs||![0,1].includes(x.candidateSeat)||
       typeof x.won!=='boolean'||
       !Array.isArray(x.score)||x.score.length!==2||
       x.score.some(v=>!Number.isInteger(v)||v<0||v>4)||
       x.score[x.won?x.candidateSeat:1-x.candidateSeat]!==4||
       x.score[x.won?1-x.candidateSeat:x.candidateSeat]>=4||
       !Number.isSafeInteger(x.rounds)||x.rounds<4||x.rounds>64||
       !Number.isSafeInteger(x.accepted_decisions)||
       x.accepted_decisions<2*x.rounds)
      throw new Error('Invalid accepted complete AQ match row');
    const key=x.seed+':'+x.candidateSeat;
    if(seen.has(key))throw new Error('Duplicate paired seed/seat');
    seen.set(key,x.won?1:0);
    count.set(x.seed,(count.get(x.seed)||0)+1);
  }
  if(seen.size!==2*r.pairs||count.size!==r.pairs||
     [...count.values()].some(x=>x!==2))
    throw new Error('Unpaired or incomplete seed block');
  return seen;
}

export function compareV4PhaseArms(baseline,candidate){
  const a=verify(baseline),b=verify(candidate);
  if(signature(baseline)!==signature(candidate)||
     baseline.mode!=='all-v3'||candidate.mode==='all-v3'||
     [...a.keys()].some(x=>!b.has(x)))
    throw new Error('Unmatched all-v3 control / candidate phase experiment');
  const n=baseline.pairs;
  const gain=[...a].reduce((s,[key,v])=>s+b.get(key)-v,0)/(2*n);
  const width=Math.sqrt(2*Math.log(40)/n);
  return {schema:'rnad-exact-aq-v4-isolated-phase-paired-gain-v1',
    source_sha:baseline.source_sha,engine_sha:baseline.engine_sha,
    original_aq_sha256:baseline.aq_base_model_sha256,
    v4_model_sha256:baseline.v4_model_sha256,
    v3_model_sha256:baseline.v3_model_sha256,
    baseline_mode:'all-v3',candidate_mode:candidate.mode,
    first_seed:baseline.seed,independent_seeds:n,
    matches_per_arm:2*n,
    baseline_wins:baseline.wins,candidate_wins:candidate.wins,
    baseline_win_rate:baseline.win_rate,
    candidate_win_rate:candidate.win_rate,
    delta:gain,conservative_95pct_delta_bounds:[
      Math.max(-1,gain-width),Math.min(1,gain+width)],
    improved_with_95pct_confidence:gain-width>0,
    degraded_with_95pct_confidence:gain+width<0,
    strength_established:false,champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const [base,candidate,out,...rest]=process.argv.slice(2);
  if(!base||!candidate||!out||rest.length)
    throw new Error('Usage: BASE.json CANDIDATE.json OUT.json');
  const r=compareV4PhaseArms(JSON.parse(readFileSync(base)),
                            JSON.parse(readFileSync(candidate)));
  writeFileSync(out,JSON.stringify(r,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(r));
}
