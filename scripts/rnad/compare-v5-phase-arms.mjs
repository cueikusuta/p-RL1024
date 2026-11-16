// Fail-closed seed-clustered paired benchmark for a frozen v5 phase swap.
import {V5_AQ_SCHEMA} from './evaluate-v5-composite-vs-aq.mjs';
import {weightsSha256 as AQ_SHA} from '../../models/aq-base.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};
const digest=x=>typeof x==='string'&&/^[0-9a-f]{64}$/.test(x);
const git=x=>typeof x==='string'&&/^[0-9a-f]{40}$/.test(x);
function verify(r){
  if(!r||r.schema!==V5_AQ_SCHEMA||
     r.policy!=='frozen-greedy-v5-v3-anti-loop-v1'||
     r.champion_promoted!==false||r.strength_established!==false||
     !['all-v3','selection-v5','draft-v5','both-v5'].includes(r.mode)||
     !git(r.source_sha)||r.engine_sha!==provenance.sourceCommit||
     r.aq_base_model_sha256!==AQ_SHA||
     !digest(r.v5_model_sha256)||!digest(r.v3_model_sha256)||
     !Number.isSafeInteger(r.seed)||r.seed<1||
     !Number.isSafeInteger(r.pairs)||r.pairs<1||r.pairs>100||
     r.seed+r.pairs-1>0xffffffff||r.matches!==2*r.pairs||
     !Number.isSafeInteger(r.wins)||r.wins<0||r.wins>r.matches||
     r.win_rate!==r.wins/r.matches||
     !Array.isArray(r.rows)||r.rows.length!==r.matches)
    throw new TypeError('Untrusted v5 paired AQ report');
  const seen=new Map();let wins=0;
  for(const x of r.rows){
    if(!Number.isSafeInteger(x.seed)||x.seed<r.seed||
       x.seed>=r.seed+r.pairs||![0,1].includes(x.candidateSeat)||
       typeof x.won!=='boolean'||!Array.isArray(x.score)||
       x.score.length!==2||
       x.score.some(s=>!Number.isInteger(s)||s<0||s>4)||
       x.score[x.won?x.candidateSeat:1-x.candidateSeat]!==4||
       x.score[x.won?1-x.candidateSeat:x.candidateSeat]>=4||
       !Number.isSafeInteger(x.rounds)||x.rounds<4||x.rounds>64||
       !Number.isSafeInteger(x.accepted_decisions)||
       x.accepted_decisions<2*x.rounds)
      throw new TypeError('Non-authoritative AQ outcome row');
    const key=x.seed+':'+x.candidateSeat;
    if(seen.has(key))throw new Error('Duplicate seed or seat');
    seen.set(key,Number(x.won));wins+=Number(x.won);
  }
  if(wins!==r.wins||seen.size!==2*r.pairs||
     Array.from({length:r.pairs},(_,i)=>r.seed+i).some(seed=>
       !seen.has(seed+':0')||!seen.has(seed+':1')))
    throw new Error('Incomplete paired AQ block');
  return seen;
}
export function compareV5PhaseAgainstV3(baseline,candidate,{familySize=1}={}){
  if(!Number.isSafeInteger(familySize)||familySize<1||familySize>64)
    throw new TypeError('Invalid preregistered family size');
  const a=verify(baseline),b=verify(candidate);
  if(baseline.mode!=='all-v3'||candidate.mode==='all-v3'||
     ['source_sha','engine_sha','aq_base_model_sha256',
      'v5_model_sha256','v3_model_sha256','seed','pairs'].some(k=>
       baseline[k]!==candidate[k])||[...a.keys()].some(k=>!b.has(k)))
    throw new Error('Unmatched v5 original AQ paired experiment');
  const n=baseline.pairs;
  const delta=[...a].reduce((s,[k,v])=>s+b.get(k)-v,0)/(2*n);
  const width=Math.sqrt(2*Math.log(40*familySize)/n);
  return {schema:'rnad-i5-v5-isolated-phase-paired-screen-v1',
    source_sha:baseline.source_sha,engine_sha:baseline.engine_sha,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    v3_model_sha256:baseline.v3_model_sha256,
    v5_model_sha256:baseline.v5_model_sha256,
    baseline_wins:baseline.wins,candidate_wins:candidate.wins,
    baseline_mode:'all-v3',candidate_mode:candidate.mode,
    independent_seed_pairs:n,matches_per_arm:2*n,
    delta,family_size:familySize,
    familywise_95pct_delta_bounds:[
      Math.max(-1,delta-width),Math.min(1,delta+width)],
    strength_established:false,champion_promoted:false};
}
