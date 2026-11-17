// Fixed-N, pre-registered AQ-Base draft confirmation protocol.
// A seed means a two-seat block; the four complete Matches per seed
// (baseline seat0/1, candidate seat0/1) are not independent observations.
// Shard output is resumable; only a complete, verified set can be evaluated.
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';

export const CONFIRM_PLAN_SCHEMA='rnad-i5-fixed-n-aq-confirmation-v1';
export const CONFIRM_SHARD_SCHEMA='rnad-i5-fixed-n-aq-shard-v1';
export const CONFIRM_AGGREGATE_SCHEMA='rnad-i5-fixed-n-aq-aggregate-v1';
const HEX64=/^[a-f0-9]{64}$/;
const HEX40=/^[a-f0-9]{40}$/;
const MODES=new Set(['draft-deepsets-soft','all-v3']);

export function sha256File(path){
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}
export function validateConfirmationPlan(plan){
  if(!plan||typeof plan!=='object'||Array.isArray(plan)||
     plan.schema!==CONFIRM_PLAN_SCHEMA||
     typeof plan.id!=='string'||!/^I5-[A-Z0-9-]{6,64}$/.test(plan.id)||
     plan.benchmark!=='AQ-Base-001 exact integrated strategy-match'||
     plan.candidate_mode!=='draft-deepsets-soft'||
     plan.baseline_mode!=='all-v3'||
     plan.model_run!==43986137367||
     plan.model_artifact!=='deepsets-soft-medium.json'||
     typeof plan.seed_namespace!=='string'||
     !/^I5-final-2026-[a-z0-9-]{10,80}$/.test(plan.seed_namespace)||
     !Number.isSafeInteger(plan.seed_count)||plan.seed_count<2048||
     plan.seed_count>8192||
     !Number.isSafeInteger(plan.shard_size)||
     ![32,64].includes(plan.shard_size)||
     plan.seed_count%plan.shard_size!==0||
     plan.alpha!==0.05||
     plan.min_delta!==0.05||
     plan.stat_method!=='paired-seed-empirical-bernstein-bonferroni-v1'||
     plan.stop_rule!=='fixed-n-no-early-look'||
     plan.candidate_training_matches!==16||
     !HEX64.test(plan.candidate_sha256??'')||
     !HEX64.test(plan.v3_sha256??'')||
     !HEX64.test(plan.v4_sha256??'')||
     !HEX64.test(plan.aq_sha256??'')||
     !HEX40.test(plan.engine_sha??''))
    throw new TypeError('Invalid immutable fixed-N candidate/AQ confirmation protocol');
  return plan;
}
export function readConfirmationPlan(path){
  const bytes=readFileSync(path);
  const plan=validateConfirmationPlan(JSON.parse(bytes.toString('utf8')));
  return {plan,plan_sha256:createHash('sha256').update(bytes).digest('hex')};
}
export function seedForOrdinal(plan,ordinal){
  validateConfirmationPlan(plan);
  if(!Number.isSafeInteger(ordinal)||ordinal<0||ordinal>=plan.seed_count)
    throw new RangeError('Invalid deterministic held-out seed index');
  const bytes=createHash('sha256').update(plan.seed_namespace+'|'+ordinal).digest();
  const seed=bytes.readUInt32BE(0);
  // Earlier pilot and teacher samples must never enter the final confirmation.
  if(seed===0||(seed>=2440000&&seed<=2450100))
    throw new Error('Held-out seed collided with known exploration/teacher set');
  return seed;
}
export function plannedSeeds(plan){
  const seeds=Array.from({length:plan.seed_count},(_,i)=>seedForOrdinal(plan,i));
  if(new Set(seeds).size!==seeds.length)
    throw new Error('Collision between independent held-out seed IDs');
  return seeds;
}
export function plannedShard(plan,index){
  const count=plan.seed_count/plan.shard_size;
  if(!Number.isSafeInteger(index)||index<0||index>=count)
    throw new RangeError('Invalid fixed-N shard index');
  const seeds=plannedSeeds(plan);
  const start=index*plan.shard_size;
  return {index,start,count:plan.shard_size,
    seeds:seeds.slice(start,start+plan.shard_size)};
}
function same(a,b){return JSON.stringify(a)===JSON.stringify(b)}
function hex(x,n){return typeof x==='string'&&
  (n===64?HEX64:HEX40).test(x)}

function validateAuthorityMatches(report,mode,seeds,plan,sourceSha,expectedDraftSha){
  if(!report||report.schema!=='rnad-i5-deepsets-draft-original-aq-paired-screen-v1'||
     report.benchmark!==plan.benchmark||report.mode!==mode||
     report.seed!==null||!same(report.explicit_seeds,seeds)||
     report.pairs!==seeds.length||report.matches!==2*seeds.length||
     !hex(report.source_sha,40)||report.source_sha!==sourceSha||
     report.engine_sha!==plan.engine_sha||
     report.aq_base_model_sha256!==plan.aq_sha256||
     report.v3_model_sha256!==plan.v3_sha256||
     report.v4_model_sha256!==plan.v4_sha256||
     report.draft_model_sha256!==expectedDraftSha||
     report.strength_established!==false||
     report.champion_promoted!==false||
     !Array.isArray(report.rows)||report.rows.length!==2*seeds.length||
     !Array.isArray(report.accepted_draft_examples)||
     report.accepted_draft_examples.length!==2*seeds.length)
    throw new TypeError('Invalid or unfrozen original AQ authority report');
  const expected=new Set(seeds.flatMap(s=>[s+':0',s+':1']));
  const seen=new Set();
  let wins=0;
  for(let i=0;i<report.rows.length;i++){
    const row=report.rows[i],trace=report.accepted_draft_examples[i];
    const key=row.seed+':'+row.seat;
    if(!expected.has(key)||seen.has(key)||
       ![0,1].includes(row.seat)||typeof row.won!=='boolean'||
       !Number.isSafeInteger(row.rounds)||row.rounds<4||
       !Number.isSafeInteger(row.accepted_decisions)||row.accepted_decisions<2*row.rounds||
       !Array.isArray(row.score)||row.score.length!==2||
       row.score.some(v=>!Number.isSafeInteger(v)||v<0||v>4)||
       row.score[row.won?row.seat:1-row.seat]!==4||
       row.score[row.won?1-row.seat:row.seat]===4||
       trace.seed!==row.seed||trace.seat!==row.seat||
       !same(trace.score,row.score)||
       trace.winner!==(row.won?row.seat:1-row.seat)||
       !Array.isArray(trace.picks)||trace.picks.length!==row.rounds)
      throw new Error('Missing, duplicate, incomplete or mismatched AQ seed/seat');
    for(let round=0;round<trace.picks.length;round++){
      const pick=trace.picks[round];
      if(pick.round!==round+1||!Number.isSafeInteger(pick.credits)||
         pick.credits<0||!Number.isSafeInteger(pick.accepted_spent)||
         pick.accepted_spent<0||pick.accepted_spent>pick.credits||
         !Number.isSafeInteger(pick.legal_candidates)||pick.legal_candidates<1||
         !Array.isArray(pick.accepted)||pick.accepted.length!==6||
         pick.accepted.some(s=>typeof s!=='string'||s.length===0)||
         !Array.isArray(pick.carried)||!Array.isArray(pick.offered))
        throw new Error('Malformed authority-accepted six-Pokemon draft evidence');
    }
    seen.add(key);
    wins+=Number(row.won);
  }
  if(seen.size!==expected.size||report.wins!==wins||
     report.win_rate!==wins/(2*seeds.length))
    throw new Error('Invalid frozen paired AQ winner aggregation');
  return new Map(report.rows.map(r=>[r.seed+':'+r.seat,Number(r.won)]));
}
export function validateConfirmationShard(shard,plan,planHash,sourceSha){
  validateConfirmationPlan(plan);
  if(!HEX64.test(planHash??'')||!HEX40.test(sourceSha??'')||
     !shard||shard.schema!==CONFIRM_SHARD_SCHEMA||
     shard.plan_sha256!==planHash||shard.source_sha!==sourceSha||
     !Number.isSafeInteger(shard.shard_index))
    throw new TypeError('Wrong evaluation plan/source/shard protocol');
  const spec=plannedShard(plan,shard.shard_index);
  if(shard.start!==spec.start||!same(shard.seeds,spec.seeds)||
     shard.match_count!==4*spec.seeds.length||
     !shard.baseline||!shard.candidate)
    throw new Error('Duplicate/absent/out-of-range confirmation shard IDs');
  const a=validateAuthorityMatches(shard.baseline,'all-v3',
    spec.seeds,plan,sourceSha,plan.candidate_sha256);
  const b=validateAuthorityMatches(shard.candidate,'draft-deepsets-soft',
    spec.seeds,plan,sourceSha,plan.candidate_sha256);
  if(a.size!==b.size||[...a.keys()].some(k=>!b.has(k)))
    throw new Error('Incomplete two-arm matched AQ seat block');
  return {a,b,spec};
}

// Distribution-free empirical Bernstein fixed-sample CI for iid seed blocks.
// Vn is the unbiased sample variance. This is NOT a sequential/optional
// stopping confidence sequence; no intermediate inference is permitted.
export function empiricalBernsteinFixed95(values,min,max,alpha){
  if(!Array.isArray(values)||values.length<2||
     ![min,max,alpha].every(Number.isFinite)||
     min>=max||alpha<=0||alpha>=1||
     values.some(x=>!Number.isFinite(x)||x<min-1e-12||x>max+1e-12))
    throw new TypeError('Invalid independent fixed-N bounded seed observations');
  const n=values.length,mean=values.reduce((s,x)=>s+x,0)/n;
  const variance=values.reduce((s,x)=>s+(x-mean)**2,0)/(n-1);
  const l=Math.log(2/alpha);
  const width=Math.sqrt(2*variance*l/n)+7*(max-min)*l/(3*(n-1));
  return {n,mean,sample_variance:variance,alpha,
    lower:Math.max(min,mean-width),upper:Math.min(max,mean+width),
    half_width:width,
    method:'fixed-n-empirical-bernstein-sample-variance-bounded'};
}
export function aggregateConfirmationShards(shards,plan,planHash,sourceSha){
  validateConfirmationPlan(plan);
  if(!Array.isArray(shards)||shards.length!==plan.seed_count/plan.shard_size)
    throw new Error('Incomplete frozen fixed-N shard set; no strength verdict');
  const seen=new Set(),pairs=new Map();
  for(const shard of shards){
    const {a,b,spec}=validateConfirmationShard(shard,plan,planHash,sourceSha);
    if(seen.has(spec.index))throw new Error('Duplicate shard index after retry');
    seen.add(spec.index);
    for(const seed of spec.seeds){
      const key0=seed+':0',key1=seed+':1';
      if(pairs.has(seed))throw new Error('Overlapping independent AQ seed');
      pairs.set(seed,{
        control:[a.get(key0),a.get(key1)],
        candidate:[b.get(key0),b.get(key1)]});
    }
  }
  if(seen.size!==plan.seed_count/plan.shard_size||
     pairs.size!==plan.seed_count)
    throw new Error('Missing independent AQ seed/seat blocks');
  const seeds=plannedSeeds(plan),bySeed=seeds.map(seed=>{
    const p=pairs.get(seed);
    if(!p||p.control.some(x=>![0,1].includes(x))||
       p.candidate.some(x=>![0,1].includes(x)))
      throw new Error('Missing paired evaluation outcomes');
    const controlWinRate=(p.control[0]+p.control[1])/2;
    const candidateWinRate=(p.candidate[0]+p.candidate[1])/2;
    return {seed,control_seat0:p.control[0],
      control_seat1:p.control[1],candidate_seat0:p.candidate[0],
      candidate_seat1:p.candidate[1],
      paired_delta:candidateWinRate-controlWinRate,
      aq_win_rate:candidateWinRate};
  });
  // Simultaneous two-sided >=95% over both hypotheses: four tails
  // (paired upper/lower and absolute upper/lower), Bonferroni alpha/4.
  const paired=empiricalBernsteinFixed95(
    bySeed.map(x=>x.paired_delta),-1,1,plan.alpha/4);
  const absolute=empiricalBernsteinFixed95(
    bySeed.map(x=>x.aq_win_rate),0,1,plan.alpha/4);
  const eligible=paired.lower>0&&paired.mean>=plan.min_delta&&
    absolute.lower>.5;
  return {schema:CONFIRM_AGGREGATE_SCHEMA,plan_sha256:planHash,
    source_sha:sourceSha,plan_id:plan.id,
    independent_seed_blocks:seeds.length,
    complete_four_win_matches:4*seeds.length,
    baseline_matches:2*seeds.length,
    candidate_matches:2*seeds.length,
    baseline_wins:bySeed.reduce((s,x)=>s+x.control_seat0+x.control_seat1,0),
    candidate_wins:bySeed.reduce((s,x)=>s+x.candidate_seat0+x.candidate_seat1,0),
    paired,aq_absolute:absolute,
    min_practical_delta:plan.min_delta,
    pass_statistical_review_gate:eligible,
    eligible_for_independent_review:eligible,
    champion_promoted:false,
    verdict:eligible?'evidence-passes-review-required':'not-proven-at-fixed-n',
    by_seed:bySeed};
}
