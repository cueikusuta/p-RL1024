// Bounded original AQ-Base complete 4-win Match screen of an opt-in
// research Deep Sets draft actor. Keeps v3 selection/battle fixed and
// checks matched seed+both-seat outcomes against frozen all-v3/draft-v4.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256 as AQ_SHA}
  from '../../models/aq-base.mjs';
import {createV4CompositeAQSession}
  from './evaluate-v4-composite-vs-aq.mjs';
import {DeepSetsDraftPolicy} from './draft-deepsets-infer.mjs';
import {projectRnadObservation,rnadLegalActions}
  from '../../src/ml/legal-action-adapter.js';
import {scoreStrategyDraftRoster} from '../../src/ai/team-assessment.js';
import {summarizeAcceptedDraftPick} from './draft-pick-review.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};
import data from '../../data/catalog/battle-data.json' with {type:'json'};

export const DRAFT_AQ_SCREEN_SCHEMA='rnad-i5-deepsets-draft-original-aq-paired-screen-v1';
const MODE=['all-v3','draft-v4','draft-deepsets-soft'];
const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function wrapForReview(actor,seat,record){
  let pending=null;
  return {
    select(view){
      if(pending)throw new Error('Unaccepted draft review pending');
      const choice=actor.select(view);
      if(record){
        const obs=projectRnadObservation(view,{seat});
        if(obs.phase==='draft'){
          const legal=rnadLegalActions(view);
          const index=legal.findIndex(x=>eq(x,choice));
          if(index<0)throw new Error('Unaccepted draft choice not legal');
          pending={obs,legal,index};
        }
      }
      return choice;
    },
    commit(){
      actor.commit();
      if(!pending)return;
      const {obs,legal,index}=pending;
      const pool=[...(obs.self.carry??[]),...(obs.self.fresh??[])];
      const byId=new Map(pool.map(p=>[p.instanceId,p]));
      const util=legal.map(a=>scoreStrategyDraftRoster(
        a.map(id=>byId.get(id)),{
          data,credits:obs.self.credits,carry:obs.self.carry??[]}));
      const review=summarizeAcceptedDraftPick(obs,legal,index,util,data,false);
      record.push({
        round:record.length+1,credits:review.credits,
        legal_candidates:legal.length,
        carried:review.roster.filter(x=>x.is_carry).map(x=>x.species),
        offered:review.roster.filter(x=>!x.is_carry)
          .map(x=>({name:x.species,cost:x.credit_cost})),
        accepted:review.accepted_roster_codes
          .map(i=>review.roster[i].species),
        accepted_spent:review.accepted_spent,
        aq_public_rank:review.teacher_rank,
        aq_public_regret:review.teacher_best_utility-review.accepted_utility,
        aq_public_top5:review.teacher_top10.slice(0,5).map(x=>({
          pokemon:review.candidate_codes[x.candidate_index]
            .map(i=>review.roster[i].species),score:x.utility})),
      });
      pending=null;
    },
    stats:()=>actor.stats(),
    finish:reward=>{
      if(pending)throw new Error('Uncommitted draft review at finish');
      return actor.finish(reward);
    }
  };
}
export function evaluateDraftDeepSetsVsAQ({v3Path,v4Path,draftPath,
  expectedDraftSha,seed=2444001,pairs=4,mode='draft-deepsets-soft'}={}){
  if(typeof v3Path!=='string'||!v3Path||
     typeof v4Path!=='string'||!v4Path||
     typeof draftPath!=='string'||!draftPath||
     !Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
     !Number.isSafeInteger(pairs)||pairs<1||pairs>32||
     seed+pairs-1>0xffffffff||!MODE.includes(mode))
    throw new TypeError('Invalid bounded Deep Sets draft AQ screen');
  const freeze=new DeepSetsDraftPolicy(draftPath,{sha256:expectedDraftSha});
  const aq=createOpponent();
  if(aq.modelSha256!==AQ_SHA)
    throw new Error('Untrusted original frozen AQ identity');
  const rows=[],traces=[];let wins=0;
  for(let i=0;i<pairs;i++)for(const seat of [0,1]){
    const actor=createV4CompositeAQSession({
      v4Path,v3Path,seat,
      mode:mode==='draft-v4'?'draft-v4':'all-v3',
      ...(mode==='draft-deepsets-soft'?{
        draftPolicy:new DeepSetsDraftPolicy(draftPath,{
          sha256:freeze.modelSha256})}:{})
    });
    const picked=i===0&&seat===0?[]:null;
    const session=wrapForReview(actor,seat,picked);
    const result=runStrategyMatch({
      seed:seed+i,
      policies:[0,1].map(s=>s===seat?'pure-rl-linear':'aq-base'),
      pureRlSessions:[0,1].map(s=>s===seat?session:null),
      pureRlAllPhases:true,strategyMode:'integrated',
      v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq});
    const stats=result.pureRlStats?.[seat];
    if(!result.completed||![0,1].includes(result.winner)||
       result.score?.[result.winner]!==4||!stats||
       stats.pending||stats.updates!==0||stats.train!==false||
       stats.accepted_decisions!==result.actionCounts[seat]+2*result.rounds||
       (picked!==null&&picked.length!==result.rounds))
      throw new Error('Incomplete authority-accepted original AQ draft Match');
    const won=result.winner===seat;
    wins+=Number(won);
    rows.push({seed:seed+i,seat,won,rounds:result.rounds,
      score:result.score,accepted_decisions:stats.accepted_decisions});
    if(picked!==null)traces.push({seed:seed+i,seat,
      score:result.score,winner:result.winner,picks:picked});
  }
  return {schema:DRAFT_AQ_SCREEN_SCHEMA,
    source_sha:verifiedRunningSourceSha(),engine_sha:provenance.sourceCommit,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    aq_base_model_sha256:AQ_SHA,
    draft_model_sha256:freeze.modelSha256,
    draft_training_source_sha:freeze.trainingSourceSha,
    v4_model_sha256:sha(v4Path),v3_model_sha256:sha(v3Path),
    mode,seed,pairs,matches:2*pairs,wins,win_rate:wins/(2*pairs),
    rows,accepted_draft_examples:traces,
    paired_delta:null,significance_established:false,
    strength_established:false,champion_promoted:false};
}
function validate(r){
  if(r?.schema!==DRAFT_AQ_SCREEN_SCHEMA||
     r.benchmark!=='AQ-Base-001 exact integrated strategy-match'||
     !MODE.includes(r.mode)||!Number.isSafeInteger(r.pairs)||r.pairs<1||
     !Number.isSafeInteger(r.seed)||r.seed<1||
     r.matches!==2*r.pairs||!Array.isArray(r.rows)||
     r.rows.length!==r.matches||
     r.wins!==r.rows.filter(x=>x.won===true).length||
     r.win_rate!==r.wins/r.matches||
     r.strength_established!==false||r.champion_promoted!==false)
    throw new TypeError('Invalid AQ Deep Sets paired report');
  const byPair=new Map();
  for(const x of r.rows){
    if(!Number.isSafeInteger(x.seed)||x.seed<r.seed||
       x.seed>=r.seed+r.pairs||![0,1].includes(x.seat)||
       typeof x.won!=='boolean'||!Array.isArray(x.score)||
       x.score[x.won?x.seat:1-x.seat]!==4||
       !Number.isSafeInteger(x.rounds)||x.rounds<4||
       !Number.isSafeInteger(x.accepted_decisions)||
       x.accepted_decisions<2*x.rounds)
      throw new TypeError('Invalid real four-win Match');
    const key=x.seed+':'+x.seat;
    if(byPair.has(key))throw new Error('Duplicate AQ seed/seat');
    byPair.set(key,Number(x.won));
  }
  return byPair;
}
export function compareDeepSetsDraftAQ(baseline,candidate){
  const a=validate(baseline),b=validate(candidate);
  if(baseline.mode!=='all-v3'&&baseline.mode!=='draft-v4')
    throw new Error('Baseline requires frozen v3 or v4 draft');
  if(candidate.mode!=='draft-deepsets-soft'||
     baseline.mode===candidate.mode||
     ['source_sha','engine_sha','aq_base_model_sha256',
      'v3_model_sha256','v4_model_sha256','seed','pairs']
       .some(k=>baseline[k]!==candidate[k])||
     a.size!==b.size||[...a.keys()].some(k=>!b.has(k)))
    throw new Error('Unmatched model, engine, seed or seat for draft A/B');
  const pairs=baseline.pairs;
  const delta=[...a].reduce((s,[k,v])=>s+b.get(k)-v,0)/(2*pairs);
  // Per-seed average of two seat outcomes is in [-1,1], independent units.
  const half=Math.sqrt(2*Math.log(40)/pairs);
  return {schema:'rnad-i5-deepsets-draft-paired-delta-v1',
    candidate_model_sha256:candidate.draft_model_sha256,
    baseline_mode:baseline.mode,candidate_mode:candidate.mode,
    independent_seeds:pairs,matches_per_arm:2*pairs,
    baseline_wins:baseline.wins,candidate_wins:candidate.wins,
    baseline_rate:baseline.win_rate,candidate_rate:candidate.win_rate,
    paired_delta:delta,
    conservative_95pct_delta_bounds:[
      Math.max(-1,delta-half),Math.min(1,delta+half)],
    confidence_method:'Hoeffding-on-paired-independent-seeds',
    strength_established:false,champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const a=process.argv.slice(2),o=new Map(),
    keys=new Set(['--v3','--v4','--draft','--draft-sha','--seed','--pairs','--mode','--out']);
  if(a.length!==16)throw new TypeError('Expected eight AQ draft screen option pairs');
  for(let i=0;i<a.length;i+=2){
    if(!keys.has(a[i])||!a[i+1]||o.has(a[i]))
      throw new TypeError('Invalid AQ draft screen option');
    o.set(a[i],a[i+1]);
  }
  const r=evaluateDraftDeepSetsVsAQ({
    v3Path:o.get('--v3'),v4Path:o.get('--v4'),
    draftPath:o.get('--draft'),expectedDraftSha:o.get('--draft-sha'),
    seed:Number(o.get('--seed')),pairs:Number(o.get('--pairs')),
    mode:o.get('--mode')});
  writeFileSync(o.get('--out'),JSON.stringify(r,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({mode:r.mode,wins:r.wins,matches:r.matches}));
}
