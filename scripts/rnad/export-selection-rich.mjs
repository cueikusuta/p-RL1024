// I5 selection A/B: ACCEPTED public/own observations, teacher targets for TRAINING ONLY.
// Stage0 authority commits actions; actor receives only projectRnadObservation.
// No opponent hidden set, teacher breakdown, or utility is part of actor inputs.
import {writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {createPublicAQTeacher,AQ_STRATEGY_SELECTION_TEACHER_ID,AQ_WEIGHTS_SHA256}
  from './aq-teacher-policy.mjs';
import {rankStrategySelections} from '../../src/ai/team-assessment.js';
import {mapPublicAQSelectionScores} from './selection-oracle.mjs';
import {encodeRnadThreePhaseDecisionV5}
  from '../../src/ml/rnad-phase-features-v5.js';
import {playableSets} from '../../src/production-runtime.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};
import provenance from '../../provenance/manifest.json' with {type:'json'};

export const RICH_SCHEMA='rnad-i5-selection-public-rich-and-teacher-targets-v1';
const moveId=x=>typeof x==='string'?x:x?.id;
const uniqueSorted=values=>['<UNK>',...[...new Set(values.filter(x=>
  typeof x==='string'&&x))].sort((a,b)=>a.localeCompare(b,'en'))];
const vocab={
  species:uniqueSorted(playableSets.map(p=>p.speciesId)),
  names:uniqueSorted(playableSets.map(p=>p.speciesNameJa)),
  abilities:uniqueSorted(playableSets.map(p=>p.abilityId)),
  items:uniqueSorted(playableSets.map(p=>p.itemId)),
  moves:uniqueSorted(playableSets.flatMap(p=>(p.moves??[]).map(moveId))),
  types:uniqueSorted(Object.values(data.species??{}).flatMap(p=>p.types??[])),
};
const vocabMaps=Object.fromEntries(Object.entries(vocab).map(([k,v])=>
  [k,new Map(v.map((s,i)=>[s,i]))]));
const vocabHash=createHash('sha256').update(JSON.stringify(vocab)).digest('hex');
export const selectionRichVocab=Object.freeze({
  schema:'rnad-i5-selection-rich-vocabulary-v1',sha256:vocabHash,
  sizes:Object.fromEntries(Object.entries(vocab).map(([k,v])=>[k,v.length]))
});
function index(kind,value){
  if(value==null||value==='')return 0;
  const found=vocabMaps[kind].get(value);
  if(found===undefined)throw new TypeError('Unknown public/own categorical '+kind);
  return found;
}
function finiteStats(s){
  if(!s||!['hp','atk','def','spa','spd','spe'].every(k=>
    Number.isFinite(s[k])&&s[k]>=0))throw new TypeError('Invalid public/own stats');
  return ['hp','atk','def','spa','spd','spe'].map(k=>Math.min(2,s[k]/300));
}
const typeAt=(types,n)=>index('types',types?.[n]);
function encodeOwn(p){
  if(!p||!p.instanceId||!p.speciesId||!Array.isArray(p.moves)||
     p.moves.length>4||p.moves.length<1)
    throw new TypeError('Incomplete own public actor roster');
  const stats=p.stats??p.baseStats,base=p.baseStats??stats;
  const cats=[
    index('species',p.speciesId),index('abilities',p.abilityId),
    index('items',p.itemId),...Array.from({length:4},(_,i)=>
      index('moves',moveId(p.moves[i]))),
    typeAt(p.types,0),typeAt(p.types,1),
  ];
  const numeric=[...finiteStats(stats),...finiteStats(base),
    Math.max(0,Math.min(2,(p.cost??0)/42)),
    Math.max(0,Math.min(1,(p.hp??stats.hp)/Math.max(1,p.maxHP??stats.hp))),
    Number(Boolean(p.mega)),
    Number(Boolean(p.canMega))];
  if(cats.length!==9||numeric.length!==16||numeric.some(x=>!Number.isFinite(x)))
    throw new TypeError('Wrong own selection token width');
  return {cats,numeric};
}
const knownFoes=new Map();
for(const set of playableSets){
  if(!set.mega&&set.speciesNameJa&&!knownFoes.has(set.speciesNameJa)){
    const s=data.species?.[set.speciesId];
    if(s?.baseStats)knownFoes.set(set.speciesNameJa,s);
  }
}
function encodeFoe(name){
  const spec=knownFoes.get(name);
  if(!spec)throw new TypeError('Missing public opponent species catalog profile');
  return {cats:[index('names',name),typeAt(spec.types,0),typeAt(spec.types,1)],
          numeric:finiteStats(spec.baseStats)};
}
export function encodeRichSelectionActor(observation,legalActions){
  if(observation?.phase!=='selection'||!Array.isArray(observation.self?.team)||
     observation.self.team.length!==6||
     !Array.isArray(observation.opponent?.species)||
     observation.opponent.species.length!==6||
     !Array.isArray(legalActions)||legalActions.length!==60)
    throw new TypeError('Expected legal public six-on-six selection');
  const own=observation.self.team.map(encodeOwn);
  const foes=observation.opponent.species.map(encodeFoe);
  const orders=new Map(observation.self.team.map((p,i)=>[p.instanceId,i]));
  const expected=[];
  for(let i=0;i<6;i++)for(let j=0;j<6;j++)for(let k=j+1;k<6;k++)
    if(i!==j&&i!==k)expected.push([i,j,k]);
  const legal=legalActions.map(action=>{
    if(!Array.isArray(action)||action.length!==3||
       action.some(id=>!orders.has(id)))
      throw new TypeError('Selection contains unknown own instance');
    return action.map(id=>orders.get(id));
  });
  if(JSON.stringify(legal)!==JSON.stringify(expected))
    throw new TypeError('Noncanonical authoritative 60-choice selection');
  return {
    schema:'rnad-i5-selection-public-actor-tensors-v1',
    vocab_sha256:vocabHash,
    own_cats:own.map(x=>x.cats),
    own_numeric:own.map(x=>x.numeric),
    foe_cats:foes.map(x=>x.cats),
    foe_numeric:foes.map(x=>x.numeric),
    legal,
  };
}
const canonical=(choice)=>JSON.stringify([choice[0],[...choice.slice(1)].sort()]);
const keys=Array.from({length:6},(_,i)=>i).flatMap(i=>
  Array.from({length:6},(_,j)=>j).flatMap(j=>
    Array.from({length:6},(_,k)=>k).filter(k=>j<k&&i!==j&&i!==k)
      .map(k=>[i,j,k])));
const trioKeys=[...new Set(keys.map(k=>[...k].sort().join(',')))].sort();
function normalizeTeacherRanking(rows,legalActions){
  const mapped=mapPublicAQSelectionScores(rows,legalActions);
  const coreByTrio=new Map(),leadById=new Map();
  const byCanonical=new Map();
  for(const row of rows){
    const canon=canonical(row.selection);
    if(!byCanonical.has(canon))byCanonical.set(canon,row);
    const id=legalActions.findIndex(a=>canonical(a)===canon);
    if(id<0)throw new TypeError('Teacher ranking contains unknown legal action');
    const trio=[...keys[id]].sort().join(',');
    const ownLead=keys[id][0];
    const core=row.breakdown?.core,lead=row.breakdown?.lead;
    if(!Number.isFinite(core)||!Number.isFinite(lead)||
       Math.abs(core+lead-row.score)>1e-7)
      throw new Error('Teacher decomposition does not reconstruct score');
    for(const [map,key,val] of [[coreByTrio,trio,core],[leadById,ownLead,lead]]){
      if(map.has(key)&&Math.abs(map.get(key)-val)>1e-7)
        throw new Error('Teacher component depends on other choice fields');
      map.set(key,val);
    }
  }
  if(coreByTrio.size!==20||leadById.size!==6||byCanonical.size!==60)
    throw new Error('Teacher decomposition coverage mismatch');
  const core=trioKeys.map(key=>coreByTrio.get(key));
  const lead=Array.from({length:6},(_,i)=>leadById.get(i));
  if(core.some(x=>!Number.isFinite(x))||lead.some(x=>!Number.isFinite(x)))
    throw new Error('Incomplete teacher targets');
  for(let i=0;i<60;i++){
    const reconstructed=core[trioKeys.indexOf([...keys[i]].sort().join(','))]+lead[keys[i][0]];
    if(Math.abs(mapped.scores[i]-reconstructed)>1e-7)
      throw new Error('The teacher score is not decomposed into trio + lead');
  }
  return {core,lead,utilities:mapped.scores,choice:mapped.bestIndex};
}
export function createSelectionRichMatch(seed){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff)
    throw new TypeError('Invalid complete Match seed');
  const teacher=createPublicAQTeacher({selectionMode:'strategy-public'});
  const accepted=[];
  const pending=new Map();
  function wrapped(args){
    const proposal=teacher(args);
    if(args.observation.phase==='selection'){
      const {observation,legalActions,seat}=args;
      const actor=encodeRichSelectionActor(observation,legalActions);
      const rank=rankStrategySelections(observation,{catalog:playableSets,data});
      const labels=normalizeTeacherRanking(rank,legalActions);
      const choice=legalActions.findIndex(x=>
        JSON.stringify(x)===JSON.stringify(proposal.action));
      if(choice!==labels.choice||pending.has(seat))
        throw new Error('Public teacher proposal / selected label mismatch');
      const old=encodeRnadThreePhaseDecisionV5(observation,legalActions);
      pending.set(seat,{seat,round:args.round,choice,actor,
        state:old.state,actions:old.actions,
        targets:{core:labels.core,lead:labels.lead,utilities:labels.utilities}});
    }
    return proposal;
  }
  wrapped.accepted=(receipt)=>{
    teacher.accepted(receipt);
    if(receipt.phase==='selection'){
      const item=pending.get(receipt.seat);
      if(!item||receipt.round!==item.round)
        throw new Error('Missing accepted selection receipt');
      pending.delete(receipt.seat);
      accepted.push(item);
    }
  };
  const match=runRnadAcceptanceSmoke({
    seed,matchId:'rnad-stage0-teacher-'+seed,
    experimentId:'RNAD-001-optional-stage0-imitation',
    sourceSha:verifiedRunningSourceSha(),
    engineSnapshotSha:provenance.sourceCommit,
    policyVersion:AQ_STRATEGY_SELECTION_TEACHER_ID,
    policy:wrapped,phasePolicy:wrapped,includeAcceptedHistory:true
  });
  if(!match.completed||match.score?.[match.winner]!==4||
     pending.size||accepted.length!==2*match.rounds||
     !match.trajectories||!match.jointTimeline)
    throw new Error('Nonaccepted selection Match');
  const selectedTrace=match.jointTimeline
    .map(({seat,index})=>match.trajectories[seat].decisions[index])
    .filter(x=>x.phase==='selection');
  if(accepted.length!==selectedTrace.length)throw new Error('Selection trace cardinality');
  for(let i=0;i<accepted.length;i++){
    const row=selectedTrace[i],e=accepted[i];
    const selected=row.legal_actions.findIndex(x=>
      JSON.stringify(x)===JSON.stringify(row.accepted_action));
    if(row.player_seat!==e.seat||selected!==e.choice)
      throw new Error('Untrusted selection row or order');
  }
  return {schema:RICH_SCHEMA,source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,seed,completed:true,
    score:match.score,winner:match.winner,rounds:match.rounds,
    teacher:AQ_STRATEGY_SELECTION_TEACHER_ID,
    teacher_model_sha256:AQ_WEIGHTS_SHA256,
    vocab:selectionRichVocab,examples:accepted};
}
if(process.argv[1]?.endsWith('/export-selection-rich.mjs')){
  if(process.argv.length!==6||process.argv[2]!=='--seed'||process.argv[4]!=='--out')
    throw new Error('Usage: node export-selection-rich.mjs --seed N --out FILE');
  const result=createSelectionRichMatch(Number(process.argv[3]));
  writeFileSync(process.argv[5],JSON.stringify(result)+'\n',{flag:'wx'});
  console.log(JSON.stringify({seed:result.seed,selection_decisions:result.examples.length}));
}
