// Opt-in RNAD-003 semantic matchup features. The earlier v1 schema and
// every original RNAD-001/002 checkpoint remain immutable and readable.
// Numbers encode public type matchup, attack coverage and team overlap,
// never hash-disguised secret opponent sets or unseen move identities.
import {encodeRnadThreePhaseDecision}
  from './rnad-phase-features.js';
import {effectiveness} from '../battle-mechanics.js';
import {publicRosterTypes} from '../ai/selection-basic.js';
import {playableSets} from '../production-runtime.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};

export const RNAD_THREE_PHASE_FEATURE_SCHEMA_V2='rnad-three-phase-features-v2';
const ATTACK_TYPES=Object.freeze(['Normal','Fire','Water','Electric','Grass',
  'Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock',
  'Ghost','Dragon','Dark','Steel','Fairy']);
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const clamp=(x,lo=0,hi=1)=>Math.min(hi,Math.max(lo,finite(x)?x:0));
const average=a=>a.length?a.reduce((sum,x)=>sum+x,0)/a.length:0;
const moves=p=>Array.isArray(p?.moves)?p.moves:[];
const moveId=m=>typeof m==='string'?m:m?.id;
const types=p=>Array.isArray(p?.types)?p.types.filter(x=>typeof x==='string'):[];
const multiplier=(attack,defenses)=>attack&&defenses?.length?
  Math.max(0,Math.min(4,effectiveness(attack,defenses,data.typeChart))):1;
function strength(p,foeTypes){
  let best=0;
  for(const m of moves(p)){
    const def=data.moves?.[moveId(m)];
    if(!def||def.category==='Status'||!(def.basePower>0))continue;
    const accuracy=def.accuracy===true?1:clamp((def.accuracy??100)/100);
    const stab=types(p).includes(def.type)?1.3:1;
    const value=clamp(def.basePower/150)*accuracy*
      stab*multiplier(def.type,foeTypes)/2.8;
    best=Math.max(best,clamp(value));
  }
  return best;
}
const defensive=(p,attackType)=>clamp(1-
  multiplier(attackType,types(p))/4);
function genericPublicTypes(){
  return ATTACK_TYPES.map(type=>[type]);
}
function publicTargetTypes(observation){
  // Match.view reveals only species names, never the opponent's private set.
  const names=observation?.opponent?.species;
  if(observation?.phase==='selection'&&Array.isArray(names)&&names.length){
    const shown=publicRosterTypes(names,playableSets,data);
    const known=shown.filter(row=>row.length>0);
    if(known.length)return known;
  }
  return genericPublicTypes();
}
function meaningfulTeamScores(selected,foeTypes,lookup){
  const rows=selected.map(p=>lookup.get(p.instanceId));
  const pressure=foeTypes.map((_,i)=>Math.max(0,...rows.map(r=>r.attacks[i])));
  const typeSet=new Set(selected.flatMap(types));
  const totalTypes=selected.reduce((n,p)=>n+types(p).length,0);
  const uniqueCoverage=totalTypes?typeSet.size/totalTypes:0;
  const defenseTypes=new Set(foeTypes.flat());
  const defensiveExposure=[...defenseTypes].map(type=>
    Math.max(0,...selected.map(p=>defensive(p,type))));
  return {coverage:average(pressure),worst:Math.min(...pressure),
    lead:average(rows[0].attacks),
    resistance:average(defensiveExposure),uniqueCoverage};
}
function encodeBattleV2(observation,features){
  const p=observation.self?.team?.[observation.self.active];
  const foeTypes=types(observation.opponent);
  if(!p)throw new TypeError('Invalid battle Actor state in v2');
  const old=features.state.slice();
  // Unlike v1, arbitrary species hashes never act as a distance metric.
  old[14]=foeTypes.length?strength(p,foeTypes):0;
  old[15]=foeTypes.length?average(foeTypes.map(t=>
    defensive(p,t))):0;
  old[25]=observation.opponent?.publicObservation?.usedMoves?.length?
    clamp(observation.opponent.publicObservation.usedMoves.length/4):0;
  const actions=features.actions.map((v,i)=>{
    const a=features.legal[i];
    const f=v.slice();
    if(a.kind==='move'&&Number.isInteger(a.index)){
      const id=moveId(p.moves?.[a.index]);
      const move=data.moves?.[id]??{};
      f[6]=types(p).includes(move.type)?1:0;
      f[13]=foeTypes.length?
        clamp(multiplier(move.type,foeTypes)/4):0;
      f[15]=move.category==='Status'?0:strength(p,foeTypes);
    }else if(a.kind==='switch'){
      const target=observation.self.team[a.index];
      f[6]=0;f[13]=foeTypes.length?average(foeTypes.map(t=>
        defensive(target,t))):0;
      f[15]=foeTypes.length?strength(target,foeTypes):0;
    }else{f[6]=0;f[13]=0;f[15]=0;}
    return f;
  });
  return {state:old,actions};
}
export function encodeRnadThreePhaseDecisionV2(observation,legalActions){
  const original=encodeRnadThreePhaseDecision(observation,legalActions);
  const state=original.state.slice(),actions=original.actions.map(x=>x.slice());
  if(observation.phase==='battle'){
    const batt=encodeBattleV2(observation,{state,actions,legal:legalActions});
    return {schema:RNAD_THREE_PHASE_FEATURE_SCHEMA_V2,...batt};
  }
  const roster=observation.phase==='draft'?
    [...(observation.self.carry??[]),...(observation.self.fresh??[])]:
    observation.self.team;
  const byId=new Map(roster.map(p=>[p.instanceId,p]));
  const publicTypes=publicTargetTypes(observation);
  const lookup=new Map(roster.map(p=>[p.instanceId,{
    attacks:publicTypes.map(t=>strength(p,t))}]));
  const pool=meaningfulTeamScores(roster,publicTypes,lookup);
  // Public preview coverage and own candidate-pool matchup statistics.
  state[23]=pool.coverage;
  state[24]=pool.resistance;
  state[25]=observation.phase==='selection'&&
    (observation.opponent?.species?.length??0)>0?1:0;
  state[26]=pool.uniqueCoverage;
  for(let i=0;i<legalActions.length;i++){
    const selection=legalActions[i].map(id=>byId.get(id));
    const scores=meaningfulTeamScores(selection,publicTypes,lookup);
    // Replace constant species-count and arbitrary identity hashes with
    // relative type coverage, public matchup and defensive complement.
    actions[i][5]=scores.uniqueCoverage;
    if(observation.phase==='selection')
      actions[i][13]=scores.resistance;
    actions[i][14]=scores.lead;
    actions[i][15]=scores.coverage;
  }
  const valid=[...state,...actions.flat()].every(finite);
  if(!valid||state.length!==32||actions.some(a=>a.length!==16))
    throw new TypeError('Invalid finite semantic three-phase vector');
  return {schema:RNAD_THREE_PHASE_FEATURE_SCHEMA_V2,state,actions};
}
