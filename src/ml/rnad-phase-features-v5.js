// v5 opt-in: retain every v4 coordinate, add public six-species
// identity/profile and lead-specific matchup coordinates. No hidden sets.
// 96D state / 48D action; old v2/v3/v4 schemas and weights are immutable.
import {encodeRnadThreePhaseDecisionV4}
  from './rnad-phase-features-v4.js';
import {playableSets} from '../production-runtime.js';
import {effectiveness} from '../battle-mechanics.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};

export const RNAD_THREE_PHASE_FEATURE_SCHEMA_V5='rnad-three-phase-features-v5';
export const RNAD_V5_STATE_FEATURES=96;
export const RNAD_V5_ACTION_FEATURES=48;
const bound=(n,max)=>Number.isFinite(n)?Math.max(0,Math.min(1,n/max)):0;
const safe=x=>typeof x==='string'&&x.length>0&&x.length<=256;
const catalog=new Map();
for(const p of playableSets){
  if(!p.mega&&safe(p.speciesNameJa)&&data.species?.[p.speciesId]?.baseStats&&
     !catalog.has(p.speciesNameJa))
    catalog.set(p.speciesNameJa,data.species[p.speciesId]);
}
function hash16(s){
  let hash=2166136261;
  for(let i=0;i<s.length;i++)hash=Math.imul(hash^s.charCodeAt(i),16777619)>>>0;
  return hash&15;
}
// No continuous species hash: four categorical identity bits plus
// publicly known base stats, deterministic and independent of hidden sets.
export function publicSpeciesProfiles(observation){
  if(observation?.phase!=='selection')return [];
  const names=observation.opponent?.species??[];
  if(!Array.isArray(names)||![0,6].includes(names.length)||
     names.some(x=>!safe(x))||new Set(names).size!==names.length)
    throw new TypeError('Expected 0 or 6 distinct public species names');
  return names.slice().sort((a,b)=>a.localeCompare(b,'ja')).map(name=>{
    const spec=catalog.get(name),s=spec?.baseStats;
    if(!s||!['hp','atk','def','spa','spd','spe'].every(k=>
        Number.isFinite(s[k])&&s[k]>0))
      throw new TypeError('Unknown or incomplete public species stats');
    const code=hash16(name);
    return {name,types:spec.types,stats:s,feature:[
      ...[0,1,2,3].map(bit=>(code>>bit)&1),
      bound(s.spe,255),bound(Math.max(s.atk,s.spa),255),
      bound((s.def+s.spd)/2,255),bound(s.hp,255)
    ]};
  });
}
const moveId=m=>typeof m==='string'?m:m?.id;
const statsSpeed=p=>p?.stats?.spe??p?.baseStats?.spe??0;
function leadComparison(observation,choice,profiles){
  if(observation.phase!=='selection'||profiles.length!==6)
    return new Array(16).fill(0);
  const team=observation.self?.team??[];
  const lead=team.find(p=>p.instanceId===choice[0]);
  if(!lead)throw new TypeError('Missing authority-legal own lead');
  const spd=statsSpeed(lead);
  const speedEdges=profiles.map(p=>bound((spd-p.stats.spe)+255,510));
  const offense=profiles.map(foe=>{
    let best=0;
    for(const m of lead.moves??[]){
      const def=data.moves?.[moveId(m)];
      if(def&&def.category!=='Status'&&def.basePower>0&&
         Array.isArray(foe.types)&&foe.types.length)
        best=Math.max(best,effectiveness(def.type,foe.types,data.typeChart));
    }
    return bound(best,4);
  });
  const chosen=choice.map(id=>team.find(p=>p.instanceId===id));
  if(chosen.some(x=>!x))
    throw new TypeError('Unknown own selection candidate');
  return [...speedEdges,...offense,
    bound(chosen.reduce((s,p)=>s+statsSpeed(p),0)/3,600),
    bound(chosen.filter(p=>p.mega).length,3),
    bound(Math.max(...chosen.map(statsSpeed))-Math.min(...chosen.map(statsSpeed)),600),
    bound(lead?.hp??lead?.stats?.hp??0,600)];
}
export function encodeRnadThreePhaseDecisionV5(observation,legalActions){
  const base=encodeRnadThreePhaseDecisionV4(observation,legalActions);
  const profiles=publicSpeciesProfiles(observation);
  const state=[...base.state];
  for(let i=0;i<6;i++)
    state.push(...(profiles[i]?.feature??new Array(8).fill(0)));
  const actions=legalActions.map((a,i)=>[
    ...base.actions[i],...leadComparison(observation,a,profiles)]);
  if(state.length!==RNAD_V5_STATE_FEATURES||
     actions.some(a=>a.length!==RNAD_V5_ACTION_FEATURES)||
     [...state,...actions.flat()].some(x=>!Number.isFinite(x)))
    throw new TypeError('Invalid finite v5 public species features');
  return {schema:RNAD_THREE_PHASE_FEATURE_SCHEMA_V5,state,actions};
}
