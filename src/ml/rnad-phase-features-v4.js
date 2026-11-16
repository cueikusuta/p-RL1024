// v4 experiment: 48D state / 32D action, with semantics separated from v3.
// The last 16 action channels describe own-role interactions and PUBLIC
// opponent matchups. Frozen v1-v3 model schemas and weights are unchanged.
import {encodeRnadThreePhaseDecisionV3}
  from './rnad-phase-features-v3.js';
import {publicRosterTypes} from '../ai/selection-basic.js';
import {analyzeSetCapabilities} from '../ai/set-capabilities.js';
import {effectiveness} from '../battle-mechanics.js';
import {playableSets} from '../production-runtime.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};

export const RNAD_THREE_PHASE_FEATURE_SCHEMA_V4='rnad-three-phase-features-v4';
const bound=(x,max=1)=>Number.isFinite(x)?Math.max(0,Math.min(1,x/max)):0;
const mean=xs=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:0;
const ownPool=o=>o.phase==='draft'?
  [...(o.self?.carry??[]),...(o.self?.fresh??[])]:o.self?.team??[];
const typeList=p=>Array.isArray(p?.types)?p.types:[];
const speed=p=>Number(p?.stats?.spe??p?.baseStats?.spe??0);
const has=(p,tag)=>analyzeSetCapabilities(p,data).tags.includes(tag);
const moveId=m=>typeof m==='string'?m:m?.id;
function publicTypes(o){
  const names=o.phase==='selection'?o.opponent?.species:[];
  if(!Array.isArray(names)||names.some(n=>typeof n!=='string'))
    throw new TypeError('Invalid public enemy species preview');
  return names.length?publicRosterTypes(names,playableSets,data).filter(x=>x.length):[];
}
function matchup(p,foeTypes){
  if(!foeTypes.length)return [0,0,0];
  const damages=foeTypes.map(types=>{
    let best=0;
    for(const m of p.moves??[]){
      const def=data.moves?.[moveId(m)];
      if(def&&def.category!=='Status'&&def.basePower>0)
        best=Math.max(best,effectiveness(def.type,types,data.typeChart));
    }
    return bound(best,4);
  });
  const threats=foeTypes.map(types=>
    Math.max(0,...types.map(t=>effectiveness(t,typeList(p),data.typeChart))));
  return [mean(damages),Math.min(...damages),mean(threats.map(v=>bound(v,4)))];
}
function semantics(selected,o,foes){
  const allSpe=selected.map(speed), avg=mean(allSpe);
  const tagged=selected.map(p=>({
    attack:has(p,'attack'),setup:has(p,'setup'),screen:has(p,'screen'),
    pass:has(p,'batonPass'),trick:has(p,'trickRoom'),
  }));
  const count=x=>tagged.filter(t=>t[x]).length;
  const complementary=(a,b)=>tagged.some((left,i)=>left[a]&&
    tagged.some((right,j)=>i!==j&&right[b]));
  const trickSlow=tagged.some((t,i)=>t.trick&&selected.some((p,j)=>
    j!==i&&tagged[j].attack&&speed(p)<speed(selected[i])));
  const lead=o.phase==='selection'?selected[0]:null;
  const leadMatch=lead?matchup(lead,foes):[0,0,0];
  const uniqueTypes=new Set(selected.flatMap(typeList));
  return [
    bound(lead?speed(lead):avg,600),
    bound(Math.min(...allSpe),600),bound(Math.max(...allSpe),600),
    bound(Math.max(...allSpe)-Math.min(...allSpe),600),
    bound(count('attack'),6),bound(count('setup'),6),
    bound(count('screen'),6),bound(count('trick'),6),
    bound(count('pass'),6),complementary('screen','setup')?1:0,
    complementary('pass','attack')?1:0,trickSlow?1:0,
    bound(selected.filter(p=>Boolean(p.mega)).length,6),
    bound(uniqueTypes.size,12),
    o.phase==='selection'?leadMatch[0]:bound(selected.reduce((n,p)=>
      n+(p.cost??0),0),42),
    o.phase==='selection'?leadMatch[1]-leadMatch[2]+1:bound(avg,600)
  ];
}
export function encodeRnadThreePhaseDecisionV4(observation,legalActions){
  const prior=encodeRnadThreePhaseDecisionV3(observation,legalActions);
  if(observation.phase==='battle')
    return {schema:RNAD_THREE_PHASE_FEATURE_SCHEMA_V4,
      state:prior.state,actions:prior.actions};
  const own=ownPool(observation);
  const byId=new Map(own.map(p=>[p.instanceId,p]));
  const foes=publicTypes(observation);
  const state=prior.state.slice();
  // Keep authority-accepted history channels [32..36] unchanged.
  if(observation.phase==='selection'){
    const allTypes=new Set(foes.flat());
    state[45]=bound(allTypes.size,12);
    state[46]=bound(foes.filter(x=>x.length).length,6);
    state[47]=bound(mean(own.map(speed)),600);
  }
  const actions=legalActions.map((choice,i)=>{
    if(!Array.isArray(choice)||choice.length!==(observation.phase==='draft'?6:3))
      throw new TypeError('Invalid phase action length');
    const selected=choice.map(id=>byId.get(id));
    if(selected.some(p=>!p))
      throw new TypeError('Selection includes unknown own Pokémon');
    return [...prior.actions[i].slice(0,16),...semantics(selected,observation,foes)];
  });
  if(state.length!==48||actions.some(a=>a.length!==32||
      a.some(x=>!Number.isFinite(x)))||state.some(x=>!Number.isFinite(x)))
    throw new TypeError('Invalid finite R-NaD v4 semantic vectors');
  return {schema:RNAD_THREE_PHASE_FEATURE_SCHEMA_V4,state,actions};
}
