// Experimental C: large, seat-visible Battle representation. v5 remains immutable.
// The 96/48 prefix is kept for exact warm-start migration, not as a new claim
// about the amount of useful v5 Battle information.
import {encodeRnadThreePhaseDecisionV5} from './rnad-phase-features-v5.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};
import {effectiveness} from '../battle-mechanics.js';

export const RNAD_THREE_PHASE_FEATURE_SCHEMA_V6='rnad-three-phase-features-v6';
export const RNAD_V6_STATE_FEATURES=384;
export const RNAD_V6_ACTION_FEATURES=160;
const TYPES=Object.freeze(['Normal','Fire','Water','Electric','Grass','Ice',
  'Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost',
  'Dragon','Dark','Steel','Fairy']);
const WEATHER=['sun','rain','sand','snow','hail','harshsun','heavyrain',
  'strongwinds','none','other'];
const TERRAIN=['electric','grassy','misty','psychic','none','other','unknown','active'];
const EFFECTS=['stealthrock','spikes','toxicspikes','stickyweb','reflect',
  'lightscreen','auroraveil','tailwind','safeguard','mist','wish','other'];
const EFFECT_TAGS=['status','boosts','heal','drain','recoil','selfdestruct',
  'multihit','contact','sound','punch','bite','powder','protect','pivot',
  'hazard','screen','weather'];
const norm=x=>typeof x==='string'?x.toLowerCase().replace(/[^a-z0-9]/g,''):'';
const number=x=>typeof x==='number'&&Number.isFinite(x)?x:0;
const bound=(x,max=1)=>Math.max(0,Math.min(1,number(x)/max));
const ratio=(x,y)=>bound(number(x)/Math.max(1,number(y)));
const flags=(value,keys)=>keys.map(k=>norm(value)===norm(k)?1:0);
const typeFlags=x=>TYPES.map(t=>Array.isArray(x)&&x.includes(t)?1:0);
const status=(value)=>{
  const s=norm(value);
  return ['none','brn','par','psn','slp','frz'].map(k=>
    k==='none'?!s?1:0:k==='psn'?['psn','tox','poison','toxic'].includes(s)?1:0:
    k==='brn'?['brn','burn'].includes(s)?1:0:
    k==='par'?['par','paralysis'].includes(s)?1:0:
    k==='slp'?['slp','sleep'].includes(s)?1:0:
    ['frz','freeze'].includes(s)?1:0);
};
function assertWidth(values,n,label){
  if(values.length!==n||values.some(x=>!Number.isFinite(x)))
    throw new TypeError('Invalid public v6 '+label+' features: '+values.length+'/'+n);
  return values;
}
function knownEffect(entry,key){
  const id=norm(entry?.id);
  return key==='other'?!EFFECTS.slice(0,-1).some(e=>id.includes(e))?1:0:
    id.includes(key)?1:0;
}
function field48(obs){
  const field=obs.field??{};
  const w=norm(field.weather),t=norm(field.terrain);
  const weather=WEATHER.map(x=>x==='none'?!w?1:0:
    x==='other'?!WEATHER.slice(0,-2).some(y=>w.includes(y))&&Boolean(w)?1:0:
    w.includes(x)?1:0);
  const terrain=TERRAIN.map(x=>x==='none'?!t?1:0:
    x==='other'?!TERRAIN.slice(0,4).some(y=>t.includes(y))&&Boolean(t)?1:0:
    x==='unknown'?field.terrain===undefined?1:0:
    x==='active'?t?1:0:t.includes(x)?1:0);
  const sides=field.sideEffects??[[],[]];
  const effects=[0,1].flatMap(i=>EFFECTS.map(k=>
    (Array.isArray(sides[i])&&sides[i].some(e=>knownEffect(e,k)))?1:0));
  const r=field.trickRoomTurns??0;
  return assertWidth([...weather,...terrain,...effects,
    r>0?1:0,bound(r,8),
    bound(obs.turn,200),obs.self?.canMega?1:0,
    obs.self?.canSwitch?1:0,bound(obs.opponent?.hpPercent,100)
  ],48,'field');
}
const stats=p=>['atk','def','spa','spd','spe'].map(k=>
  bound(p?.stats?.[k]??p?.baseStats?.[k],350));
function own48(p,index,active){
  if(!p)return new Array(48).fill(0);
  const boosts=p.boosts??p.stages??{};
  const movePP=(p.moves??[]).slice(0,4).map((m,i)=>{
    const ownPP=typeof m==='object'?m?.pp:p.pp?.[i];
    const maximum=typeof m==='object'?m?.maxPP:undefined;
    return maximum?ratio(ownPP,maximum):bound(ownPP,64);
  });
  const fields=[1,p.hp>0?1:0,index===active?1:0,
    ratio(p.hp,p.maxHP),bound(p.maxHP,700),bound(p.cost,40),
    p.mega?1:0,...stats(p),...typeFlags(p.types),...status(p.status),
    ...['atk','def','spa','spd','spe','accuracy','evasion']
      .map(k=>bound(number(boosts[k])+6,12)),
    ...Array.from({length:4},(_,i)=>movePP[i]??0),p.itemId?1:0];
  return assertWidth(fields,48,'own team member');
}
function visibleOpponent(o){
  const p=o?.opponent??{};
  if(!p.speciesId)return null;
  const shown=p.publicObservation??{};
  return {speciesId:p.speciesId,types:p.types??[],hpPercent:p.hpPercent,
    status:p.status,usedMoves:shown.usedMoves??[],
    observedItemIds:shown.observedItemIds??[],
    observedAbilityIds:shown.observedAbilityIds??[],lastSeen:0,
    appearanceId:shown.appearanceId??null};
}
function foe32(p,isActive){
  if(!p)return new Array(32).fill(0);
  const hpKnown=Number.isFinite(p.hpPercent);
  return assertWidth([1,isActive?1:0,
    hpKnown?bound(p.hpPercent,100):0,hpKnown?1:0,
    ...typeFlags(p.types),...status(p.status),
    bound(p.usedMoves?.length,4),p.observedItemIds?.length?1:0,
    p.observedAbilityIds?.length?1:0,bound(p.lastSeen,64)
  ],32,'public opponent');
}
function opponentRoster(obs){
  const current=visibleOpponent(obs);
  const seen=Array.isArray(obs.publicOpponentHistory)?obs.publicOpponentHistory:[];
  if(seen.length>3)throw new TypeError('Oversized accepted public opponent history');
  const roster=current?[current]:[];
  for(const p of seen){
    if(!p||typeof p.speciesId!=='string'||!p.speciesId||
       p.types!==undefined&&(!Array.isArray(p.types)||
         p.types.some(x=>typeof x!=='string')))
      throw new TypeError('Invalid visible opponent history');
    if(roster.some(x=>x.appearanceId&&p.appearanceId?
      x.appearanceId===p.appearanceId:x.speciesId===p.speciesId))continue;
    if(roster.length<3)roster.push(p);
  }
  return roster;
}
const moveId=m=>typeof m==='string'?m:m?.id;
function action112(obs,action){
  const p=obs.self?.team?.[obs.self?.active];
  if(!p)throw new TypeError('Missing own active team member');
  const move=action.kind==='move'&&Number.isInteger(action.index)?
    p.moves?.[action.index]:null;
  const def=data.moves?.[moveId(move)]??{};
  const swapped=action.kind==='switch'?obs.self.team[action.index]:null;
  const attacking=action.kind==='move'&&def.basePower>0;
  const cat=def.category??'';
  const type=def.type??'';
  const effectIds=[norm(moveId(move)),norm(def.volatileStatus),
    norm(def.status),norm(def.weather),norm(def.terrain)];
  const tags=EFFECT_TAGS.map(k=>{
    const f=def.flags??{};
    switch(k){
      case 'status':return def.status||def.volatileStatus?1:0;
      case 'boosts':return def.boosts||def.self?.boosts?1:0;
      case 'heal':return def.heal?1:0;
      case 'drain':return def.drain?1:0;
      case 'recoil':return def.recoil?1:0;
      case 'selfdestruct':return def.selfdestruct?1:0;
      case 'multihit':return def.multihit?1:0;
      case 'contact':case 'sound':case 'punch':case 'bite':case 'powder':
        return f[k]?1:0;
      default:return effectIds.some(s=>s.includes(k))?1:0;
    }
  });
  const mult=attacking&&Array.isArray(obs.opponent?.types)&&obs.opponent.types.length?
    effectiveness(type,obs.opponent.types,data.typeChart):0;
  const pp=typeof move==='object'?move?.pp:p.pp?.[action.index];
  const mx=typeof move==='object'?move?.maxPP:undefined;
  const movePart=[...typeFlags(type?[type]:[]),
    cat==='Physical'?1:0,cat==='Special'?1:0,cat==='Status'?1:0,
    ...tags,
    bound(def.basePower,250),
    def.accuracy===true?1:bound(def.accuracy??100,100),
    bound(number(def.priority)+7,14),
    mx?ratio(pp,mx):bound(pp,64),
    (p.types??[]).includes(type)?1:0,bound(mult,4),
    bound(number(def.basePower)*number(mult),1000),
    def.recoil?1:0,def.heal?1:0,action.mega?1:0];
  assertWidth(movePart,48,'move action');
  const target=swapped??{};
  const boosts=target.boosts??target.stages??{};
  const switchPart=[...typeFlags(target.types),...status(target.status),
    ...stats(target),...['atk','def','spa','spd','spe','accuracy','evasion']
      .map(k=>bound(number(boosts[k])+6,12)),
    ratio(target.hp,target.maxHP),target.mega?1:0,
    target.itemId?1:0,swapped?.hp>0?1:0];
  assertWidth(switchPart,40,'switch action');
  const foeTypes=typeFlags(obs.opponent?.types);
  const context=[...foeTypes,bound(p.stats?.spe??p.baseStats?.spe,350),
    bound(swapped?.stats?.spe??swapped?.baseStats?.spe,350),
    ratio(p.hp,p.maxHP),bound(obs.opponent?.hpPercent,100),
    bound(number(def.priority)+7,14),bound(mult,4)];
  assertWidth(context,24,'action context');
  return assertWidth([...movePart,...switchPart,...context],112,'action extension');
}
export function encodeRnadThreePhaseDecisionV6(observation,legalActions){
  const base=encodeRnadThreePhaseDecisionV5(observation,legalActions);
  if(observation.phase!=='battle')
    throw new TypeError('v6 is a Battle-only feature schema');
  const own=observation.self?.team??[];
  if(!Array.isArray(own)||own.length>3||!own.length)
    throw new TypeError('Expected up to three own Battle Pokémon');
  const foe=opponentRoster(observation);
  const state=assertWidth([...base.state,...field48(observation),
    ...[0,1,2].flatMap(i=>own48(own[i],i,observation.self.active)),
    ...[0,1,2].flatMap(i=>foe32(foe[i],i===0))
  ],RNAD_V6_STATE_FEATURES,'state');
  const actions=legalActions.map((a,i)=>assertWidth(
    [...base.actions[i],...action112(observation,a)],
    RNAD_V6_ACTION_FEATURES,'action'));
  return {schema:RNAD_THREE_PHASE_FEATURE_SCHEMA_V6,state,actions};
}
