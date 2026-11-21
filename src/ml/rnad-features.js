// RNAD-001 deterministic, seat-scoped features. Never accept a raw Match,
// Battle, opponent true set, or privileged training-only state.
import {RNAD_OBSERVATION_SCHEMA} from './legal-action-adapter.js';
import catalog from '../../data/catalog/battle-data.json' with {type:'json'};

export const RNAD_STATE_FEATURES=32;
export const RNAD_ACTION_FEATURES=16;
export const RNAD_FEATURE_SCHEMA='rnad-battle-features-v1';
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const safe=(x,defaultValue=0)=>finite(x)?x:defaultValue;
const clamp=(x,lo=0,hi=1)=>Math.min(hi,Math.max(lo,safe(x)));
const ratio=(x,y)=>clamp(safe(x)/Math.max(1,safe(y,1)));
function hash(name){
  if(typeof name!=='string'||!name)return 0;
  let n=2166136261;
  for(const char of name){n=Math.imul(n^char.charCodeAt(0),16777619)>>>0;}
  return (n%1000000)/1000000;
}
const alive=team=>team.filter(p=>safe(p.hp)>0).length;
const kinds=(arr)=>Array.isArray(arr)?arr.filter(x=>typeof x==='string'):[];
const moveId=move=>typeof move==='string'?move:move?.id;
const stat=(p,n)=>clamp(safe(p?.stats?.[n])/300);
const boost=(p,n)=>clamp((safe(p?.boosts?.[n])+6)/12);
function numeric(result,length){
  if(result.length!==length||result.some(v=>!finite(v)))
    throw new TypeError('Nonfinite or malformed RNAD features');
  return result;
}
export function encodeRnadState(observation){
  if(observation?.schema!==RNAD_OBSERVATION_SCHEMA||
     observation.phase!=='battle'||!Array.isArray(observation.self?.team))
    throw new TypeError('Expected projected battle observation (never raw Match)');
  const own=observation.self;
  const p=own.team[own.active];
  if(!p||!finite(p.hp)||!finite(p.maxHP)||p.maxHP<=0)
    throw new TypeError('Invalid own public/secret state');
  const foe=observation.opponent??{};
  const effects=observation.field?.sideEffects??[[],[]];
  const state=[
    1,clamp(safe(observation.turn)/500),clamp(safe(observation.context?.round)/7),
    ratio(p.hp,p.maxHP),clamp(safe(foe.hpPercent)/100),
    clamp(alive(own.team)/3),clamp(safe(foe.remaining)/3),
    own.canMega?1:0,own.needsSwitch?1:0,own.canSwitch?1:0,
    hash(p.status),hash(foe.status),
    hash(kinds(p.types).join('|')),hash(kinds(foe.types).join('|')),
    hash(p.speciesId),hash(foe.speciesId),
    stat(p,'atk'),stat(p,'def'),stat(p,'spa'),stat(p,'spd'),stat(p,'spe'),
    boost(p,'atk'),boost(p,'def'),boost(p,'spe'),
    hash(observation.field?.weather),hash(observation.field?.terrain),
    observation.field?.trickRoomTurns?1:0,
    clamp(effects[0]?.length/6),clamp(effects[1]?.length/6),
    hash(foe.publicObservation?.usedMoves?.join('|')),
    clamp((p.moves?.length??0)/4),clamp(safe(foe.stages?.spe,0)/6+.5)
  ];
  return numeric(state,RNAD_STATE_FEATURES);
}
export function encodeRnadAction(observation,action){
  const own=observation.self;
  const active=own.team?.[own.active];
  if(!active)throw new TypeError('Invalid action observation');
  const move=action.kind==='move'&&Number.isInteger(action.index)?
    active.moves?.[action.index]:null;
  const def=catalog.moves?.[moveId(move)]??{};
  const switched=action.kind==='switch'?own.team[action.index]:null;
  const features=[
    action.kind==='move'?1:0,
    action.kind==='switch'?1:0,
    action.index==='struggle'?1:0,
    Number.isInteger(action.index)?clamp(action.index/6):0,
    action.mega?1:0,
    Number.isInteger(action.targetIndex)?clamp(action.targetIndex/6):0,
    hash(moveId(move)),
    clamp(safe(def.basePower)/200),
    clamp(safe(def.accuracy,100)/100),
    def.category==='Physical'?1:0,
    def.category==='Special'?1:0,
    def.category==='Status'?1:0,
    clamp(safe(def.priority)/6+.5),
    hash(def.type),
    switched?ratio(switched.hp,switched.maxHP):ratio(safe(move?.pp),safe(move?.maxPP,32)),
    hash(switched?.speciesId)
  ];
  return numeric(features,RNAD_ACTION_FEATURES);
}
export function encodeRnadDecision(observation,legalActions){
  if(!Array.isArray(legalActions)||legalActions.length===0)
    throw new TypeError('Cannot infer without authoritative legal candidates');
  return {
    schema:RNAD_FEATURE_SCHEMA,state:encodeRnadState(observation),
    actions:legalActions.map(action=>encodeRnadAction(observation,action))
  };
}
