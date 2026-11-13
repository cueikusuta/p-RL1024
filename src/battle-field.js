import { requireRule } from './draft.js';
import { grounded, indirectDamage, drainHealth, chainModifiers } from './battle-effects.js';
import { consumeHeldItem } from './battle-mechanics.js';
import { changeBoosts, healHP, scaledStat, boostStage } from './battle-moves.js';
import { setStatus, canSetStatus, statusSpeed } from './battle-status.js';
import { modify, effectiveness } from './battle-mechanics.js';

const HAZARDS={stealthrock:1,spikes:3,toxicspikes:2,stickyweb:1};
export function createField() {
  return {weather:'',weatherTurns:0,terrain:'',terrainTurns:0,trickRoomTurns:0,sides:[{},{}],wishes:[null,null],healingWishes:[null,null],futureMoves:[null,null]};
}
export function updateWeatherSuppression(field,active=[],{includeFainted=false}={}) {field.weatherSuppressed=active.some(p=>(includeFainted||p?.hp>0) && ['cloudnine','airlock'].includes(p?.abilityId));return field.weatherSuppressed;}
export function setWeather(field,id,source) {
  requireRule(['sunnyday','raindance','sandstorm','snowscape'].includes(id),'Unsupported weather');
  if (field.weather===id)return false;
  field.weather=id;
  field.weatherTurns=source.itemId===({sunnyday:'heatrock',raindance:'damprock',sandstorm:'smoothrock',snowscape:'icyrock'}[id]) ? 8 : 5;
  return true;
}
export function electricSeed(pokemon,field) {
  if (!pokemon.hp || pokemon.itemId!=='electricseed' || field.terrain!=='electricterrain')return false;
  // Terrain seeds activate even for ungrounded holders.
  consumeHeldItem(pokemon);changeBoosts(pokemon,{def:1},{effect:'electricseed'});return true;
}
export function psychicSeed(pokemon,field) {
  if (!pokemon.hp || pokemon.itemId!=='psychicseed' || field.terrain!=='psychicterrain')return false;
  consumeHeldItem(pokemon);changeBoosts(pokemon,{spd:1},{effect:'psychicseed'});return true;
}
export function grassySeed(pokemon,field) {
  if (!pokemon.hp || pokemon.itemId!=='grassyseed' || field.terrain!=='grassyterrain')return false;
  consumeHeldItem(pokemon);changeBoosts(pokemon,{def:1},{effect:'grassyseed'});return true;
}
export function setTerrain(field,id,source,active=[]) {
  requireRule(['electricterrain','grassyterrain','mistyterrain','psychicterrain'].includes(id),'Unsupported terrain');
  if (field.terrain===id)return false;
  field.terrain=id;field.terrainTurns=source.itemId==='terrainextender' ? 8 : 5;
  for(const pokemon of active){electricSeed(pokemon,field);grassySeed(pokemon,field);psychicSeed(pokemon,field);updateMimicry(pokemon,field.terrain);}
  return true;
}
export function updateMimicry(pokemon,terrain) {
  if(!pokemon || pokemon.abilityId!=='mimicry')return false;
  const type={electricterrain:'Electric',grassyterrain:'Grass',mistyterrain:'Fairy',psychicterrain:'Psychic'}[terrain];
  const baseTypes=pokemon.transformed ? pokemon.transformOriginal?.originalTypes : pokemon.originalTypes;
  const types=type?[type]:[...(baseTypes??pokemon.types)];
  if(pokemon.types?.join()===types.join())return false;
  pokemon.types=types;return true;
}
export function toggleTrickRoom(field) {
  field.trickRoomTurns=field.trickRoomTurns ? 0 : 5;return true;
}
export function fieldSpeed(pokemon,field,seat=null) {
  let speed=scaledStat(pokemon.stats.spe,boostStage(pokemon,'spe'));
  const modifiers=[];
  if (!field.weatherSuppressed && (pokemon.abilityId==='chlorophyll' && field.weather==='sunnyday' || pokemon.abilityId==='sandrush' && field.weather==='sandstorm' || pokemon.abilityId==='swiftswim' && field.weather==='raindance' || pokemon.abilityId==='slushrush' && ['snowscape','hail'].includes(field.weather)))modifiers.push(8192);
  if(pokemon.abilityId==='surgesurfer'&&field.terrain==='electricterrain')modifiers.push(8192);
  if (pokemon.itemId==='choicescarf')modifiers.push(6144);
  if (pokemon.volatiles?.unburden && !pokemon.itemId)modifiers.push(8192);
  if (pokemon.itemId==='ironball')modifiers.push(2048);
  if(seat!==null && field.sides?.[seat]?.tailwind)modifiers.push(8192);
  speed=modify(speed,chainModifiers(modifiers));
  return Math.max(1,statusSpeed(speed,pokemon));
}
export function placeHazard(field,seat,id) {
  requireRule(seat===0||seat===1,'Invalid side');requireRule(Object.hasOwn(HAZARDS,id),'Unknown hazard');
  const side=field.sides[seat];
  if ((side[id]??0)>=HAZARDS[id])return false;
  side[id]=(side[id]??0)+1;return true;
}
export function clearHazards(field,seat) {
  let removed=false;
  for(const id of Object.keys(HAZARDS))if(field.sides[seat][id]){delete field.sides[seat][id];removed=true;}
  return removed;
}
export function clearScreens(field) {
  let removed=false;
  for(const side of field.sides)for(const id of ['reflect','lightscreen','auroraveil'])if(side[id]){delete side[id];removed=true;}
  return removed;
}
export function defogField(field,targetSeat=1) {
  const a=clearHazards(field,0),b=clearHazards(field,1);
  for(const id of ['reflect','lightscreen','auroraveil'])delete field.sides[targetSeat]?.[id];
  // The pinned callback clears terrain/screens but only hazards (or its
  // separate evasion drop) contribute to the move's returned success value.
  field.terrain='';field.terrainTurns=0;return a||b;
}
export function rapidSpinField(field,seat,pokemon) {
  clearHazards(field,seat);
  delete pokemon.volatiles.leechseed;delete pokemon.volatiles.partiallytrapped;
}
export function entryHazards(field,seat,pokemon,foe,chart,rng,onDamage=()=>{},onStatus=()=>{}) {
  const side=field.sides[seat];
  // Retain placement order; independent side conditions are processed in that order.
  for(const [id,layers]of Object.entries(side)) {
    if (!pokemon.hp)break;
    if (id==='stealthrock') {
      const lost=indirectDamage(pokemon,pokemon.maxHP*effectiveness('Rock',pokemon.types,chart)/8);
      onDamage({id,damage:lost,target:pokemon});
    }
    if (!grounded(pokemon))continue;
    if (id==='spikes')onDamage({id,damage:indirectDamage(pokemon,pokemon.maxHP*[0,3,4,6][layers]/24),target:pokemon});
    if (id==='toxicspikes') {
      if (pokemon.types.includes('Poison'))delete side.toxicspikes;
      else if(!pokemon.types.includes('Steel')) {
        const status=layers===2?'tox':'psn';
        // Expose a cause only when this hazard actually set a lasting status.
        // A Lum Berry can cure it immediately during setStatus.
        if(setStatus(pokemon,status,rng,{chart,source:foe,sourceEffectId:'toxicspikes'}) && pokemon.status===status)
          onStatus({id:'toxicspikes',target:pokemon,status});
      }
    }
    if (id==='stickyweb')changeBoosts(pokemon,{spe:-1},{source:foe,effect:'stickyweb'});
  }
}
export function entryWeather(field,pokemon,active=[],{allowPendingFaint=false}={}) {
  if (!pokemon.hp&&!allowPendingFaint)return false;
  const weather={drizzle:'raindance',drought:'sunnyday',orichalcumpulse:'sunnyday',sandstream:'sandstorm',snowwarning:'snowscape'}[pokemon.abilityId];
  if(weather)return setWeather(field,weather,pokemon);
  const terrain={electricsurge:'electricterrain',hadronengine:'electricterrain',grassysurge:'grassyterrain',psychicsurge:'psychicterrain',mistysurge:'mistyterrain'}[pokemon.abilityId];
  return terrain ? setTerrain(field,terrain,pokemon,active) : false;
}
export function residualWeather(field,pokemon,chart) {
  if(field.weatherSuppressed)return 0;
  if (field.weather!=='sandstorm' || !pokemon.hp || ['sandforce','sandrush'].includes(pokemon.abilityId))return 0;
  if (pokemon.types.some(type=>chart[type.toLowerCase()]?.sandstorm===3))return 0;
  return indirectDamage(pokemon,pokemon.maxHP/16);
}
export function weatherUpkeep(field) {
  if (field.weatherTurns && --field.weatherTurns===0)field.weather='';
}
export function lateFieldUpkeep(field) {
  if (field.trickRoomTurns)--field.trickRoomTurns;
  if (field.terrainTurns && --field.terrainTurns===0)field.terrain='';
  for(const side of field.sides??[])for(const id of ['tailwind','reflect','lightscreen','auroraveil']) {
    const condition=side[id];
    if(condition && --condition.duration<=0)delete side[id];
  }
}
export function setSideCondition(field,seat,id,source) {
  requireRule(['tailwind','reflect','lightscreen','auroraveil'].includes(id),'Unsupported side condition');
  const side=field.sides[seat];
  if(side[id] || id==='auroraveil' && (field.weatherSuppressed || !['snowscape','hail'].includes(field.weather)))return false;
  side[id]={duration:source.itemId==='lightclay' && id!=='tailwind'?8:id==='tailwind'?4:5};
  return true;
}
export function perishSongTurn(pokemon) {
  const state=pokemon.volatiles?.perishsong;
  if(!state || !pokemon.hp)return 0;
  state.duration--;
  if(state.duration<=0) {
    const damage=pokemon.hp;pokemon.hp=0;delete pokemon.volatiles.perishsong;return damage;
  }
  return 0;
}
export function useWish(field,seat,pokemon,turn) {
  if (field.wishes[seat])return false;
  field.wishes[seat]={hp:pokemon.maxHP/2,turn:turn+1};return true;
}
export function residualWish(field,seat,pokemon,turn) {
  const wish=field.wishes[seat];if(!wish || wish.healing || wish.turn>turn)return 0;
  field.wishes[seat]=null;return healHP(pokemon,wish.hp);
}
export function useLeechSeed(pokemon,sourceSeat) {
  if (!pokemon.hp || pokemon.types.includes('Grass') || pokemon.volatiles.leechseed)return false;
  pokemon.volatiles.leechseed={sourceSeat};return true;
}
export function residualLeechSeed(pokemon,active) {
  const seed=pokemon.volatiles.leechseed;if(!seed || !pokemon.hp)return null;
  const receiver=active[seed.sourceSeat];if(!receiver?.hp)return null;
  const damage=indirectDamage(pokemon,pokemon.maxHP/8);
  const healed=drainHealth(receiver,pokemon,damage);
  return {damage,healed,receiver};
}
export function useYawn(pokemon,turn,field,chart,{ignoreAbility=false}={}) {
  if(!ignoreAbility && pokemon.abilityId==='flowerveil' && pokemon.types.includes('Grass'))return false;
  if (pokemon.volatiles.yawn || !canSetStatus(pokemon,'slp',{chart,ignoreAbility,sunny:!field.weatherSuppressed&&field.weather==='sunnyday',electricTerrain:field.terrain==='electricterrain',mistyTerrain:field.terrain==='mistyterrain',grounded:grounded(pokemon)}))return false;
  pokemon.volatiles.yawn={turn:turn+1};return true;
}
export function residualYawn(pokemon,turn,field,chart,rng) {
  if (!pokemon.volatiles.yawn || pokemon.volatiles.yawn.turn>turn)return false;
  delete pokemon.volatiles.yawn;
  return setStatus(pokemon,'slp',rng,{chart,sunny:!field.weatherSuppressed&&field.weather==='sunnyday',
    electricTerrain:field.terrain==='electricterrain',mistyTerrain:field.terrain==='mistyterrain',grounded:grounded(pokemon)});
}
