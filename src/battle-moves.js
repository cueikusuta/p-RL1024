import { confirmedBoostMoveEffects, confirmedBoostSecondaryMoves, confirmedBoostItemEffects, confirmedBoostAbilityEffects } from './battle-boost-public-causes.js';
import { requireRule } from './draft.js';
import { modify, consumeHealingBerry, consumeHeldItem } from './battle-mechanics.js';

const boostObservers=new WeakMap();
export function observeBoostEvents(pokemon,observer) {
  if(typeof observer==='function')boostObservers.set(pokemon,observer);
  else boostObservers.delete(pokemon);
}
export const BOOST_STATS = ['atk','def','spa','spd','spe','accuracy','evasion'];
export function boostStage(pokemon, stat) {
  return pokemon.boosts?.[stat] ?? (stat === 'atk' ? pokemon.atkStage : stat === 'def' ? pokemon.defStage : 0) ?? 0;
}
function writeStage(pokemon, stat, value) {
  pokemon.boosts ??= Object.fromEntries(BOOST_STATS.map(id => [id,boostStage(pokemon,id)]));
  pokemon.boosts[stat] = value;
  if (stat === 'atk') pokemon.atkStage = value;
  if (stat === 'def') pokemon.defStage = value;
}
export function changeBoosts(target, changes, {source = target, ignoreAbility = false, effect = '', causeMoveId = null} = {}) {
  requireRule(changes && Object.entries(changes).every(([stat,n]) => BOOST_STATS.includes(stat) && Number.isInteger(n)), 'Invalid boosts');
  if (target.hp <= 0) return {};
  const applied = {};
  // Sticky Web and Octolock are publicly confirmed move effects; carry their
  // causes through existing boost events without adding extra log lines.
  // Secondary stat drops keep `effect: secondary` to preserve blocked-effect privacy.
  // Only a resolved, explicitly approved attack can supply its public move name.
  const secondaryCause=effect==='secondary' && confirmedBoostSecondaryMoves.has(causeMoveId) ? causeMoveId : null;
  const publicMoveCause=confirmedBoostMoveEffects.has(effect)?{causeMoveId:effect}:(secondaryCause?{causeMoveId:secondaryCause}:{});
  const publicItemCause=confirmedBoostItemEffects.has(effect)?{causeItemId:effect}:{};
  const publicAbilityCause=confirmedBoostAbilityEffects.has(effect)?{causeAbilityId:effect}:{};
  for (const [stat, requestedAmount] of Object.entries(changes)) {
    const amount=!ignoreAbility&&target.abilityId==='contrary' ? -requestedAmount : requestedAmount;
    if(requestedAmount<0 && source!==target && !ignoreAbility && effect==='intimidate' && stat==='atk' && target.abilityId==='guarddog') {
      changeBoosts(target,{atk:1},{source:target,ignoreAbility:true,effect:'guarddog'});
      continue;
    }
    if (amount < 0 && source !== target && !ignoreAbility && (
      ['clearbody','whitesmoke'].includes(target.abilityId) || target.abilityId==='flowerveil' && target.types?.includes('Grass') || stat === 'atk' && target.abilityId === 'hypercutter' ||
      stat === 'def' && target.abilityId==='bigpecks' || effect === 'intimidate' && ['innerfocus','scrappy','owntempo','oblivious'].includes(target.abilityId))) {if(effect!=='secondary')boostObservers.get(target)?.({kind:'blocked',stat,...publicMoveCause,...(target.abilityId==='clearbody' && !publicMoveCause.causeMoveId?{causeAbilityId:'clearbody'}:{})});continue;}
    if(amount<0 && source!==target && !ignoreAbility && target.abilityId==='mirrorarmor' && effect!=='mirrorarmor' && boostStage(target,stat)>-6){boostObservers.get(target)?.({kind:source.hp>0?'reflected':'blocked',stat,...publicMoveCause,causeAbilityId:'mirrorarmor'});changeBoosts(source,{[stat]:amount},{source:target,effect:'mirrorarmor'});continue;}
    if(amount<0 && stat==='accuracy' && source!==target && !ignoreAbility && ['keeneye','illuminate'].includes(target.abilityId)){boostObservers.get(target)?.({kind:'blocked',stat,...publicMoveCause});continue;}
    const before = boostStage(target,stat);
    const after = Math.max(-6,Math.min(6,before + amount));
    if (after === before) {if(amount)boostObservers.get(target)?.({kind:'limit',stat,amount,...publicMoveCause,...publicAbilityCause,...publicItemCause});continue;}
    writeStage(target,stat,after); applied[stat] = after - before;
    boostObservers.get(target)?.({stat,amount:after-before,
      causeSource:['intimidate','gooey'].includes(effect)?source:publicAbilityCause.causeAbilityId?target:source,
      ...publicMoveCause,...publicAbilityCause,...publicItemCause,
      // Intimidate is a publicly revealed entry ability. Preserve its confirmed
      // source on the resolved stat change without exposing arbitrary effects.
      ...(effect==='intimidate' && source!==target ? {causeAbilityId:'intimidate',causeSpeciesName:source.speciesName} : {})});
    if (after>before)target.statsRaisedThisTurn=true;
    if (after<before)target.statsLoweredThisTurn=true;
    // Each successful lowered stat independently triggers Defiant/Competitive.
    if (after < before && source !== target) {
      const response = target.abilityId === 'defiant' ? 'atk' : target.abilityId === 'competitive' ? 'spa' : null;
      if (response) { const previous=boostStage(target,response);writeStage(target,response,Math.min(6,previous+2));
        const actual=boostStage(target,response)-previous;if(actual)boostObservers.get(target)?.({stat:response,amount:actual,causeAbilityId:target.abilityId,causeSource:target}); }
    }
  }
  return applied;
}
export function whiteHerb(pokemon) {
  if (pokemon.hp <= 0 || pokemon.itemId !== 'whiteherb') return false;
  const negative = BOOST_STATS.filter(stat => boostStage(pokemon,stat) < 0);
  if (!negative.length) return false;
  for (const stat of negative) writeStage(pokemon,stat,0);
  consumeHeldItem(pokemon,{kind:'statreset'}); return true;
}
export function resetBoosts(pokemon,{notify=false}={}) {
  const changed=BOOST_STATS.some(stat=>boostStage(pokemon,stat)!==0);
  for (const stat of BOOST_STATS) writeStage(pokemon,stat,0);
  if(notify && changed)boostObservers.get(pokemon)?.({kind:'reset'});
}
export function scaledStat(value, stage) {
  requireRule(Number.isInteger(stage) && stage>=-6 && stage<=6 && value>0,'Invalid stat stage');
  return Math.floor(stage>=0 ? value*(2+stage)/2 : value*2/(2-stage));
}

// Declarative move data stays immutable: return a fresh effective definition.
export function effectiveMove(id, definition, actor, foe, {weather='',terrain='',gravity=false,grounded = () => true, hit=1, turn=0, rng=null, effectiveSpeeds=null} = {}) {
  const move = structuredClone(definition);
  move.id = id;
  if(id==='aurawheel')move.type=actor.speciesId==='morpekohangry'?'Dark':'Electric';
  const weightPower = () => {
    requireRule(Number.isFinite(foe.weightkg) && foe.weightkg > 0, 'Missing battle weight');
    return foe.weightkg >= 200 ? 120 : foe.weightkg >= 100 ? 100 : foe.weightkg >= 50 ? 80 : foe.weightkg >= 25 ? 60 : foe.weightkg >= 10 ? 40 : 20;
  };
  if (id === 'grassknot' || id === 'lowkick') move.basePower = weightPower();
  if (id === 'heavyslam') {
    requireRule(actor.weightkg > 0 && foe.weightkg > 0, 'Missing battle weight');
    move.basePower = actor.weightkg >= foe.weightkg*5 ? 120 : actor.weightkg >= foe.weightkg*4 ? 100 : actor.weightkg >= foe.weightkg*3 ? 80 : actor.weightkg >= foe.weightkg*2 ? 60 : 40;
  }
  if (id === 'heatcrash') {
    requireRule(actor.weightkg > 0 && foe.weightkg > 0, 'Missing battle weight');
    move.basePower = actor.weightkg >= foe.weightkg*5 ? 120 : actor.weightkg >= foe.weightkg*4 ? 100 : actor.weightkg >= foe.weightkg*3 ? 80 : actor.weightkg >= foe.weightkg*2 ? 60 : 40;
  }
  if (id === 'eruption' || id === 'waterspout') move.basePower = Math.max(1,Math.floor(move.basePower*actor.hp/actor.maxHP));
  if (id === 'judgment' && actor.abilityId==='multitype')move.type=actor.types[0]??'Normal';
  if (id === 'avalanche' && actor.lastDamagedBy?.turn === turn && actor.lastDamagedBy?.sourceId === foe.instanceId && actor.lastDamagedBy?.damage > 0) move.basePower *= 2;
  if (id === 'barbbarrage' && ['psn','tox'].includes(foe.status)) move.basePower *= 2;
  if (id === 'expandingforce' && terrain === 'psychicterrain' && grounded(actor)) move.basePower = modify(move.basePower,6144);
  if(id==='venoshock' && ['psn','tox'].includes(foe.status))move.basePower*=2;
  if(id==='acrobatics' && !actor.itemId)move.basePower*=2;
  if (id === 'hex' && foe.status) move.basePower *= 2;
  if (id === 'infernalparade' && foe.status) move.basePower *= 2;
  if (foe.volatiles?.minimize && move.flags?.tramplesminimize)move.basePower*=2;
  if (id === 'ragefist') move.basePower = Math.min(350,50 + 50*(actor.timesAttacked ?? 0));
  if (id === 'lastrespects') move.basePower = 50 + 50*(actor.lastRespectsFainted ?? 0);
  if (id === 'storedpower') move.basePower = 20 + 20*BOOST_STATS.reduce((total,stat)=>total+Math.max(0,boostStage(actor,stat)),0);
  if (id === 'gyroball') {
    const userSpeed=effectiveSpeeds?.actor ?? actor.stats.spe;
    const targetSpeed=effectiveSpeeds?.target ?? foe.stats.spe;
    move.basePower=Math.min(150,Math.floor(25*targetSpeed/userSpeed)+1);
  }
  if (id === 'watershuriken' && actor.speciesId === 'greninjaash' && actor.abilityId === 'battlebond' && !actor.transformed) move.basePower += 5;
  if (id === 'mistyexplosion' && terrain === 'mistyterrain' && grounded(actor)) move.basePower = modify(move.basePower,6144);
  if (id === 'gravapple' && gravity) move.basePower = modify(move.basePower,6144);
  if (id === 'stompingtantrum' && actor.moveLastTurnResult === false) move.basePower *= 2;
  if (id === 'risingvoltage' && terrain === 'electricterrain' && grounded(foe)) move.basePower *= 2;
  if (id === 'knockoff' && foe.itemId && foe.itemId!==foe.megaStoneId && (actor.abilityId==='moldbreaker'||foe.abilityId!=='stickyhold')) move.basePower = modify(move.basePower,6144);
  if (id === 'tripleaxel') move.basePower = 20*hit;
  if (id === 'weatherball') {
    const type = {sunnyday:'Fire',raindance:'Water',sandstorm:'Rock',snowscape:'Ice'}[weather];
    if (type) {move.type=type;move.basePower*=2;}
  }
  if(move.ohko)move.accuracy=Math.max(1,Math.min(100,(move.ohko==='Ice'&&!actor.types.includes('Ice')?20:30)+(actor.level??50)-(foe.level??50)));
  if(id==='shellsidearm' && move.category==='Special') {
    const attack=scaledStat(actor.stats.atk,boostStage(actor,'atk')),specialAttack=scaledStat(actor.stats.spa,boostStage(actor,'spa'));
    const defense=scaledStat(foe.stats.def,boostStage(foe,'def')),specialDefense=scaledStat(foe.stats.spd,boostStage(foe,'spd'));
    const physical=Math.floor(Math.floor(Math.floor((2*(actor.level??50)/5+2)*move.basePower*attack)/defense)/50);
    const special=Math.floor(Math.floor(Math.floor((2*(actor.level??50)/5+2)*move.basePower*specialAttack)/specialDefense)/50);
    if(physical>special || physical===special && rng?.integer(2)===0) {
      move.category='Physical';move.flags={...move.flags,contact:1};
    }
  }
  if (id === 'growth' && weather === 'sunnyday') move.boosts = {atk:2,spa:2};
  if (id === 'bodypress') move.overrideOffensiveStat = 'def';
  // Sheer Force's ModifyMove (priority 0) precedes King's Rock (-1).
  // For that ability, add the Rock's secondary after ability processing instead.
  if(actor.itemId==='kingsrock' && actor.abilityId!=='sheerforce' && move.category!=='Status') {
    const effects=move.secondaries ?? (move.secondary ? [move.secondary] : []);
    if(!effects.some(effect=>effect.volatileStatus==='flinch'))move.secondaries=[...effects,{chance:10,volatileStatus:'flinch'}];
  }
  return move;
}
export function fixedMoveDamage(id, actor, foe, turn) {
  if (id === 'superfang' || id === 'ruination') return Math.max(1,Math.floor(foe.hp/2));
  if (id === 'endeavor') return actor.hp<foe.hp ? foe.hp-actor.hp : 0;
  if (id === 'counter' || id === 'mirrorcoat') {
    const hit=actor.lastDamagedBy;
    if(!hit || hit.turn!==turn || hit.category!==(id==='counter'?'Physical':'Special') || !(hit.damage>=0))return 0;
    return Math.max(1,hit.damage*2);
  }
  if(['fissure','guillotine','horndrill','sheercold'].includes(id))return foe.hp>0 ? foe.maxHP : 0;
  if (id === 'metalburst') {
    const hit=actor.lastDamagedBy;
    return hit?.turn === turn ? Math.max(1,Math.floor(hit.damage*1.5)) : 0;
  }
  return null;
}
export function damageStats(move, actor, foe, crit, ignoreAbility=false) {
  const offense = move.overrideOffensiveStat ?? (move.category === 'Physical' ? 'atk' : 'spa');
  const defenseStat = move.overrideDefensiveStat ?? (move.category === 'Physical' ? 'def' : 'spd');
  const offensivePokemon=move.overrideOffensivePokemon==='target' ? foe : actor;
  const a = !ignoreAbility && foe.abilityId==='unaware' && offensivePokemon===actor ? 0 : boostStage(offensivePokemon,offense);
  const d = move.ignoreDefensive || actor.abilityId==='unaware' ? 0 : boostStage(foe,defenseStat);
  const defense=scaledStat(foe.stats[defenseStat],crit && d>0 ? 0 : d);
  let attack=scaledStat(offensivePokemon.stats[offense],crit && a<0 ? 0 : a);
  if(actor.itemId==='lightball' && String(actor.speciesId??'').startsWith('pikachu'))attack*=2;
  return {attack,defense:!ignoreAbility && move.category==='Physical' && foe.abilityId==='furcoat' ? defense*2 : defense};
}

export function healHP(pokemon, amount) {
  if (pokemon.hp <= 0 || pokemon.volatiles?.healblock) return 0;
  const healed = Math.max(0, Math.min(pokemon.maxHP-pokemon.hp,Math.floor(amount)));
  pokemon.hp += healed; return healed;
}
export function bellyDrum(pokemon,{effect=''}={}) {
  if (pokemon.hp <= pokemon.maxHP/2 || boostStage(pokemon,'atk') >= 6) return false;
  pokemon.hp -= Math.floor(pokemon.maxHP/2);
  changeBoosts(pokemon,{atk:12},{effect}); consumeHealingBerry(pokemon); return true;
}
export function painSplit(actor, foe, onHPChange=()=>{}) {
  if (actor.hp <= 0 || foe.hp <= 0) return false;
  const average = Math.max(1,Math.floor((actor.hp+foe.hp)/2));
  const actorBefore=actor.hp,foeBefore=foe.hp;
  actor.hp = Math.min(actor.maxHP,average); foe.hp = Math.min(foe.maxHP,average);
  // Report the actual split before an immediately triggered berry can mask it.
  onHPChange(actor,actor.hp-actorBefore);
  onHPChange(foe,foe.hp-foeBefore);
  consumeHealingBerry(actor);consumeHealingBerry(foe);return true;
}
export function weatherHeal(pokemon, weather) {
  const modifier = weather === 'sunnyday' ? 2732 : ['raindance','sandstorm','snowscape'].includes(weather) ? 1024 : 2048;
  return healHP(pokemon,modify(pokemon.maxHP,modifier));
}
export function makeSubstitute(pokemon) {
  pokemon.volatiles ??= {};
  if (pokemon.volatiles.substitute || pokemon.hp<=pokemon.maxHP/4) return false;
  const amount = Math.floor(pokemon.maxHP/4);
  pokemon.hp -= amount; pokemon.volatiles.substitute={hp:amount};
  delete pokemon.volatiles.partiallytrapped;
  consumeHealingBerry(pokemon);return true;
}
