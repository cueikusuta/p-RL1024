// Read-only, uncalibrated matchup assessment. Opponent sets are catalog
// hypotheses, never actual opposing selections. Battle remains the rule oracle.
import {effectiveness,elementalDamage} from '../battle-mechanics.js';
import {calculateChampionsStats} from '../champions-stats.js';
import {fieldSpeed} from '../battle-field.js';
import {projectCompactOpponentBelief} from './public-belief-adapter.js';
import {megaMatchesObservation} from './opponent-belief.js';

const clamp=n=>Math.max(-1,Math.min(1,n));
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const stage=n=>n>=0?(2+Math.min(6,n))/2:2/(2-Math.max(-6,n));
const signal=(score,evidence=[])=>({
  status:'evaluated',score:clamp(score),evidence,
});
const unknown=reason=>({status:'unknown',score:null,evidence:[],reason});
// Use the same effective-speed rules as Battle, but only with public field
// effects and the candidate's hypothesized item/ability (never the real set).
function projectedSpeed(pokemon,field,seat) {
  const effects=field?.sideEffects;
  if(!Array.isArray(effects)||effects.length!==2||
    !effects.every(Array.isArray)||!finite(pokemon.stats?.spe))return null;
  // The compact public view cannot establish whether Unburden activated.
  if(pokemon.abilityId==='unburden'&&!pokemon.itemId)return null;
  const weatherSpeed={chlorophyll:'sunnyday',sandrush:'sandstorm',
    swiftswim:'raindance',slushrush:'snowscape'};
  const weatherMatch=weatherSpeed[pokemon.abilityId];
  const needsWeather=Boolean(weatherMatch&&weatherMatch===field.weather)||
    (pokemon.abilityId==='slushrush'&&field.weather==='hail');
  // Cloud Nine / Air Lock suppression is not part of the compact view.
  // When a weather speed modifier matters, require an explicit known flag.
  if(needsWeather&&typeof field.weatherSuppressed!=='boolean')return null;
  const sides=effects.map(rows=>Object.fromEntries(rows
    .filter(row=>row&&typeof row.id==='string')
    .map(row=>[row.id,{duration:row.duration??1,layers:row.layers??1}])));
  return fieldSpeed(pokemon,{weather:field.weather??'',
    terrain:field.terrain??'',weatherSuppressed:field.weatherSuppressed===true,
    sides},seat);
}
const stageStat=(pokemon,stat,ignoreBoost=false)=>{
  const current=Number(pokemon.stats?.[stat]??0);
  return Math.max(1,Math.floor(current*stage(ignoreBoost?0:(pokemon.boosts?.[stat]??0))));
};
const percent=(damage,target)=>100*damage/Math.max(1,target.maxHP??target.stats?.hp??1);
const moveHit=move=>move?.accuracy===true?1:
  (finite(move?.accuracy)?Math.max(0,Math.min(1,move.accuracy/100)):1);

function hypothesisCombatant(set,belief,publicOpponent,data) {
  const mega=megaMatchesObservation(set,belief.observations);
  const speciesId=mega?set.mega.speciesId:set.speciesId;
  const species=data.species?.[speciesId];
  if(!species)return null;
  const stats=calculateChampionsStats(species.baseStats,set.statPoints??{},
    data.natures?.[set.nature]??{});
  const types=Array.isArray(publicOpponent.types)&&publicOpponent.types.length
    ?publicOpponent.types:species.types;
  // Merely seeing an Air Balloon is not proof it popped. It might still
  // be equipped; only a separately confirmed public pop can remove it.
  const popped=publicOpponent.airBalloonPopped===true;
  return {
    stats, maxHP:stats.hp, hpPercent:publicOpponent.hpPercent??100,
    types, status:publicOpponent.status??'',
    boosts:publicOpponent.stages??{},
    abilityId:mega?set.mega.abilityId:set.abilityId,
    itemId:set.itemId==='airballoon'&&popped?null:set.itemId,
    moves:set.moves??[],
  };
}

// A point estimate for supported regular attacks, using the engine's shared
// type chart and damage rounding. Unmodelled moves return null, never zero.
export function estimateAttackPercent(attacker,defender,move,data,field={}) {
  // The shared elemental formula cannot model script-driven power, copied
  // attacker stats, multi-hit rolls or moves with custom damage callbacks.
  if(!move||move.category==='Status'||!(move.basePower>0)||
    move.ohko||move.multihit||move.overrideOffensivePokemon||
    move.overrideOffensiveStat||move.overrideDefensiveStat||
    move.referenceHandlers?.some(handler=>[
      'basePowerCallback','damageCallback','onModifyBasePower',
      'onBasePower','onModifyAtk','onModifySpA','onModifyDef',
      'onModifySpD'].includes(handler)))return null;
  if(!Array.isArray(defender?.types)||!defender.types.length)return null;
  const noAbility=attacker.abilityId==='moldbreaker';
  const ignoreGhost=attacker.abilityId==='scrappy'
    &&['Normal','Fighting'].includes(move.type);
  const types=ignoreGhost?defender.types.filter(t=>t!=='Ghost'):defender.types;
  const factor=types.length?effectiveness(move.type,types,data.typeChart):1;
  if(factor===0)return 0;
  const ability=noAbility?null:defender.abilityId;
  if((move.type==='Ground'&&ability==='levitate')
    ||(move.type==='Ground'&&defender.itemId==='airballoon')
    ||(move.type==='Electric'&&['voltabsorb','lightningrod','motordrive'].includes(ability))
    ||(move.type==='Water'&&['waterabsorb','dryskin','stormdrain'].includes(ability))
    ||(move.type==='Fire'&&ability==='flashfire')
    ||(move.type==='Grass'&&ability==='sapsipper'))return 0;
  const special=move.category==='Special';
  let power=Math.floor(move.basePower);
  if(attacker.abilityId==='technician'&&power<=60)power=Math.floor(power*1.5);
  let atk=stageStat(attacker,special?'spa':'atk',ability==='unaware');
  const def=stageStat(defender,special?'spd':'def',
    attacker.abilityId==='unaware');
  if((!special&&attacker.itemId==='choiceband')
    ||(special&&attacker.itemId==='choicespecs'))atk=Math.floor(atk*1.5);
  // The active pair is fully represented here by our known combatant and
  // a single public-catalog opponent hypothesis. Cloud Nine / Air Lock
  // suppress damage weather even if the weather itself is still displayed.
  const suppressesWeather=field.weatherSuppressed===true||
    ['cloudnine','airlock'].includes(attacker.abilityId)||
    ['cloudnine','airlock'].includes(defender.abilityId);
  let weather=4096;
  if(!suppressesWeather&&field.weather==='sunnyday')
    weather=move.type==='Fire'?6144:move.type==='Water'?2048:4096;
  if(!suppressesWeather&&field.weather==='raindance')
    weather=move.type==='Water'?6144:move.type==='Fire'?2048:4096;
  const damage=elementalDamage({
    attack:atk,defense:def,power,roll:8,
    stab:attacker.types?.includes(move.type)??false,
    effectiveness:factor,weatherModifier:weather,
    burned:!special&&attacker.status==='brn'&&attacker.abilityId!=='guts',
    finalModifier:attacker.itemId==='lifeorb'?5325:4096,
  });
  // Survival items are not present damage reduction; record as a cap only
  // when no damage has already broken the item.
  const fullHealth=finite(defender.hp)&&finite(defender.maxHP)
    ?defender.hp===defender.maxHP
    :defender.hpPercent===100;
  const survives=defender.itemId==='focussash'&&fullHealth;
  return percent(survives?Math.min(damage,Math.max(0,defender.maxHP-1)):damage,defender);
}


function visibleEntryLossPercent(actor,field,data) {
  const side=field?.sideEffects?.[0];
  if(!Array.isArray(side))return null;
  const has=id=>side.some(e=>e.id===id);
  const layers=id=>side.find(e=>e.id===id)?.layers??0;
  if(actor.abilityId==='magicguard')return 0;
  let lost=has('stealthrock')?12.5*
    effectiveness('Rock',actor.types,data.typeChart):0;
  const grounded=actor.itemId==='ironball'||(
    actor.itemId!=='airballoon'&&!actor.types.includes('Flying')&&
    !['levitate','eelevate'].includes(actor.abilityId));
  if(grounded) {
    const spikes=layers('spikes');
    lost+=spikes===1?12.5:spikes===2?100/6:spikes>=3?25:0;
  }
  // Toxic Spikes/Sticky Web impose future status/speed costs, not instant
  // direct HP loss; they must be handled by later turn-risk modelling.
  return lost;
}

const selfCombatant=(own,ownSet,data)=>{
  if(!own?.stats||!own?.types||!ownSet?.setId)throw new TypeError('Known own combatant and set required');
  return {...own,maxHP:own.maxHP??own.stats.hp,
    moves:(own.moves??ownSet.moves??[]).map(slot=>typeof slot==='string'?slot:slot.id)
      .filter(Boolean)};
};
const maxDamage=(actor,target,data,field)=>{
  const entries=actor.moves.map(id=>({
    id,move:data.moves?.[id],
    damage:estimateAttackPercent(actor,target,data.moves?.[id],data,field),
    accuracy:moveHit(data.moves?.[id]),
  }));
  // A counter, OHKO or other unsupported attack can be the strongest attack:
  // do not understate danger using only the supported part of a mixed set.
  if(entries.some(e=>e.move?.category!=='Status'&&e.damage===null))
    return null;
  const supported=entries.filter(e=>e.damage!==null);
  // A pure status set has no estimated direct damage, which is not proof
  // that it cannot win through toxic, hazards, boosts or other effects.
  return supported.length
    ?Math.max(0,...supported.map(e=>e.damage*e.accuracy)):null;
};

// Input is own public/private-as-authorized combatant plus a catalog-derived
// belief and a public opponent view. An actual secret enemy set is NEVER read.
export function assessMatchup({ownSet,own,opponentBelief,opponentPublic,catalog,data,field={}}) {
  if(!Array.isArray(catalog)||!data?.moves||!data?.typeChart)
    throw new TypeError('Expected public catalog and battle definitions');
  const me=selfCombatant(own,ownSet,data);
  if(!opponentBelief||!opponentPublic)throw new TypeError('Expected public opponent belief and state');
  const unknownResult=reason=>{
    const missing=unknown(reason);
    return {modelVersion:'matchup-p1-v1',ownSetId:ownSet.setId,
      opponentBelief,attack:missing,defense:missing,moveOrder:missing,
      sustain:missing,switchIn:missing,breakthroughRisk:missing,
      uncertainty:signal(1,['unresolved-opponent-belief']),
      moveEstimates:[]};
  };
  if(opponentBelief.unknown||!opponentBelief.hypotheses?.length)
    return unknownResult(opponentBelief.unknownReason??'no-hypotheses');
  const byId=new Map(catalog.map(set=>[set.setId,set]));
  const considered=opponentBelief.hypotheses.map(h=>{
    const set=byId.get(h.setId);
    const model=set&&hypothesisCombatant(set,opponentBelief,opponentPublic,data);
    if(!model||!finite(h.weight)||h.weight<0)return null;
    return {set,model,weight:h.weight};
  });
  if(considered.some(x=>!x))return unknownResult('invalid-catalog-hypothesis');
  const total=considered.reduce((n,h)=>n+h.weight,0);
  if(!(total>0))return unknownResult('invalid-belief-weights');
  const moves=me.moves.map(id=>{
    const def=data.moves[id];
    const outcomes=considered.map(h=>estimateAttackPercent(me,h.model,def,data,field));
    const known=outcomes.every(n=>n!==null);
    const expectedPercent=known?outcomes.reduce((n,v,i)=>
      n+v*considered[i].weight/total,0):null;
    const immuneCandidateWeight=known?outcomes.reduce((n,v,i)=>
      n+(v===0?considered[i].weight/total:0),0):null;
    return {id,expectedPercent,immuneCandidateWeight,
      minPercent:known?Math.min(...outcomes):null,
      maxPercent:known?Math.max(...outcomes):null};
  });
  const informative=moves.filter(m=>m.expectedPercent!==null);
  // Supported weak attacks cannot prove low pressure or a failed breakthrough
  // when the same moveset includes an attack this evaluator cannot model.
  const unmodelledOwnAttack=moves.some(m=>
    data.moves[m.id]?.category!=='Status'&&m.expectedPercent===null);
  // A status-only set still has an evaluable defense, move order and
  // switch-in risk. Only attack-dependent signals are unknown.
  // Attack pressure is accuracy-adjusted. Uncalibrated relative scores
  // are not win probabilities and are never advertised as certain hits.
  const best=informative.length?Math.max(...informative.map(m=>
    m.expectedPercent*moveHit(data.moves[m.id]))):null;
  const attackUnknown=unmodelledOwnAttack||best===null;
  const attackReason=unmodelledOwnAttack
    ?'unmodelled-own-attacks':'no-modelled-attacking-moves';
  const incoming=considered.map(h=>maxDamage(h.model,me,data,field));
  const pressureKnown=incoming.every(value=>value!==null);
  const theirPressure=pressureKnown?considered.reduce((n,h,i)=>n+
    incoming[i]*h.weight/total,0):null;
  // Trick Room reverses normal speed ordering. Move priority and choice
  // locking remain move-specific and are not predicted by this broad axis.
  const trickRoom=(field?.trickRoomTurns??0)>0;
  const mySpeed=projectedSpeed(me,field,0);
  const enemySpeeds=considered.map(h=>projectedSpeed(h.model,field,1));
  const speedKnown=mySpeed!==null&&enemySpeeds.every(n=>n!==null);
  const speedDelta=speedKnown?considered.reduce((n,h,i)=>n+
    (trickRoom?-1:1)*(mySpeed-enemySpeeds[i])/
      Math.max(1,mySpeed,enemySpeeds[i])*h.weight/total,0):null;
  const entryLoss=visibleEntryLossPercent(me,field,data);
  const healthRemaining=100*(me.hp??me.maxHP)/Math.max(1,me.maxHP);
  const incomingWorst=pressureKnown?Math.max(...incoming):null;
  const switchIn=entryLoss===null
    ?unknown('unknown-own-entry-hazards')
    :incomingWorst===null
      ?unknown('unmodelled-opponent-attacks')
      :signal((healthRemaining-entryLoss-incomingWorst)/100,
        ['own-entry-hazard-loss:'+entryLoss.toFixed(2),'catalog-hypotheses:'+considered.length]);
  const healing=me.moves.some(id=>['recover','roost','slackoff','rest','moonlight',
    'synthesis','morningsun','strengthsap'].includes(id));
  const evidence=['catalog-hypotheses:'+considered.length];
  // Gini ambiguity of the current, still-uncalibrated catalog weights.
  // Unlike dividing by the entire catalog size this retains discrimination
  // between two plausible sets and a single surviving set.
  const uncertainty=1-considered.reduce((n,h)=>n+(h.weight/total)**2,0);
  return {
    modelVersion:'matchup-p1-v1',ownSetId:ownSet.setId,opponentBelief,
    attack:attackUnknown?unknown(attackReason):signal(best/45-1,evidence),
    defense:pressureKnown
      ?signal(1-theirPressure/45,evidence)
      :unknown('unmodelled-opponent-attacks'),
    moveOrder:speedKnown?signal(speedDelta,evidence):unknown('unmodelled-speed-field'),
    sustain:pressureKnown
      ?signal((healing?0.35:0)-theirPressure/100,evidence)
      :unknown('unmodelled-opponent-attacks'),
    switchIn,
    breakthroughRisk:attackUnknown?unknown(attackReason):signal(1-best/45,evidence),
    uncertainty:signal(uncertainty,evidence),
    moveEstimates:moves,
  };
}

// Strict convenience boundary for running P1 directly on Battle's authorized
// compact CPU view. The caller supplies ONLY its own known set.
export function assessCompactMatchup(view,{ownSet,catalog,data,previousBelief=null}={}) {
  const own=view?.self?.team?.[view.self.active];
  if(!own||!ownSet?.setId)
    throw new TypeError('Expected own selected set and compact AI view');
  const beliefState=projectCompactOpponentBelief(view,catalog,previousBelief);
  const assessment=assessMatchup({
    ownSet,own,opponentBelief:beliefState.belief,
    opponentPublic:view.opponent,catalog,data,field:view.field,
  });
  return {beliefState,assessment};
}
