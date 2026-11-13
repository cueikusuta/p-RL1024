import { transformCombatant, clearTransform } from './battle-transform.js';
import { requireRule } from './draft.js';
import { moveLabels, itemLabels, abilityLabels, statusLabels } from '../data/catalog/labels-ja.js';
import { calculateChampionsStats } from './champions-stats.js';
import { elementalDamage, residualItem, consumeHealingBerry, observeHeldItemEvents, announceHeldItemEvent } from './battle-mechanics.js';
import { beforeMoveStatus, residualStatus, switchOutStatus, clearStatus, addConfusion, statusBerry } from './battle-status.js';
import { changeBoosts, resetBoosts, boostStage, BOOST_STATS, whiteHerb, healHP, observeBoostEvents } from './battle-moves.js';
import { observeEffectEvents, notifyEffectEvent, boostMessage, conditionMessage, snapshotConditions, notifyConditionChanges } from './battle-messages.js';
import { runDamageHits, endTurnProtection } from './battle-hit.js';
import { createField, fieldSpeed, weatherUpkeep, lateFieldUpkeep, entryHazards, entryWeather, electricSeed, grassySeed, psychicSeed, residualWeather, residualWish, residualLeechSeed, residualYawn, perishSongTurn, updateWeatherSuppression, updateMimicry } from './battle-field.js';
import { moveRestriction, canStruggle, registerMoveUsed, residualActionRestrictions, legalMoveSlots, solarBeamCharge, electroShotCharge, mentalHerb } from './battle-actions.js';
import { startTurnFlags, prepareProtean } from './battle-secondary.js';
import { productionCombatant } from './battle-production.js';
import { createProductionAttackEvents } from './battle-production-events.js';
import { publishResolvedPublicEvent } from './battle-publication.js';
import { projectPublicDamageEvents, projectPublicMoveUse, projectPublicDelayedMoveArrival, projectPublicBoostChange, projectPublicHeldItemEvent, projectPublicMoveStatusEvent, projectPublicStatusAbilityEvent, projectPublicMoveHealEvent, projectPublicMoveHpLoss, projectPublicAbilityHeal, projectPublicAbilityStatusCure, projectPublicFieldChange, projectPublicHarvestRestoration, projectPublicEntryItemReveal, projectPublicEntryAbilityEvent, projectPublicDamagingHitAbility, projectPublicAbilityLifecycle, projectPublicConditionAbilityEvent, projectPublicNaturalCure, projectPublicAbilityMoveBlock, projectPublicAbilityMoveOutcome, projectPublicNeutralStatusChange, projectPublicNeutralMoveOutcome, projectPublicConfirmedConditionHeal, projectPublicConfirmedGuardBlock, projectPublicConfirmedConditionTransition, projectPublicNeutralBoostNotice, projectPublicUnableToAct, projectPublicNaturalStatusEnd } from './public-effect-events.js';
import { resolveAttack, indirectDamage, grounded, afterAttackItems } from './battle-effects.js';
import { executeStatusMove, endRoost } from './battle-status-moves.js';
import { entryAbility, residualAbility, copyTrace, updateAbility, updateUnnerve, endAbility } from './battle-abilities.js';
import { resolveSelfdestruct, explosionWinner } from './battle-explosion.js';
import { resolveUTurnPivot } from './battle-pivot.js';
import { resolveRoar } from './battle-roar.js';
import { resolveVoltSwitchPivot } from './battle-voltswitch.js';
import { resolveFlipTurnPivot } from './battle-flipturn.js';
import { resolveWhirlwind } from './battle-whirlwind.js';
import { resolveStruggle, STRUGGLE_DEFINITION } from './battle-struggle.js';
import { resolveDragonTail } from './battle-dragontail.js';
import { changeBattleForm, updateDynamicForms } from './battle-forms.js';
import { PUBLIC_BATTLE_CONDITIONS } from './public-battle-conditions.js';

export { PUBLIC_BATTLE_CONDITIONS };

// Independent engine for the explicitly supported development fixtures.
// Mechanical reference: Showdown b8f23c99629132611ba59e99480cb148ea8c5d3d.
export const MOVES = Object.freeze({
  tackle: { name: 'たいあたり', power: 40, pp: 20, priority: 0, target: 'foe' },
  quickattack: { name: 'でんこうせっか', power: 40, pp: 20, priority: 1, target: 'foe' },
  growl: { name: 'なきごえ', power: 0, pp: 20, priority: 0, target: 'foe', atkChange: -1 },
  swordsdance: { name: 'つるぎのまい', power: 0, pp: 20, priority: 0, target: 'self', atkChange: 2 },
  recover: { name: 'じこさいせい', power: 0, pp: 8, priority: 0, target: 'self', heal: true },
  struggle: { name: 'わるあがき', power: 50, pp: 0, priority: 0, target: 'foe', recoil: true },
});

export function championsStats(base, points = {}, nature = 'neutral') {
  // Only neutral fixture natures are supported. Do not silently ignore a real nature.
  requireRule(nature === 'neutral', 'Unsupported nature');
  return calculateChampionsStats(base, points);
}
export function boostedStat(value, stage) {
  requireRule(Number.isInteger(stage) && stage >= -6 && stage <= 6, 'Invalid stat stage');
  return Math.floor(stage >= 0 ? value * (2 + stage) / 2 : value * 2 / (2 - stage));
}
// Neutral physical damage only; unsupported type/ability/item interactions never enter this engine.
export function physicalDamage({ attack, defense, power, atkStage = 0, defStage = 0, crit = false, roll = 0, stab = true, burned = false }) {
  requireRule(Number.isInteger(roll) && roll >= 0 && roll < 16, 'Invalid damage roll');
  requireRule(attack > 0 && defense > 0 && power > 0, 'Invalid damage parameters');
  const atk = boostedStat(attack, crit && atkStage < 0 ? 0 : atkStage);
  const def = boostedStat(defense, crit && defStage > 0 ? 0 : defStage);
  return elementalDamage({ attack: atk, defense: def, power, crit, roll, stab, burned });
}
export function winnerAfterFaints(living, lastFaintSeat) {
  requireRule(Array.isArray(living) && living.length === 2 && living.every(value => typeof value === 'boolean'), 'Invalid living teams');
  if (living[0] && living[1]) return undefined;
  if (living[0]) return 0;
  if (living[1]) return 1;
  requireRule(lastFaintSeat === 0 || lastFaintSeat === 1, 'Missing last faint');
  // Gen 9 checkWin(faintData) awards the side of the last processed faint.
  return lastFaintSeat;
}
export class SeededRandom {
  #state;
  constructor(seed) {
    requireRule(Number.isInteger(seed) && seed > 0 && seed <= 0xffffffff, 'Invalid seed');
    this.#state = seed;
  }
  integer(limit) {
    let x = this.#state;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this.#state = x >>> 0;
    return Math.floor(this.#state / 0x100000000 * limit);
  }
  checkpoint() { return this.#state; }
}
function combatant(set) {
  requireRule(set.fixture === true && set.abilityId === 'fixture-inert' && set.itemId.startsWith('fixture-item-'), 'Unsupported battle set');
  requireRule(Array.isArray(set.types) && set.types.length === 1 && set.types[0] === 'Normal', 'Unsupported battle types');
  requireRule(Array.isArray(set.moves) && set.moves.length === 4 && new Set(set.moves).size === 4, 'Invalid move set');
  const moves = set.moves.map(id => {
    requireRule(MOVES[id] && id !== 'struggle', 'Unsupported move');
    return { id, name: MOVES[id].name, pp: MOVES[id].pp, maxPP: MOVES[id].pp };
  });
  const stats = championsStats(set.baseStats, set.statPoints, set.nature);
  return { instanceId: set.instanceId, speciesId: set.speciesId, speciesName: set.speciesName ?? set.speciesId,
    level: 50, stats, hp: stats.hp, maxHP: stats.hp, moves, atkStage: 0, defStage: 0, status: '', entered: false };
}
export class Battle {
  #headlessPresentation = false;
  #data;
  #definitions;
  #teams;
  #field = createField();
  #resolvingMove = false;
  #suppressionRefreshPending = false;
  #active = [0,0];
  #entrySequence = [0,0];
  #observations = [[],[]];
  #faintAppearances = [null,null];
  #pending = [null,null];
  #rng;
  #log = [];
  // Private text is frozen when the corresponding public line is emitted.
  // Indexes refer to #log; neither overrides nor private causes enter opponent views.
  #seatLogOverrides = [];
  #publicMessageKinds = new Map();
  #statusReflectionCauses = new Map();
  #audioEvents = [];
  #audioSequence = 0;
  #presentationEvents = [];
  #presentationSequence = 0;
  #presentationAction = 0;
  #phase = 'turn';
  #turn = 1;
  #request = 1;
  #outcome = null;
  #lastFaintSeat = null;
  #totalFainted = [0,0];
  #explosionSeat = null;
  #suspendedTurn = null;
  #seed;
  #actions = [];
  #sets;
  #megaUsed = [false,false];
  constructor(teams, seed, {data=null,headlessPresentation=false}={}) {
    requireRule(Array.isArray(teams) && teams.length === 2 && teams.every(team => Array.isArray(team) && team.length === 3), 'Expected two selected teams');
    this.#data=data;
    this.#headlessPresentation=Boolean(headlessPresentation);
    this.#definitions=data ? {...data.moves,struggle:STRUGGLE_DEFINITION} : MOVES;
    this.#teams = teams.map(team => team.map(set=>data ? productionCombatant(set,data) : combatant(set)));
    this.#bindHeldItemEvents();
    this.#sets = structuredClone(teams);
    this.#seed = seed;
    this.#rng = new SeededRandom(seed);
    for (const seat of [0,1])this.#markEntered(seat);
    for (const seat of [0,1])this.#startIllusion(seat);
    for (const seat of [0,1])this.#announce(seat);
    if (data) {
      const order=[0,1].sort((a,b)=>fieldSpeed(this.#current(b),this.#field,b)-fieldSpeed(this.#current(a),this.#field,a) || (this.#rng.integer(2) ? 1 : -1));
      for(const seat of order)this.#enter(seat);
      for(const seat of [0,1])this.#current(seat).activeTurns=1;
    }
  }
  #seat(seat) { requireRule(seat === 0 || seat === 1, 'Invalid player'); }
  #hasFaintedReserve(seat) { return this.#teams[seat].some((member,index)=>index!==this.#active[seat]&&member.hp===0); }
  #bindHeldItemEvents() {
    for(const team of this.#teams)for(const pokemon of team)observeEffectEvents(pokemon,event=>{
      if(![this.#current(0),this.#current(1)].includes(pokemon))return;
      if(event.kind==='status-reflect') {
        if([this.#current(0),this.#current(1)].includes(event.target))
          this.#statusReflectionCauses.set(event.target,{source:pokemon,ability:event.ability,status:event.status});
        return;
      }
      if(event.kind==='heal') {
        // The mechanics observer only emits 'heal' after positive healing.
        // Resolve active appearance now; never infer from the current ability.
        const resolved=projectPublicAbilityHeal({
          target:pokemon,abilityId:event.ability,confirmed:true,
        },{appearanceOf:holder=>this.#activeAppearanceOf(holder)});
        if(resolved){this.#publishPublicEffect(resolved);return;}
      }
      if((event.id==='disable' && !event.ended && event.causeAbilityId==='cursedbody')
        || (event.ended && ['taunt','attract'].includes(event.id) && event.causeAbilityId==='oblivious')) {
        const source=event.causeAbilityId==='oblivious'?pokemon:event.causeSource;
        const publicEvent=projectPublicConditionAbilityEvent({
          target:pokemon,source,conditionId:event.id,abilityId:event.causeAbilityId,
          ended:Boolean(event.ended),confirmed:true,
        },{appearanceOf:holder=>this.#activeAppearanceOf(holder)});
        if(publicEvent){this.#publishPublicEffect(publicEvent);return;}
      }
      if(event.id && !event.causeAbilityId) {
        const publicEvent=projectPublicConfirmedConditionTransition({
          target:pokemon,id:event.id,ended:Boolean(event.ended),
          causeMoveId:event.causeMoveId,confirmed:true,
        },{appearanceOf:subject=>this.#activeAppearanceOf(subject)});
        if(publicEvent){this.#publishPublicEffect(publicEvent);return;}
      }
      const text=event.kind==='heal' ? `${pokemon.speciesName} は ${abilityLabels[event.ability]??'何らかの効果'}で HPを回復した！` : conditionMessage(pokemon,event);
      // Compatibility fallback: never infer ability/item knowledge from text.
      if(text)this.#message(text);
    });
    for(const team of this.#teams)for(const pokemon of team)observeBoostEvents(pokemon,event=>{
      if(![this.#current(0),this.#current(1)].includes(pokemon))return;
      // A completed stage change is projected before either subscriber runs.
      // Failed/blocked/reset notices retain the existing compatibility path.
      const publicChange=projectPublicBoostChange({target:pokemon,
        source:event.causeSource??pokemon,change:event},
      {appearanceOf:holder=>this.#appearanceOf(holder)});
      if(publicChange){this.#publishPublicEffect(publicChange);return;}
      const neutral=projectPublicNeutralBoostNotice({
        target:pokemon,kind:event.kind,confirmed:true,
      },{appearanceOf:subject=>this.#activeAppearanceOf(subject)});
      if(neutral){this.#publishPublicEffect(neutral);return;}
      // Unmapped notices stay text-only. Never inspect their wording for a
      // private item or ability, and do not invent a new disclosure here.
      this.#message(boostMessage(pokemon,event));
    });
    for(const team of this.#teams)for(const pokemon of team)observeHeldItemEvents(pokemon,({item,kind,other,otherItem})=>{
      // The existing item observer is called only after the effect resolves.
      // Freeze the active public appearance now, not the private current item.
      const publicEvent=projectPublicHeldItemEvent({holder:pokemon,item,kind,other,otherItem},{
        appearanceOf:subject=>this.#activeAppearanceOf(subject),
      });
      if(!publicEvent)return;
      this.#publishPublicEffect(publicEvent);
      if(kind==='exchange')this.#presentState();
    });
  }
  // The public status appearance is captured at the confirmed status change.
  // Only the current active holder may gain a new observed ability, exactly as
  // in the pre-migration #recordObservedAbility path.
  #publishStatusAbilityEvent(target,source,abilityId,variant,statusId,moveId=null) {
    const publicEvent=projectPublicStatusAbilityEvent({
      target,source,abilityId,variant,statusId,moveId,
      moveName:moveId ? moveLabels[moveId]??this.#definitions[moveId]?.name : null,
    },{appearanceOf:pokemon=>this.#activeAppearanceOf(pokemon)});
    if(!publicEvent)return false;
    this.#publishPublicEffect(publicEvent);
    return true;
  }
  #publishStatusMoveChange(target,moveId,statusId,change) {
    const event=projectPublicMoveStatusEvent({
      target,moveId,moveName:moveLabels[moveId]??this.#definitions[moveId]?.name,
      statusId,change,confirmed:true,
    },{appearanceOf:pokemon=>this.#activeAppearanceOf(pokemon)});
    if(!event)return false;
    this.#publishPublicEffect(event);
    return true;
  }
  #publishNeutralBoostNotice(target,kind) {
    const event=projectPublicNeutralBoostNotice({
      target,kind,confirmed:true,
    },{appearanceOf:subject=>this.#activeAppearanceOf(subject)});
    if(!event)return false;
    this.#publishPublicEffect(event);
    return true;
  }
  #publishNeutralStatusChange(target,before,causeConditionId=null) {
    const event=projectPublicNeutralStatusChange({
      target,before,after:target.status,confirmed:true,causeConditionId,
    },{appearanceOf:pokemon=>this.#activeAppearanceOf(pokemon)});
    if(!event)return false;
    this.#publishPublicEffect(event);
    return true;
  }
  #publishNeutralMoveOutcome(moveId,moveName,outcome,reason='generic') {
    const event=projectPublicNeutralMoveOutcome({
      moveId,moveName,outcome,reason,confirmed:true,
    });
    if(!event)return false;
    this.#publishPublicEffect(event);
    return true;
  }
  #logStatusChanges(before, pokemonList,{restActor=null,cureStatusTarget=null,cureStatusMoveId=null,directStatusTarget=null,directStatusMoveId=null,bouncedStatusTarget=null,bouncedStatusMoveId=null,bouncedStatusSource=null,secondaryStatusTarget=null,secondaryStatusMoveId=null,secondaryStatus=null,contactStatusCauses=null}={}) {
    for(const pokemon of pokemonList) {
      const status=before.get(pokemon);
      const reflected=this.#statusReflectionCauses.get(pokemon);
      this.#statusReflectionCauses.delete(pokemon);
      if(status===pokemon.status)continue;
      // Rest directly replaces the user's status with sleep. If a held berry
      // immediately clears sleep, do not claim that the entrant remains asleep.
      if(pokemon===restActor && pokemon.status==='slp') {
        if(!this.#publishStatusMoveChange(pokemon,'rest',pokemon.status,'applied'))
          this.#message(`${pokemon.speciesName} は ${moveLabels.rest}で ${statusLabels.slp}状態になった！`);
      } else if(pokemon===cureStatusTarget && cureStatusMoveId && status && !pokemon.status) {
        // Attribute only a confirmed cure from a resolved move (Heal Bell or
        // Sparkling Aria). Bench cures remain private; other sources are not inferred.
        if(!this.#publishStatusMoveChange(pokemon,cureStatusMoveId,pokemon.status,'cleared'))
          this.#message(`${pokemon.speciesName} は ${moveLabels[cureStatusMoveId]??this.#definitions[cureStatusMoveId].name}で 状態異常が治った！`);
      } else if(reflected?.ability==='synchronize' && reflected.status===pokemon.status) {
        if(!this.#publishStatusAbilityEvent(pokemon,reflected.source,'synchronize','synchronize',pokemon.status)) {
          this.#message(`${pokemon.speciesName} は ${reflected.source.speciesName}の ${abilityLabels.synchronize}で ${statusLabels[pokemon.status]}状態になった！`);
        }
      } else if(pokemon===bouncedStatusTarget && bouncedStatusMoveId && bouncedStatusSource
          && pokemon.status===this.#definitions[bouncedStatusMoveId]?.status) {
        // Magic Bounce's successful reflection is a public ability activation.
        // The result line must name the reflector, move, and actual recipient.
        if(!this.#publishStatusAbilityEvent(pokemon,bouncedStatusSource,'magicbounce','magicbounce',pokemon.status,bouncedStatusMoveId)) {
          this.#message(`${pokemon.speciesName} は ${bouncedStatusSource.speciesName}の ${abilityLabels.magicbounce}で ${moveLabels[bouncedStatusMoveId]??this.#definitions[bouncedStatusMoveId].name}をはね返され、${statusLabels[pokemon.status]}状態になった！`);
        }
      } else if(pokemon===directStatusTarget && directStatusMoveId
          && pokemon.status===this.#definitions[directStatusMoveId]?.status) {
        // Only attribute the final ailment to a direct, unreflected status move.
        // Synchronize, Magic Bounce and instant berry cures can change the actor too.
        if(!this.#publishStatusMoveChange(pokemon,directStatusMoveId,pokemon.status,'applied'))
          this.#message(`${pokemon.speciesName} は ${moveLabels[directStatusMoveId]??this.#definitions[directStatusMoveId].name}で ${statusLabels[pokemon.status]}状態になった！`);
      } else if(pokemon===secondaryStatusTarget && secondaryStatusMoveId && pokemon.status===secondaryStatus) {
        if(!this.#publishStatusMoveChange(pokemon,secondaryStatusMoveId,pokemon.status,'applied'))
          this.#message(`${pokemon.speciesName} は ${moveLabels[secondaryStatusMoveId]??this.#definitions[secondaryStatusMoveId].name}で ${statusLabels[pokemon.status]}状態になった！`);
      } else if(contactStatusCauses?.get(pokemon)?.status===pokemon.status) {
        const {causeAbilityId,source}=contactStatusCauses.get(pokemon);
        // This status change and its ability source were both publicly announced.
        // Attribute the observation to the ability holder, not the recipient.
        if(!this.#publishStatusAbilityEvent(pokemon,source,causeAbilityId,'contact',pokemon.status)) {
          this.#message(`${pokemon.speciesName} は ${abilityLabels[causeAbilityId]}で ${statusLabels[pokemon.status]}状態になった！`);
        }
      } else {
        if(!this.#publishNeutralStatusChange(pokemon,status))
          this.#message(pokemon.status ? `${pokemon.speciesName} は ${statusLabels[pokemon.status]}状態になった！` : `${pokemon.speciesName} の状態異常が治った！`);
      }
    }
  }
  #weatherName(weather) { return ({sunnyday:'ひざしがつよい',raindance:'雨',sandstorm:'砂嵐',snowscape:'雪'})[weather] ?? ''; }
  #terrainName(terrain) { return ({electricterrain:'エレキフィールド',grassyterrain:'グラスフィールド',mistyterrain:'ミストフィールド',psychicterrain:'サイコフィールド'})[terrain] ?? ''; }
  #logWeatherChange(before,sourceMoveId=null,sourceAbility=null) {
    if(before===this.#field.weather)return;
    const publicEvent=projectPublicFieldChange({
      type:'weather',before,after:this.#field.weather,sourceMoveId,
      sourceMoveName:sourceMoveId ? moveLabels[sourceMoveId]??this.#definitions[sourceMoveId]?.name : null,
      sourceAbility,
    },{appearanceOf:pokemon=>this.#appearanceOf(pokemon)});
    if(publicEvent)this.#publishPublicEffect(publicEvent);
    else this.#message('天候がおさまった！');
  }
  #logTerrainChange(before,sourceMoveId=null,sourceAbility=null) {
    if(before===this.#field.terrain)return;
    const publicEvent=projectPublicFieldChange({
      type:'terrain',before,after:this.#field.terrain,sourceMoveId,
      sourceMoveName:sourceMoveId ? moveLabels[sourceMoveId]??this.#definitions[sourceMoveId]?.name : null,
      sourceAbility,
    },{appearanceOf:pokemon=>this.#appearanceOf(pokemon)});
    if(publicEvent)this.#publishPublicEffect(publicEvent);
    else this.#message('フィールドの効果が消えた！');
  }
  #current(seat) { return this.#teams[seat][this.#active[seat]]; }
  #markEntered(seat) {
    const pokemon=this.#current(seat);
    pokemon.entered=true;
    // Entry alone cannot link this appearance to a preview slot: it may be
    // an Illusion. Exact species confirmation is public only after a body hit,
    // formal Illusion reveal, or Mega evolution.
    pokemon.publicIdentityKnown=false;
    // A public key identifies an appearance, never a secret party member.
    pokemon.publicInstanceId=`entry-${++this.#entrySequence[seat]}`;
    this.#observations[seat].push({publicInstanceId:pokemon.publicInstanceId,usedMoves:[],active:true});
    this.#faintAppearances[seat]=null;
  }
  #publicObservation(pokemon, record) {
    return {publicInstanceId:record.publicInstanceId,
      speciesId:pokemon.illusionSpeciesId??pokemon.speciesId,
      speciesName:pokemon.illusionSpeciesName??pokemon.speciesName,
      rosterSpeciesName:pokemon.publicIdentityKnown ? pokemon.baseSpeciesName??pokemon.speciesName : null,
      ...(pokemon.megaEvolved ? {...(pokemon.illusionInstanceId?{}:{baseSpeciesName:pokemon.baseSpeciesName}),types:[...(pokemon.illusionTypes??pokemon.types)]} : {}),
      ...(pokemon.publicIdentityKnown&&pokemon.megaEvolved&&!pokemon.illusionInstanceId
        ? {publicBaseStats:{...this.#data.species[pokemon.speciesId].baseStats}} : {}),
      usedMoves:[...record.usedMoves],...(record.observedItemIds?.length?{observedItemIds:[...record.observedItemIds]}:{}),
      ...(record.observedAbilityIds?.length?{observedAbilityIds:[...record.observedAbilityIds]}:{}),status:pokemon.status,
      hpPercent:pokemon.hp===0?0:Math.max(1,Math.ceil(pokemon.hp/pokemon.maxHP*100)),active:pokemon.hp>0};
  }
  #publicHistory(seat) {
    const pokemon=this.#current(seat);
    return this.#observations[seat].map(record=>record.active && record.publicInstanceId===pokemon.publicInstanceId && pokemon.hp>0
      ? this.#publicObservation(pokemon,record) : record);
  }
  #message(text,{privateSeat=null,privateText=null,publicKind=null}={}) {
    const index=this.#log.length;
    this.#log.push(text);
    // Only an actual server-confirmed public event can set this hint. Do not
    // infer a disclosure or a kind from the Japanese message text.
    if(typeof publicKind==='string' && publicKind) {
      this.#publicMessageKinds.set(index,publicKind);
      if(this.#publicMessageKinds.size>80)
        this.#publicMessageKinds.delete(this.#publicMessageKinds.keys().next().value);
    }
    if((privateSeat===0 || privateSeat===1) && typeof privateText==='string'
      && privateText && privateText!==text)
      this.#seatLogOverrides.push({index,seat:privateSeat,text:privateText});
  }
  #viewLog(seat) {
    const start=Math.max(0,this.#log.length-80);
    const overrides=new Map(this.#seatLogOverrides
      .filter(entry=>entry.seat===seat && Number.isInteger(entry.index)
        && entry.index>=start && entry.index<this.#log.length)
      .map(entry=>[entry.index,entry.text]));
    return this.#log.slice(start).map((message,index)=>overrides.get(start+index)??message);
  }
  #appearanceOf(pokemon) {
    for(const seat of [0,1]) {
      if(!this.#teams[seat].includes(pokemon))continue;
      const id=pokemon.publicInstanceId;
      if(id && this.#observations[seat].some(record=>record.publicInstanceId===id))
        return {seat,publicInstanceId:id};
    }
    return null;
  }
  #publishPublicEffect(publicEvent,{privateSeat=null,privateText=null}={}) {
    publishResolvedPublicEvent(publicEvent,{
      observations:this.#observations,
      appendMessage:text=>this.#message(text,{privateSeat,privateText,publicKind:publicEvent.kind}),
    });
  }
  // Only active appearances are eligible for observer disclosures. Historical
  // delayed effects must use #appearanceOf without guessing the current owner.
  #activeAppearanceOf(pokemon) {
    for(const seat of [0,1]) {
      if(this.#current(seat)!==pokemon)continue;
      const record=this.#observations[seat].at(-1);
      if(record?.active && record.publicInstanceId===pokemon.publicInstanceId)
        return {seat,publicInstanceId:pokemon.publicInstanceId};
    }
    return null;
  }
  #publishMoveUse(actor,moveId) {
    const move=this.#definitions[moveId];
    const moveName=moveLabels[moveId]??move.name;
    const event=projectPublicMoveUse({actor,moveId,moveName},{
      appearanceOf:pokemon=>this.#appearanceOf(pokemon),
    });
    // Choice lock is confirmed by registerMoveUsed before this announcement.
    // The owner already knows their own held item; the opponent does not.
    const ownerSeat=this.#teams.findIndex(team=>team.includes(actor));
    const ownChoiceLock=actor.itemId==='choicescarf'
      && actor.volatiles?.choicelock?.move===moveId;
    this.#publishPublicEffect(event,ownChoiceLock && (ownerSeat===0||ownerSeat===1)
      ?{privateSeat:ownerSeat,
        privateText:`${event.actorName} の ${moveName}！（自分の${itemLabels.choicescarf}で技を固定中）`}
      :{});
  }
  #publishMoveHeal(target,moveId,moveName,amount) {
    const event=projectPublicMoveHealEvent({target,moveId,moveName,amount,confirmed:true},{
      appearanceOf:pokemon=>this.#activeAppearanceOf(pokemon),
    });
    if(!event)return false;
    this.#publishPublicEffect(event);
    return true;
  }
  #publishMoveHpLoss(target,moveId,moveName,delta) {
    const event=projectPublicMoveHpLoss({target,moveId,moveName,delta,confirmed:true},{
      appearanceOf:pokemon=>this.#activeAppearanceOf(pokemon),
    });
    if(!event)return false;
    this.#publishPublicEffect(event);
    return true;
  }
  #startIllusion(seat) {
    const pokemon=this.#current(seat);
    if(pokemon.abilityId!=='illusion')return false;
    pokemon.trueSpeciesName ??=pokemon.speciesName;
    pokemon.speciesName=pokemon.trueSpeciesName;
    delete pokemon.illusionInstanceId;delete pokemon.illusionSpeciesId;delete pokemon.illusionSpeciesName;delete pokemon.illusionTypes;
    if(!pokemon.hp){delete pokemon.trueSpeciesName;return false;}
    for(let index=this.#teams[seat].length-1;index>this.#active[seat];index--) {
      const disguise=this.#teams[seat][index];
      if(disguise.hp<=0)continue;
      pokemon.illusionInstanceId=disguise.instanceId;
      pokemon.illusionSpeciesId=disguise.illusionSpeciesId??disguise.speciesId;
      pokemon.illusionSpeciesName=disguise.illusionSpeciesName??disguise.speciesName;
      pokemon.illusionTypes=[...(disguise.illusionTypes??disguise.types)];
      pokemon.publicIdentityKnown=false;
      pokemon.speciesName=pokemon.illusionSpeciesName;
      return true;
    }
    delete pokemon.trueSpeciesName;
    return false;
  }
  #breakIllusion(pokemon,{silent=false}={}) {
    if(!pokemon?.illusionInstanceId)return false;
    if(!silent)pokemon.publicIdentityKnown=true;
    pokemon.speciesName=pokemon.trueSpeciesName??pokemon.baseSpeciesName??pokemon.speciesName;
    delete pokemon.trueSpeciesName;
    delete pokemon.illusionInstanceId;delete pokemon.illusionSpeciesId;delete pokemon.illusionSpeciesName;delete pokemon.illusionTypes;
    if(!silent)this.#message(`${pokemon.speciesName} の イリュージョンが解けた！`);
    return true;
  }
  #sound(event) {
    if(this.#headlessPresentation){this.#audioSequence++;return;}
    this.#audioEvents.push({sequence:++this.#audioSequence,...event});
    // Retain more than a complete resolved turn, without unbounded room storage.
    this.#audioEvents = this.#audioEvents.slice(-128);
  }
  #frames() {
    if(this.#headlessPresentation)return [null,null];
    // Store only display data. Never put both seats' frames on the wire.
    return [0,1].map(seat=>{
      const {self,opponent,revealed,field,log,turn,request}=this.#viewState(seat);
      const team=self.team.map(p=>({instanceId:p.instanceId,speciesId:p.speciesId,speciesName:p.speciesName,entered:p.entered===true,
        // Preserve the per-event form flag; the viewer must not backfill a future Mega state.
        megaEvolved:p.megaEvolved===true,
        // Ability changes (Trace, Mummy, Skill Swap, etc.) must not be
        // backfilled from the latest state into earlier replay frames.
        ...(p.abilityId!==undefined?{abilityId:p.abilityId}:{}),
        // A post-Mega stat update must not overwrite earlier playback frames.
        ...(p.stats?{stats:{...p.stats}}:{}),
        hp:p.hp,maxHP:p.maxHP,...(p.itemId!==undefined?{itemId:p.itemId}:{}),...(p.types?{types:[...p.types]}:{}),status:p.status,atkStage:p.atkStage,defStage:p.defStage,
        boosts:p.boosts ?? {atk:p.atkStage,def:p.defStage},conditions:p.conditions,conditionDurations:p.conditionDurations,moves:p.moves}));
      const start=Math.max(0,this.#log.length-80);
      const publicMessageKinds=Object.fromEntries([...this.#publicMessageKinds]
        .filter(([index])=>index>=start && index<this.#log.length)
        .map(([index,kind])=>[index-start,kind]));
      return {self:{team,active:self.active},opponent,revealed,field,log,turn,request,
        ...(Object.keys(publicMessageKinds).length?{publicMessageKinds,publicMessageStart:start}:{})};
    });
  }
  #present(event) {
    const recorded={sequence:++this.#presentationSequence,...event};
    if(this.#headlessPresentation)return recorded;
    this.#presentationEvents.push(recorded);
    this.#presentationEvents=this.#presentationEvents.slice(-64);
    return recorded;
  }
  #presentState() {
    if(this.#headlessPresentation){this.#presentationSequence++;return;}
    this.#present({kind:'state',after:this.#frames()});
  }
  #impact(seat,id,effectiveness,hit,before) {
    this.#sound({kind:'hit',effectiveness});
    return this.#present({kind:'hit',action:this.#presentationAction,seat,moveId:id,
      hit:hit.hit,substitute:hit.substitute,effectiveness,audioSequence:this.#audioSequence,
      before,after:this.#frames()});
  }
  #announce(seat,{switching=false}={}) {
    this.#message(`プレイヤー${seat+1}は ${this.#current(seat).speciesName} を繰り出した！`);
    this.#sound({kind:'cry',speciesId:this.#current(seat).illusionSpeciesId??this.#current(seat).speciesId});
    this.#present({kind:'cry',seat,switching,speciesId:this.#current(seat).illusionSpeciesId??this.#current(seat).speciesId,
      audioSequence:this.#audioSequence,after:this.#frames()});
  }
  submit(seat, action, request) {
    const sequence=this.#presentationSequence;
    this.#submit(seat,action,request);
    if(sequence!==this.#presentationSequence)this.#presentState();
  }
  #submit(seat, action, request) {
    this.#seat(seat);
    requireRule(this.#phase !== 'finished' && request === this.#request, 'Stale battle action');
    requireRule(action && typeof action === 'object' && !this.#pending[seat], 'Action already confirmed or invalid');
    const current = this.#current(seat);
    requireRule(action.mega===undefined || typeof action.mega==='boolean','Invalid mega flag');
    if(action.mega)requireRule(this.#phase==='turn' && action.kind==='move' && current.hp>0 && current.mega && !current.megaEvolved && current.itemId===current.megaStoneId && !this.#megaUsed[seat],'Mega evolution unavailable');
    if (this.#phase === 'pivot') {
      const frame = this.#suspendedTurn;
      const required=frame?.requiredSeats ?? [frame?.actorSeat];
      requireRule(frame?.kind === 'pivot' && required.includes(seat), 'Pivot choice belongs to the other player');
      requireRule(action.kind === 'switch', 'Choose a replacement');
      requireRule(Number.isInteger(action.index) && action.index >= 0 && action.index < 3 && action.index !== this.#active[seat], 'Invalid switch');
      requireRule(this.#teams[seat][action.index].hp > 0, 'Cannot switch to fainted member');
      this.#actions.push({ seat, request, action: { kind: action.kind, index: action.index } });
      this.#pending[seat]={kind:'switch',index:action.index,actor:current};
      if(!required.every(i=>this.#pending[i]))return;
      const switchOrder=this.#switchOrder(required);
      for(const i of switchOrder)this.#switch(i,this.#pending[i].index,{transfer:frame.switchTransfers?.[i]});
      this.#pending = [null,null];
      if (this.#checkEnd()) { this.#suspendedTurn = null; this.#request++; return; }
      if ([0,1].some(i => this.#current(i).hp === 0)) {
        this.#suspendedTurn = { ...frame, kind: 'replacement' };
        this.#phase = 'replacement'; this.#request++;
        return;
      }
      this.#resumeSuspendedTurn(frame);
      return;
    }
    if(this.#phase==='replacement' && this.#suspendedTurn?.kind==='replacement')
      requireRule(current.hp===0,'Only a fainted active member needs a replacement');
    requireRule(this.#phase !== 'replacement' || current.hp === 0, 'No replacement needed');
    if (action.kind === 'switch') {
      requireRule(current.hp===0 || !current.volatiles?.lockedmove,'Cannot switch during a locked move');
      if(current.hp>0 && this.#data && this.#isShadowTrapped(seat))requireRule(false,'Cannot switch while trapped');
      requireRule(current.hp===0 || !current.volatiles?.twoturnmove,'Cannot switch during a charging move');
      requireRule(Number.isInteger(action.index) && action.index >= 0 && action.index < 3 && action.index !== this.#active[seat], 'Invalid switch');
      requireRule(this.#teams[seat][action.index].hp > 0, 'Cannot switch to fainted member');
    } else {
      requireRule(this.#phase === 'turn' && current.hp > 0 && action.kind === 'move', 'Invalid battle action');
      if (action.index === 'struggle') requireRule(canStruggle(current,this.#definitions,{hasRevivalTarget:this.#hasFaintedReserve(seat)}), 'Struggle unavailable');
      else {
        requireRule(Number.isInteger(action.index) && action.index >= 0 && action.index < 4, 'Invalid move');
        requireRule(current.moves[action.index].pp > 0 || current.volatiles?.twoturnmove?.move===current.moves[action.index].id || current.volatiles?.lockedmove?.move===current.moves[action.index].id, 'No PP remaining');
        requireRule(!moveRestriction(current,current.moves[action.index].id,this.#definitions[current.moves[action.index].id],{selecting:true}), 'Move unavailable');
        const moveId=current.moves[action.index].id;
        if(this.#data && moveId==='revivalblessing')requireRule(Number.isInteger(action.targetIndex) && action.targetIndex>=0 && action.targetIndex<3 && action.targetIndex!==this.#active[seat] && this.#teams[seat][action.targetIndex].hp<=0,'Revival Blessing needs a fainted ally selection');
        else requireRule(action.targetIndex===undefined,'Unexpected move target selection');
      }
    }
    this.#pending[seat] = { kind: action.kind, index: action.index, actor: current, ...(action.mega?{mega:true}:{}), ...(action.targetIndex!==undefined?{targetIndex:action.targetIndex}:{}) };
    this.#actions.push({ seat, request, action: { kind: action.kind, index: action.index, ...(action.mega?{mega:true}:{}), ...(action.targetIndex!==undefined?{targetIndex:action.targetIndex}:{}) } });
    if (this.#phase === 'replacement') {
      if (this.#suspendedTurn?.kind === 'replacement') {
        this.#resolveSuspendedReplacements();
        return;
      }
      if ([0,1].every(i => this.#current(i).hp > 0 || this.#pending[i])) {
        // Both requests are collected before revealing simultaneous replacements.
        for (const i of this.#switchOrder([0,1].filter(i=>this.#pending[i])))this.#switch(i,this.#pending[i].index);
        this.#pending = [null,null];
        if(!this.#checkEnd())this.#phase=[0,1].some(i=>this.#current(i).hp===0) ? 'replacement' : 'turn';
        this.#request++;
      }
    } else if (this.#pending.every(Boolean)) this.#resolve();
  }
  #switchOrder(seats) {
    const speed=seat=>{
      const p=this.#current(seat);
      // Upstream clears fainted boosts/volatiles, and inactive Pokemon ignore
      // ability/item modifiers before their instaswitch action is ordered.
      return fieldSpeed(p.hp>0 ? p : {...p,boosts:{spe:0},abilityId:'',itemId:''},this.#field,seat);
    };
    return [...seats].sort((a,b)=>(this.#field.trickRoomTurns ? speed(a)-speed(b) : speed(b)-speed(a)) || (this.#rng.integer(2) ? 1 : -1));
  }
  #switch(seat, index,{transfer=null}={}) {
    const outgoing=this.#current(seat),other=this.#current(1-seat);
    delete outgoing.pendingEmergencyExit;
    delete outgoing.pendingEmergencyCandidate;
    const outgoingStatus=outgoing.status;
    if(outgoing.hp>0)this.#message(`${outgoing.speciesName} はボールに戻った！`);
    // Snapshot the disclosed appearance before switch-out healing, form resets,
    // or silent Illusion cleanup. Every switch path uses this same boundary.
    this.#present({kind:'withdraw',seat,fainted:outgoing.hp===0,after:this.#frames()});
    // Freeze only what was observed, before clearing masks/forms or healing.
    const record=this.#observations[seat].at(-1);
    if(record?.active && outgoing.hp>0)Object.assign(record,this.#publicObservation(outgoing,record),{active:false});
    if(outgoing.illusionInstanceId)this.#breakIllusion(outgoing,{silent:true});
    if(this.#data && outgoing.hp>0 && outgoing.abilityId==='zerotohero' && outgoing.baseSpeciesId==='palafin' && outgoing.speciesId==='palafin' && changeBattleForm(outgoing,'palafinhero',this.#data))outgoing.heroMessageDisplayed=false;
    if(outgoing.hp>0 && outgoing.abilityId==='regenerator')outgoing.hp=Math.min(outgoing.maxHP,outgoing.hp+Math.floor(outgoing.maxHP/3));
    if(other.volatiles?.attract?.sourceId===outgoing.instanceId)delete other.volatiles.attract;
    for(const id of ['partiallytrapped','trapped','octolock','jawlock'])if(other.volatiles?.[id]?.sourceId===outgoing.instanceId)delete other.volatiles[id];
    // SwitchOut observes the current (possibly copied) ability before the
    // Transform/clearVolatile boundary restores the authored ability.
    switchOutStatus(this.#current(seat));
    if(outgoingStatus!==outgoing.status) {
      // SwitchOut is public, but the outgoing appearance was already frozen.
      // The confirmed event owns both the Japanese log and its disclosure.
      const publicEvent=projectPublicNaturalCure({
        holder:outgoing,abilityId:outgoing.abilityId,
        previousStatus:outgoingStatus,afterStatus:outgoing.status,confirmed:true,
      },{appearanceOf:pokemon=>this.#appearanceOf(pokemon)});
      if(publicEvent)this.#publishPublicEffect(publicEvent);
    }
    clearTransform(outgoing);
    resetBoosts(this.#current(seat));
    this.#current(seat).volatiles={};
    // Champions clearVolatile resets Rage Fist's per-entry hit counter.
    outgoing.timesAttacked=0;
    this.#current(seat).moveLastTurnResult=undefined;
    if(this.#data) {
      this.#current(seat).types=[...this.#current(seat).originalTypes];
      this.#current(seat).abilityId=this.#current(seat).originalAbilityId;
      this.#current(seat).proteanUsed=false;
      if(this.#current(seat).abilityId==='stancechange' && this.#current(seat).baseSpeciesId==='aegislash')changeBattleForm(this.#current(seat),'aegislash',this.#data);
      if(outgoing.baseSpeciesId==='morpeko' && outgoing.speciesId==='morpekohangry')changeBattleForm(outgoing,'morpeko',this.#data);
    }
    this.#active[seat] = index; this.#markEntered(seat);
    const entryBeforeHP=this.#current(seat).hp;
    this.#current(seat).activeMoveActions = 0;
    this.#current(seat).activeTurns = 0;
    this.#startIllusion(seat);
    const entrySequence=this.#presentationSequence;
    this.#announce(seat,{switching:true});
    if(this.#data)this.#enter(seat);
    if(transfer && this.#current(seat).hp>0) {
      const incoming=this.#current(seat);
      if(transfer.boosts) {
        const previous=BOOST_STATS.map(stat=>boostStage(incoming,stat));
        incoming.boosts={...transfer.boosts};
        incoming.atkStage=incoming.boosts.atk??0;incoming.defStage=incoming.boosts.def??0;
        if(BOOST_STATS.some((stat,index)=>boostStage(incoming,stat)!==previous[index])
          && !this.#publishNeutralBoostNotice(incoming,'copy'))
          this.#message(boostMessage(incoming,{kind:'copy'}));
      }
      if(transfer.volatiles)incoming.volatiles={...incoming.volatiles,...structuredClone(transfer.volatiles)};
      // Switch-in effects resolve before transfer, but the successful transfer
      // must already be visible as the incoming sprite enters. Only project the
      // public flag into recorded frames; actual battle resolution stays intact.
      if(transfer.volatiles?.substitute && incoming.volatiles?.substitute) {
        for(const event of this.#presentationEvents) {
          if(event.sequence<=entrySequence)continue;
          for(const frames of [event.before,event.after])if(frames)for(const [viewer,frame] of frames.entries()) {
            const target=viewer===seat ? frame.self.team[frame.self.active] : frame.opponent;
            if(target && !target.conditions.includes('substitute'))target.conditions.push('substitute');
          }
        }
      }
    }
    if(this.#field.healingWishes?.[seat] && this.#current(seat).hp>0 && (this.#current(seat).hp<this.#current(seat).maxHP || this.#current(seat).status)) {
      const recipient=this.#current(seat);
      const beforeHP=recipient.hp;
      const restoredHP=recipient.hp<recipient.maxHP, curedStatus=Boolean(recipient.status);
      recipient.hp=recipient.maxHP;recipient.status='';recipient.statusState={};this.#field.healingWishes[seat]=null;
      // The source move was already revealed. This delayed confirmed change
      // has no new owner disclosure, and neither outcome is speculative.
      if(restoredHP && !this.#publishMoveHeal(recipient,'healingwish',moveLabels.healingwish,recipient.hp-beforeHP))
        this.#message(`${recipient.speciesName} は ${moveLabels.healingwish}で HPを回復した！`);
      if(curedStatus && !this.#publishStatusMoveChange(recipient,'healingwish',recipient.status,'cleared'))
        this.#message(`${recipient.speciesName} は ${moveLabels.healingwish}で 状態異常が治った！`);
    }
    const entered=this.#current(seat),candidate=entered.pendingEmergencyCandidate;
    if(entered.abilityId==='emergencyexit' && entered.hp>0 && entryBeforeHP>entered.maxHP/2 && entered.hp<=entered.maxHP/2)entered.pendingEmergencyExit=true;
    if(candidate && entered.hp>0 && candidate.beforeHP>entered.maxHP/2 && entered.hp<=entered.maxHP/2)entered.pendingEmergencyExit=true;
    delete entered.pendingEmergencyCandidate;
    this.#presentState();
  }
  #damageEvent(event) {
    if (!event.damage)return;
    const beforeHP=Number.isFinite(event.beforeHP)?event.beforeHP:Math.min(event.target.maxHP,event.target.hp+event.damage);
    if(event.direct && !event.futureMove) {
      event.target.publicIdentityKnown=true;
      this.#breakIllusion(event.target);
    }
    if(this.#resolvingMove && event.target.abilityId==='emergencyexit' && beforeHP>event.target.maxHP/2 && event.target.hp<=event.target.maxHP/2)
      event.target.pendingEmergencyCandidate ??={beforeHP};
    for(const publicEvent of projectPublicDamageEvents(event,{
      appearanceOf:pokemon=>this.#appearanceOf(pokemon),
      // Reveal policy is resolved at the server event boundary, never by a
      // subscriber looking at the current internal Pokemon after publication.
      canDisclose:({kind,id,source})=>kind==='ability'
        ? source.abilityId===id : kind==='item' && source.itemId===id,
    }))this.#publishPublicEffect(publicEvent);
    if(!event.target.hp) {
      const maskedFaintedName=event.target.speciesName;
      const seat=this.#teams[0].includes(event.target) ? 0 : 1;
      if(beforeHP>0)this.#totalFainted[seat]=Math.min(100,this.#totalFainted[seat]+1);
      const observation=this.#observations[seat].find(record=>record.publicInstanceId===event.target.publicInstanceId);
      if(observation) {
        const frozen=this.#publicObservation(event.target,observation);
        Object.assign(observation,frozen,{active:false});
        this.#faintAppearances[seat]={speciesId:frozen.speciesId,speciesName:frozen.speciesName,types:[...(event.target.illusionTypes??event.target.types??[])]};
      }
      this.#breakIllusion(event.target,{silent:true});
      this.#lastFaintSeat=seat;
      this.#message(`${event.direct ? event.target.speciesName : maskedFaintedName} は倒れた！`);
      if(event.direct && !event.futureMove && event.target.volatiles?.destinybond && event.source?.hp>0) {
        const attacker=event.source,loss=attacker.hp;attacker.hp=0;delete event.target.volatiles.destinybond;
        this.#lastFaintSeat=this.#teams[0].includes(attacker)?0:1;
        // A single confirmed public event reports Destiny Bond's cause and the faint.
        if(loss)this.#damageEvent({target:attacker,damage:loss,kind:'condition',id:'destinybond'});
      }
      this.#suppressionRefreshPending=true;
      if(!this.#resolvingMove)this.#flushSuppressionRefresh();
    }
  }
  #flushSuppressionRefresh() {
    if(!this.#suppressionRefreshPending)return false;
    const active=[this.#current(0),this.#current(1)];
    updateWeatherSuppression(this.#field,active);
    updateUnnerve(active);
    this.#suppressionRefreshPending=false;
    return true;
  }
  #isShadowTrapped(seat) {
    const current=this.#current(seat),foe=this.#current(1-seat);
    if(!current.hp || !foe.hp || current.itemId==='shedshell' || current.types?.includes('Ghost'))return false;
    return Boolean(current.volatiles?.partiallytrapped || current.volatiles?.trapped || current.volatiles?.octolock || current.volatiles?.jawlock || current.volatiles?.noretreat) ||
      foe.abilityId==='shadowtag' && current.abilityId!=='shadowtag';
  }
  #updateItems() {
    for(const pokemon of [this.#current(0),this.#current(1)]) {
      mentalHerb(pokemon);statusBerry(pokemon);consumeHealingBerry(pokemon);
    }
  }
  #publishEntryAbility(holder,abilityId,{friskTarget=null,friskedItemId=null,allowFrozen=false}={}) {
    const event=projectPublicEntryAbilityEvent({
      holder,abilityId,confirmed:true,friskTarget,friskedItemId,
    },{appearanceOf:pokemon=>allowFrozen?this.#appearanceOf(pokemon):this.#activeAppearanceOf(pokemon)});
    if(!event)return false;
    this.#publishPublicEffect(event);
    return true;
  }
  #announceEntryAbility(pokemon,foe,entry) {
    if(entry.announcedAbilityId) {
      if(!this.#publishEntryAbility(pokemon,entry.announcedAbilityId)) {
        this.#message(`${pokemon.speciesName} の ${abilityLabels[entry.announcedAbilityId]}！`);
      }
    }
    if(entry.intrepidSword && Object.keys(entry.intrepidSword).length){
      if(!this.#publishEntryAbility(pokemon,'intrepidsword'))this.#message(`${pokemon.speciesName} の ${abilityLabels.intrepidsword}！`);
    }
    if(entry.intimidate && Object.keys(entry.intimidate).length) {
      if(!this.#publishEntryAbility(pokemon,'intimidate')) {
        this.#message(`${pokemon.speciesName} の ${abilityLabels.intimidate}！`);
      }
    }
    if(entry.clearedScreens) {
      if(!this.#publishEntryAbility(pokemon,'screencleaner')) {
        this.#message(`${pokemon.speciesName} の ${abilityLabels.screencleaner ?? 'とくせい'}！`);
      }
    }
    if(entry.friskedItemId) {
      // Frisk has already announced the actual item to both players.
      if(!this.#publishEntryAbility(pokemon,'frisk',{friskTarget:foe,friskedItemId:entry.friskedItemId})) {
        this.#message(`${pokemon.speciesName} の おみとおし！ ${foe.speciesName} は ${itemLabels[entry.friskedItemId]??entry.friskedItemId} を持っている！`);
      }
    }
  }
  #updateWhiteHerb() {
    // Showdown's onAnySwitchIn/onAnyAfterMega/onAnyAfterMove/onResidual,
    // not a per-hit Update: a multi-hit move may lower stats repeatedly.
    for(const pokemon of [this.#current(0),this.#current(1)])whiteHerb(pokemon);
  }
  #enter(seat) {
    const current=this.#current(seat);
    const initialStatus=current.status,initialWeather=this.#field.weather,initialTerrain=this.#field.terrain;
    let weatherCause=null,terrainCause=null,hazardStatusCause=null;
    const announceEntryField=pokemon=>{
      const beforeWeather=this.#field.weather,beforeTerrain=this.#field.terrain;
      if(!entryWeather(this.#field,pokemon,[this.#current(0),this.#current(1)]))return;
      if(!this.#publishEntryAbility(pokemon,pokemon.abilityId)) {
        this.#message(`${pokemon.speciesName} の ${abilityLabels[pokemon.abilityId] ?? 'とくせい'}！`);
      }
      // A confirmed entry ability may set weather or terrain. Track the last
      // actual setter, rather than attributing the final field to the entrant
      // when a Trace or another effect changed it afterwards.
      const source={speciesName:pokemon.illusionSpeciesName??pokemon.speciesName,abilityId:pokemon.abilityId,pokemon};
      if(beforeWeather!==this.#field.weather)weatherCause={...source,weather:this.#field.weather};
      if(beforeTerrain!==this.#field.terrain)terrainCause={...source,terrain:this.#field.terrain};
    };
    updateWeatherSuppression(this.#field,[this.#current(0),this.#current(1)]);
    entryHazards(this.#field,seat,current,this.#current(1-seat),this.#data.typeChart,this.#rng,
      e=>this.#damageEvent({...e,kind:'hazard'}),e=>{hazardStatusCause=e;});
    if(current.hp) {
      // A held Air Balloon is publicly revealed on each active entry, after
      // hazards have been resolved. The public name may be an Illusion mask.
      if(current.itemId==='airballoon') {
        const event=projectPublicEntryItemReveal({holder:current,itemId:'airballoon',confirmed:true},
          {appearanceOf:pokemon=>this.#appearanceOf(pokemon)});
        if(event)this.#publishPublicEffect(event);
      }
      if(current.abilityId==='zerotohero' && current.baseSpeciesId==='palafin' && current.speciesId==='palafinhero' && !current.heroMessageDisplayed) {
        if(!this.#publishEntryAbility(current,'zerotohero')) {
          this.#message(`${current.speciesName} の ${abilityLabels.zerotohero}！`);
        }
        current.heroMessageDisplayed=true;
      }
      if(current.abilityId==='imposter')this.#transform(seat,current,this.#current(1-seat),{startAbility:false});
      if(copyTrace(current,this.#current(1-seat),this.#data.abilityFlags,this.#rng)) {
        if(!this.#publishEntryAbility(current,'trace')) {
          this.#message(`${current.speciesName} の ${abilityLabels.trace}！`);
        }
      }
      announceEntryField(current);
      const entry=entryAbility(current,this.#current(1-seat),this.#teams[seat],this.#field,this.#totalFainted[seat]);
      this.#announceEntryAbility(current,this.#current(1-seat),entry);
      electricSeed(current,this.#field);grassySeed(current,this.#field);psychicSeed(current,this.#field);updateMimicry(current,this.#field.terrain);
      current.entryStarted=true;
      const foe=this.#current(1-seat);
      if(foe.entryStarted && copyTrace(foe,current,this.#data.abilityFlags,this.#rng)) {
        if(!this.#publishEntryAbility(foe,'trace')) {
          this.#message(`${foe.speciesName} の ${abilityLabels.trace}！`);
        }
        announceEntryField(foe);
        this.#announceEntryAbility(foe,current,entryAbility(foe,current,this.#teams[1-seat],this.#field,this.#totalFainted[1-seat]));
      }
    }
    updateWeatherSuppression(this.#field,[this.#current(0),this.#current(1)]);
    updateUnnerve([this.#current(0),this.#current(1)]);
    this.#updateItems();
    this.#updateWhiteHerb();
    updateDynamicForms([this.#current(0),this.#current(1)],this.#field,this.#data);
    if(initialStatus!==current.status) {
      const toxicSpikes=hazardStatusCause?.id==='toxicspikes' && hazardStatusCause.status===current.status;
      if(!this.#publishNeutralStatusChange(current,initialStatus,toxicSpikes?'toxicspikes':null)) {
        const cause=toxicSpikes?'どくびしで ':'';
        this.#message(current.status ? `${current.speciesName} は ${cause}${statusLabels[current.status]}状態になった！` : `${current.speciesName} の状態異常が治った！`);
      }
    }
    this.#logWeatherChange(initialWeather,null,weatherCause);
    this.#logTerrainChange(initialTerrain,null,terrainCause);
  }
  #transform(seat,actor,target,{startAbility=true}={}) {
    const previous=actor.abilityId;
    const boostsBefore=BOOST_STATS.map(stat=>boostStage(actor,stat));
    if(!transformCombatant(actor,target,this.#definitions,this.#data.abilityFlags))return false;
    this.#message(actor.speciesName+' に へんしんした！');
    if(BOOST_STATS.some((stat,index)=>boostStage(actor,stat)!==boostsBefore[index])
      && !this.#publishNeutralBoostNotice(actor,'copy'))
      this.#message(boostMessage(actor,{kind:'copy'}));
    if(startAbility && previous!==actor.abilityId) {
      copyTrace(actor,target,this.#data.abilityFlags,this.#rng);
      entryWeather(this.#field,actor,[this.#current(0),this.#current(1)]);
      this.#announceEntryAbility(actor,target,entryAbility(actor,target,this.#teams[seat],this.#field,this.#totalFainted[seat]));
      updateMimicry(actor,this.#field.terrain);
    }
    updateWeatherSuppression(this.#field,[this.#current(0),this.#current(1)]);
    updateUnnerve([this.#current(0),this.#current(1)]);
    return true;
  }
  #resolve(resuming=false) {
    if(!resuming){
      this.#message(`── ターン ${this.#turn} ──`);
      for(const seat of [0,1])startTurnFlags(this.#current(seat));
      for(const seat of [0,1]) {
        const action=this.#pending[seat];
        if(action?.kind==='move' && action.actor.moves[action.index]?.id==='beakblast') {
          action.actor.volatiles??={};action.actor.volatiles.beakblast=true;
        }
      }
    }
    if(!this.#resolvePreMoveMegaPhase())return;
    this.#continueTurn(this.#turnOrder(),0);
  }
  #resolvePreMoveMegaPhase() {
    // Mega events follow voluntary switches, before moves; Gen 7 recalculates
    // move speed and priority after the transformation and new ability starts.
    const megaSeats=[0,1].filter(seat=>this.#pending[seat].mega);
    if(!megaSeats.length)return true;
    for(const seat of this.#switchOrder([0,1].filter(i=>this.#pending[i].kind==='switch' && !this.#pending[i].resolved))){
      this.#switch(seat,this.#pending[seat].index);this.#pending[seat].resolved=true;
    }
    if(this.#checkEnd()){this.#pending=[null,null];this.#request++;return false;}
    if([0,1].some(i=>this.#current(i).hp===0)){
      this.#suspendedTurn=this.#turnFrame([],0,{kind:'replacement',beforeMega:true});
      this.#phase='replacement';this.#pending=[null,null];this.#request++;return false;
    }
    for(const seat of this.#switchOrder(megaSeats))this.#megaEvolve(seat);
    return true;
  }
  #movePriority(action,move) {
    const actor=action.actor;
    const id=action.index==='struggle'?'struggle':actor.moves[action.index]?.id;
    return move.priority
      +(actor.abilityId==='prankster'&&move.category==='Status'?1:0)
      +(actor.abilityId==='galewings'&&move.type==='Flying'&&actor.hp===actor.maxHP?1:0)
      +(id==='grassyglide'&&this.#field.terrain==='grassyterrain'&&grounded(actor)?1:0);
  }
  #turnOrder() {
    const order = [0,1];
    const rank = seat => {
      const action = this.#pending[seat];
      if(action.kind==='switch')return 10;
      const move=this.#definitions[action.index==='struggle'?'struggle':action.actor.moves[action.index].id];
      const priority=this.#movePriority(action,move);
      let fractionalPriority=0;
      if(move.category!=='Status' && action.actor.abilityId==='quickdraw' && this.#rng.integer(10)<3)fractionalPriority=0.1;
      // Pinned items.ts#quickclaw: +0.1 applies only to non-positive move priority.
      if(priority<=0 && fractionalPriority<=0 && action.actor.itemId==='quickclaw' && this.#rng.integer(5)===0)fractionalPriority=0.1;
      action.priority=priority;
      return priority+fractionalPriority;
    };
    const speedDifference = fieldSpeed(this.#pending[1].actor,this.#field,1) - fieldSpeed(this.#pending[0].actor,this.#field,0);
    const difference = rank(1) - rank(0) || (this.#field.trickRoomTurns ? -speedDifference : speedDifference);
    if (difference > 0 || (difference === 0 && this.#rng.integer(2) === 1)) order.reverse();
    return order;
  }
  #megaEvolve(seat) {
    const p=this.#current(seat),form=this.#data.species[p.mega.speciesId];
    const initialWeather=this.#field.weather,initialTerrain=this.#field.terrain;
    let weatherCause=null,terrainCause=null;
    const recordMegaFieldChange=pokemon=>{
      const beforeWeather=this.#field.weather,beforeTerrain=this.#field.terrain;
      if(!entryWeather(this.#field,pokemon,[this.#current(0),this.#current(1)]))return false;
      // Keep the actual setter internally so the final public field announcement
      // can be attributed to the right appearance, even after Trace.
      const source={speciesName:pokemon.illusionSpeciesName??pokemon.speciesName,
        abilityId:pokemon.abilityId,pokemon};
      if(beforeWeather!==this.#field.weather)weatherCause={...source,weather:this.#field.weather};
      if(beforeTerrain!==this.#field.terrain)terrainCause={...source,terrain:this.#field.terrain};
      return true;
    };
    requireRule(!this.#megaUsed[seat] && !p.megaEvolved && p.itemId===p.megaStoneId && form,'Mega evolution unavailable');
    this.#megaUsed[seat]=true;p.megaEvolved=true;p.publicIdentityKnown=true;
    p.speciesId=p.mega.speciesId;p.speciesName=this.#data.megaStones[p.megaStoneId].nameJa ?? form.name;
    p.stats=calculateChampionsStats(form.baseStats,p.statPoints,this.#data.natures[p.nature]);
    p.types=[...form.types];p.originalTypes=[...form.types];p.weightkg=form.weightkg;
    if(p.abilityId==='flashfire')delete p.volatiles?.flashfire;
    p.abilityId=p.mega.abilityId;p.originalAbilityId=p.mega.abilityId;
    updateWeatherSuppression(this.#field,[this.#current(0),this.#current(1)]);
    copyTrace(p,this.#current(1-seat),this.#data.abilityFlags,this.#rng);
    recordMegaFieldChange(p);const entry=entryAbility(p,this.#current(1-seat),this.#teams[seat],this.#field,this.#totalFainted[seat]);
    updateUnnerve([this.#current(0),this.#current(1)]);
    this.#updateItems();
    this.#updateWhiteHerb();
    updateDynamicForms([this.#current(0),this.#current(1)],this.#field,this.#data);
    this.#message(`プレイヤー${seat+1}の ${p.speciesName} がメガシンカした！`);
    this.#announceEntryAbility(p,this.#current(1-seat),entry);
    // Mega evolution changes the opposing ability before the move phase.
    // Pinned Showdown dispatches Update after Mega, so a Trace that could
    // not copy the previous ability gets another chance here.
    const foe=this.#current(1-seat);
    if(copyTrace(foe,p,this.#data.abilityFlags,this.#rng)) {
      if(!this.#publishEntryAbility(foe,'trace')) {
        this.#message(`${foe.speciesName} の ${abilityLabels.trace}！`);
      }
      if(recordMegaFieldChange(foe) && !this.#publishEntryAbility(foe,foe.abilityId)) {
        this.#message(`${foe.speciesName} の ${abilityLabels[foe.abilityId] ?? 'とくせい'}！`);
      }
      this.#announceEntryAbility(foe,p,entryAbility(foe,p,this.#teams[1-seat],this.#field,this.#totalFainted[1-seat]));
      updateWeatherSuppression(this.#field,[this.#current(0),this.#current(1)]);
      updateUnnerve([this.#current(0),this.#current(1)]);
      this.#updateItems();
      updateDynamicForms([this.#current(0),this.#current(1)],this.#field,this.#data);
    }
    this.#logWeatherChange(initialWeather,null,weatherCause);
    this.#logTerrainChange(initialTerrain,null,terrainCause);
    // Field-change publication already owns its ability disclosure. Do not
    // duplicate it from the mutable weather/terrain setter after publication.
    this.#presentState();
  }
  #continueTurn(order,startIndex) {
    for (let position=startIndex;position<order.length;position++) {
      const result=this.#resolveTurnAction(order,position);
      if(result==='suspended')return;
      if(result==='finish-turn')break;
    }
    this.#finishTurn(order);
  }
  #resolveTurnAction(order,position) {
    const seat=order[position],action=this.#pending[seat];
    if(!action || action.resolved || action.actor.hp===0 || action.actor!==this.#current(seat))return 'continue';
    let moveResult=null;
    if(action.kind==='switch')this.#switch(seat,action.index);
    else moveResult=this.#move(seat,action);
    this.#current(seat).actedTurn=this.#turn;
    // DamagingHit may exchange an ability at HP0 while fainting is pending.
    // Restore authored identity only after the move's reactions have finished.
    if(this.#data)for(const pokemon of [this.#current(0),this.#current(1)])if(!pokemon.hp) {
      endAbility(pokemon,pokemon.abilityId);
      clearTransform(pokemon);
      pokemon.abilityId=pokemon.originalAbilityId;
      resetBoosts(pokemon);
      pokemon.volatiles={};
    }
    if(this.#data)this.#flushSuppressionRefresh();
    // After the active move clears, Own Tempo's Update is no longer suppressed
    // by Mold Breaker. Its earlier temporary confusion can still consume Lum.
    if(this.#data)for(const pokemon of [this.#current(0),this.#current(1)]) {
      const statusBeforeUpdate=pokemon.status;
      updateAbility(pokemon);
      if(statusBeforeUpdate && !pokemon.status) {
        const event=projectPublicAbilityStatusCure({
          holder:pokemon,abilityId:pokemon.abilityId,
          previousStatus:statusBeforeUpdate,afterStatus:pokemon.status,confirmed:true,
        },{appearanceOf:subject=>this.#activeAppearanceOf(subject)});
        if(event)this.#publishPublicEffect(event);
        else {
          this.#message(`${pokemon.speciesName} の ${abilityLabels[pokemon.abilityId] ?? 'とくせい'}で状態異常が治った！`);
        }
      }
      if(pokemon.hp>0 && pokemon.abilityId==='owntempo' && pokemon.volatiles?.confusion) {
        delete pokemon.volatiles.confusion;
        const publicEvent=projectPublicConditionAbilityEvent({
          target:pokemon,source:pokemon,conditionId:'confusion',abilityId:'owntempo',
          ended:true,confirmed:true,
        },{appearanceOf:subject=>this.#activeAppearanceOf(subject)});
        if(publicEvent)this.#publishPublicEffect(publicEvent);
      }
    }
    if(this.#data){this.#updateItems();this.#updateWhiteHerb();}
    if(this.#checkEnd())return 'finish-turn';
    const forcedResult=this.#resolveForcedSwitches(seat,action,moveResult,order,position);
    if(forcedResult!=='continue')return forcedResult;
    return this.#suspendForActionSwitches(seat,action,moveResult,order,position) ? 'suspended' : 'continue';
  }
  #resolveForcedSwitches(seat,action,moveResult,order,position) {
    if(moveResult?.switchItem?.kind==='forced') {
      const reserves=this.#teams[seat].filter(p=>p!==action.actor && p.hp>0);
      const incoming=reserves[this.#rng.integer(reserves.length)];
      this.#switch(seat,this.#teams[seat].indexOf(incoming));
      if(this.#checkEnd())return 'finish-turn';
      if(!this.#current(seat).hp) {
        this.#suspendedTurn=this.#turnFrame(order,position+1,{kind:'replacement'});
        this.#phase='replacement';this.#pending=[null,null];this.#request++;
        return 'suspended';
      }
    }
    if(moveResult?.forcedSwitch && moveResult.success) {
      const targetSeat=moveResult.forcedSwitchSeat??1-seat;
      const outgoingIndex=this.#teams[targetSeat].indexOf(moveResult.outgoing);
      const incomingIndex=this.#teams[targetSeat].indexOf(moveResult.incoming);
      requireRule(outgoingIndex===this.#active[targetSeat] && incomingIndex>=0,'Invalid forced replacement');
      this.#switch(targetSeat,incomingIndex);
      if(this.#checkEnd())return 'finish-turn';
      if(this.#current(targetSeat).hp===0) {
        this.#suspendedTurn={kind:'replacement',turnOrder:[...order],startIndex:position+1,
          actions:this.#suspendedActions({includeMegaAndResolved:false})};
        this.#phase='replacement';this.#pending=[null,null];this.#request++;
        return 'suspended';
      }
    }
    return 'continue';
  }
  #suspendForActionSwitches(seat,action,moveResult,order,position) {
    const requiredSeats=[],switchTransfers=[null,null];
    if(moveResult?.switchItem?.kind==='selected' && this.#current(1-seat)===moveResult.switchItem.pokemon && this.#current(1-seat).hp>0)requiredSeats.push(1-seat);
    const pivotResolver={uturn:resolveUTurnPivot,voltswitch:resolveVoltSwitchPivot,flipturn:resolveFlipTurnPivot}[moveResult?.moveId];
    if(pivotResolver && action.actor===this.#current(seat)) {
      const decision=pivotResolver({moveId:moveResult.moveId,result:moveResult.hitResult,
        actorHp:action.actor.hp,hasBench:this.#teams[seat].some(p=>p!==action.actor && p.hp>0),
        commanded:Boolean(action.actor.volatiles?.commanded)});
      if(decision.switch)requiredSeats.push(seat);
    }
    if(moveResult?.switchMode && action.actor===this.#current(seat) && action.actor.hp>0 && !action.actor.volatiles?.noretreat && !action.actor.volatiles?.commanded && this.#teams[seat].some(p=>p!==action.actor&&p.hp>0)) {
      requiredSeats.push(seat);switchTransfers[seat]=moveResult.switchTransfer ?? null;
    }
    for(const affectedSeat of [0,1]) {
      const affected=this.#current(affectedSeat);
      if(affected.pendingEmergencyExit) {
        delete affected.pendingEmergencyExit;
        if(affected.hp>0 && this.#teams[affectedSeat].some(p=>p!==affected&&p.hp>0))requiredSeats.push(affectedSeat);
      }
    }
    if(!requiredSeats.length)return false;
    const uniqueSeats=[...new Set(requiredSeats)];
    this.#suspendedTurn=this.#turnFrame(order,position+1,{kind:'pivot',actorSeat:uniqueSeats[0],requiredSeats:uniqueSeats,switchTransfers});
    this.#phase='pivot';this.#pending=[null,null];this.#request++;
    return true;
  }
  #suspendedActions({includeMegaAndResolved=true}={}) {
    return this.#pending.map((pending,i)=>pending ? {kind:pending.kind,index:pending.index,
      ...(includeMegaAndResolved && pending.mega?{mega:true}:{}),
      ...(pending.targetIndex!==undefined?{targetIndex:pending.targetIndex}:{}),
      ...(pending.priority!==undefined?{priority:pending.priority}:{}),
      ...(includeMegaAndResolved && pending.resolved?{resolved:true}:{}),actorSeat:i,actorId:pending.actor.instanceId}:null);
  }
  #turnFrame(order,startIndex,extra) {
    return {...extra,turnOrder:[...order],startIndex,actions:this.#suspendedActions()};
  }
  #resumeSuspendedTurn(frame) {
    const actions=frame.actions.map((item,seat)=>item ? {kind:item.kind,index:item.index,mega:item.mega,targetIndex:item.targetIndex,priority:item.priority,resolved:item.resolved,
      actor:this.#teams[seat].find(p=>p.instanceId===item.actorId)} : null);
    // The already-resolved side must not be replayed. Preserve full order for
    // residual tie handling, plus the next action position, in server-only state.
    const order=frame.turnOrder;
    this.#pending=actions;
    this.#suspendedTurn=null;
    this.#phase='turn';
    if(frame.endTurn){this.#advanceAfterTurn();return;}
    if(frame.beforeMega){this.#resolve(true);return;}
    this.#continueTurn(order,frame.startIndex);
  }
  #advanceAfterTurn() {
    if(this.#outcome)return;
    this.#pending=[null,null];this.#request++;this.#turn++;
    this.#phase=[0,1].some(i=>this.#current(i).hp===0)?'replacement':'turn';
  }
  #resolveSuspendedReplacements() {
    const required=[0,1].filter(i=>this.#current(i).hp===0);
    if(!required.every(i=>this.#pending[i]))return;
    const frame=this.#suspendedTurn;
    for(const seat of this.#switchOrder(required))if(this.#pending[seat])this.#switch(seat,this.#pending[seat].index);
    this.#pending=[null,null];
    if(this.#checkEnd()) { this.#suspendedTurn=null;this.#request++;return; }
    if([0,1].some(i=>this.#current(i).hp===0)) { this.#request++;return; }
    this.#request++;
    this.#resumeSuspendedTurn(frame);
  }
  #finishTurn(order) {
    const residualForcedSwitches=[];
    const residualHP=[this.#current(0),this.#current(1)].map((pokemon,seat)=>({pokemon,seat,beforeHP:pokemon.hp}));
    if (!this.#outcome) {
      this.#resolveWeatherAndScheduledEffects(order,residualForcedSwitches);
      this.#resolveResidualDamageAndStatus();
      this.#expireEndTurnEffects();
    }
    if(this.#data && !this.#outcome){this.#updateItems();this.#updateWhiteHerb();}
    // Pinned runAction handles the residual-start snapshot after Update, so a
    // Sitrus Berry that heals the holder above half prevents Emergency Exit.
    if(!this.#outcome)for(const {pokemon,seat,beforeHP} of residualHP) {
      if(this.#current(seat)===pokemon && pokemon.abilityId==='emergencyexit' && pokemon.hp>0 && beforeHP>pokemon.maxHP/2 && pokemon.hp<=pokemon.maxHP/2)
        pokemon.pendingEmergencyExit=true;
    }
    if(this.#resolveEndTurnSwitches(order,residualForcedSwitches))return;
    this.#pending = [null,null]; this.#request++;
    if (this.#outcome) return;
    this.#turn++;
    this.#phase = [0,1].some(i => this.#current(i).hp === 0) ? 'replacement' : 'turn';
  }
  #resolveWeatherAndScheduledEffects(order,residualForcedSwitches) {
    const weatherBefore=this.#field.weather;
    weatherUpkeep(this.#field);
    this.#logWeatherChange(weatherBefore);
    if(this.#data)updateDynamicForms([this.#current(0),this.#current(1)],this.#field,this.#data);
    if(this.#data) {
      const announceHealing=(pokemon,amount,conditionId,abilityId=null)=>{
        if(amount>0) {
          // The activating ability has an independent confirmed subscriber.
          if(abilityId)notifyEffectEvent(pokemon,{kind:'heal',ability:abilityId});
          else {
            const publicEvent=projectPublicConfirmedConditionHeal({
              target:pokemon,conditionId,amount,confirmed:true,
            },{appearanceOf:subject=>this.#activeAppearanceOf(subject)});
            if(publicEvent)this.#publishPublicEffect(publicEvent);
          }
        }
      };
      for(const seat of [...order].sort((a,b)=>fieldSpeed(this.#current(b),this.#field,b)-fieldSpeed(this.#current(a),this.#field,a))) {
        const target=this.#current(seat);
        this.#damageEvent({target,damage:residualWeather(this.#field,target,this.#data.typeChart),kind:'weather',id:'sandstorm'});
        if(this.#checkEnd())break;
        if(!this.#field.weatherSuppressed && this.#field.weather==='sunnyday' && target.abilityId==='solarpower')this.#damageEvent({target,damage:indirectDamage(target,target.maxHP/8),kind:'ability',abilityId:'solarpower'});
        if(this.#checkEnd())break;
        if(!this.#field.weatherSuppressed && this.#field.weather==='raindance' && target.abilityId==='dryskin')
          announceHealing(target,healHP(target,Math.max(1,Math.floor(target.maxHP/8))),abilityLabels.dryskin,'dryskin');
        if(!this.#field.weatherSuppressed && this.#field.weather==='sunnyday' && target.abilityId==='dryskin')this.#damageEvent({target,damage:indirectDamage(target,target.maxHP/8),kind:'ability',abilityId:'dryskin'});
        if(this.#checkEnd())break;
        if(!this.#field.weatherSuppressed && this.#field.weather==='snowscape' && target.abilityId==='icebody')
          announceHealing(target,healHP(target,target.maxHP/16),abilityLabels.icebody,'icebody');
      }
      // Future Sight is a slot residual (order 3); simultaneous impacts are
      // sorted by the recipient's effective Speed, independently of Trick Room.
      // Snapshot the order before one impact can change the active target.
      const dueFutureSeats=this.#outcome?[]:[0,1].filter(seat=>this.#field.futureMoves[seat]?.dueTurn<=this.#turn)
        .sort((a,b)=>fieldSpeed(this.#current(b),this.#field,b)-fieldSpeed(this.#current(a),this.#field,a)
          || (this.#rng.integer(2)?1:-1));
      if(!this.#outcome)for(const targetSeat of dueFutureSeats) {
        const future=this.#field.futureMoves[targetSeat];
        if(!future || future.dueTurn>this.#turn)continue;
        this.#field.futureMoves[targetSeat]=null;
        const target=this.#current(targetSeat);
        if(!target.hp)continue;
        const actor=this.#teams[future.sourceSeat].find(member=>member.instanceId===future.sourceId);
        if(!actor)continue;
        // The earlier use is already public. The scheduled effect is now due,
        // but it has not yet checked accuracy or immunity. Freeze the original
        // source and never attribute the arrival to the active replacement.
        const impactEvent=projectPublicDelayedMoveArrival({
          source:actor,moveId:future.moveId,moveName:'未来予知',confirmed:true,
        },{appearanceOf:pokemon=>this.#appearanceOf(pokemon)});
        if(impactEvent)this.#publishPublicEffect(impactEvent);
        else this.#message(`${actor.speciesName} の 未来予知！`);
        // A delayed Future Sight impact is still a move hit: its per-move
        // EmergencyExit event is eligible during residual upkeep. The separate
        // residual HP snapshot below handles crossings across the whole event.
        this.#resolvingMove=true;
        let result;
        try { result=this.#productionMove(future.sourceSeat,future.moveId,actor,false,null,0,true); }
        finally { this.#resolvingMove=false; }
        if(result?.switchItem?.kind==='forced')residualForcedSwitches.push({seat:future.sourceSeat,actor});
        if(this.#checkEnd())break;
      }
      // Wish is a slot condition with residual order 4. Same-order handlers
      // resolve by the recipient's effective Speed, not by seat or Trick Room.
      if(!this.#outcome)for(const seat of [...order].sort((a,b)=>
        fieldSpeed(this.#current(b),this.#field,b)-fieldSpeed(this.#current(a),this.#field,a))) {
        const target=this.#current(seat),healed=residualWish(this.#field,seat,target,this.#turn);
        announceHealing(target,healed,'wish');
      }
      // Grassy Terrain has residual order 5/suborder 2: every weather
      // callback (order 1) and Wish (order 4) finishes before terrain healing.
      // Leftovers follows at order 5/suborder 4.
      if(!this.#outcome)for(const seat of [...order].sort((a,b)=>
        fieldSpeed(this.#current(b),this.#field,b)-fieldSpeed(this.#current(a),this.#field,a))) {
        const target=this.#current(seat);
        if(this.#field.terrain==='grassyterrain' && grounded(target))
          announceHealing(target,healHP(target,target.maxHP/16),'grassyterrain');
      }
      if(!this.#outcome)for(const seat of [0,1]) {
        const target=this.#current(seat),restored=residualItem(target);
        if(restored>0) {
          // The existing item observer publishes one confirmed event to both
          // the public log and the opponent's item knowledge, independently.
          announceHeldItemEvent(target,{item:'leftovers',kind:'healed'});
        }
      }
      // Leech Seed is residual order 8: same-order handlers run by effective
      // Speed, regardless of seat or Trick Room. Ties consume seeded RNG.
      // Snapshot the order before any drain can faint a source-side active.
      const leechOrder=this.#outcome?[]:[0,1].filter(seat=>this.#current(seat).hp && this.#current(seat).volatiles?.leechseed)
        .sort((a,b)=>fieldSpeed(this.#current(b),this.#field,b)-fieldSpeed(this.#current(a),this.#field,a)
          || (this.#rng.integer(2)?1:-1));
      if(!this.#outcome)for(const seat of leechOrder) {
        const target=this.#current(seat),event=residualLeechSeed(target,[this.#current(0),this.#current(1)]);
        if(event) {
          this.#damageEvent({target,damage:event.damage,kind:'condition',id:'leechseed'});
          if(event.healed>0)announceHealing(event.receiver,event.healed,'leechseed');
          if(event.healed<0)this.#damageEvent({target:event.receiver,damage:-event.healed,kind:'ability',abilityId:'liquidooze',source:target});
          if(this.#checkEnd())break;
        }
      }
    }
  }
  #resolveResidualDamageAndStatus() {
    // Reference residual priorities: poison/toxic before burn; speed resolves ties.
    const residualOrder = [0,1].filter(seat => ['brn','psn','tox'].includes(this.#current(seat).status)).sort((a,b) => {
      const priority = seat => this.#current(seat).status === 'brn' ? 10 : 9;
      const speed = seat => fieldSpeed(this.#current(seat),this.#field,seat);
      return priority(a) - priority(b) || speed(b) - speed(a) || (this.#rng.integer(2) ? 1 : -1);
    });
    for (const seat of residualOrder) {
      if(this.#outcome)break;
      const actor = this.#current(seat);
      const beforeHP=actor.hp;
      const lost = residualStatus(actor);
      if (lost) {
        this.#damageEvent({target:actor,damage:lost,beforeHP,kind:'status',status:actor.status});
        if (this.#checkEnd()) break;
      }
    }
    if(this.#data && !this.#outcome) {
      // Ghost Curse residual order 12: faster holders resolve first.
      // Trick Room changes action order, not residual event speed.
      const curseOrder=[0,1].filter(seat=>this.#current(seat).hp && this.#current(seat).volatiles.curse)
        .sort((a,b)=>fieldSpeed(this.#current(b),this.#field,b)-fieldSpeed(this.#current(a),this.#field,a)
          || (this.#rng.integer(2)?1:-1));
      for(const seat of curseOrder) {
        const target=this.#current(seat);
        this.#damageEvent({target,damage:indirectDamage(target,target.maxHP/4),kind:'condition',id:'curse'});
        if(this.#checkEnd())break;
      }
    }
    if(this.#data && !this.#outcome) {
      // Partial trapping and Salt Cure share residual order 13 in Showdown.
      // Sort all handlers together by effective Speed, regardless of effect type.
      // Trick Room changes action order, not residual handler order.
      const order13=[];
      for(const seat of [0,1]) {
        const target=this.#current(seat);
        if(!target.hp)continue;
        if(target.volatiles.partiallytrapped)order13.push({seat,effect:'partiallytrapped'});
        if(target.volatiles.saltcure)order13.push({seat,effect:'saltcure'});
      }
      order13.sort((a,b)=>
        fieldSpeed(this.#current(b.seat),this.#field,b.seat) -
          fieldSpeed(this.#current(a.seat),this.#field,a.seat) ||
        (a.effect===b.effect ? (this.#rng.integer(2)?1:-1)
          : a.effect==='partiallytrapped' ? -1 : 1));
      for(const {seat,effect} of order13) {
        if(this.#outcome)break;
        const target=this.#current(seat);
        if(!target.hp)continue;
        if(effect==='partiallytrapped') {
          const trap=target.volatiles.partiallytrapped;
          if(!trap)continue;
          const source=this.#teams.flat().find(p=>p.instanceId===trap.sourceId);
          if(!source?.hp || ![this.#current(0),this.#current(1)].includes(source)) {
            delete target.volatiles.partiallytrapped;
            continue;
          }
          if(--trap.duration<=0){delete target.volatiles.partiallytrapped;continue;}
          this.#damageEvent({target,damage:indirectDamage(target,target.maxHP/8),kind:'condition',id:'partiallytrapped'});
        } else {
          if(!target.volatiles.saltcure)continue;
          const divisor=target.types.some(type=>type==='Water'||type==='Steel')?8:16;
          this.#damageEvent({target,damage:indirectDamage(target,target.maxHP/divisor),kind:'condition',id:'saltcure'});
        }
        if(this.#checkEnd())break;
      }
    }
    if(this.#data && !this.#outcome) {
      // Octolock has residual order 14 in the pinned reference, after trapping
      // damage (order 13). Keep the effects separate even on the same holder.
      const octoOrder=[0,1].filter(seat=>this.#current(seat).hp &&
        this.#current(seat).volatiles.octolock)
        .sort((a,b)=>fieldSpeed(this.#current(b),this.#field,b)-fieldSpeed(this.#current(a),this.#field,a)
          || (this.#rng.integer(2)?1:-1));
      for(const seat of octoOrder) {
        const target=this.#current(seat),octolock=target.volatiles.octolock;
        if(!octolock)continue;
        const source=this.#teams.flat().find(p=>p.instanceId===octolock.sourceId);
        if(source?.hp && [this.#current(0),this.#current(1)].includes(source)) {
          changeBoosts(target,{def:-1,spd:-1},{source,effect:'octolock'});whiteHerb(target);
        } else delete target.volatiles.octolock;
      }
    }
    if(this.#data && !this.#outcome) {
      // Yawn residual order 23 precedes Perish Song order 24.
      // Residual handlers use effective Speed, not Trick Room action order.
      const yawnOrder=[0,1].filter(seat=>this.#current(seat).hp && this.#current(seat).volatiles?.yawn);
      yawnOrder.sort((a,b)=>fieldSpeed(this.#current(b),this.#field,b)-fieldSpeed(this.#current(a),this.#field,a)
        || (this.#rng.integer(2)?1:-1));
      for(const seat of yawnOrder) {
        const target=this.#current(seat);
        const conditionBefore=snapshotConditions(target);
        const applied=residualYawn(target,this.#turn,this.#field,this.#data.typeChart,this.#rng);
        notifyConditionChanges(target,conditionBefore);
        // A sleep berry may cure the status immediately; never report enduring
        // sleep when the actual post-berry status is clear.
        if(applied && target.status==='slp'
          && !this.#publishStatusMoveChange(target,'yawn',target.status,'applied'))
          this.#message(`${target.speciesName} は ${moveLabels.yawn}で ${statusLabels.slp}状態になった！`);
      }
    }
    if(this.#data && !this.#outcome) {
      // Expiring Perish callbacks queue faints before fieldEvent flushes them.
      // Perish Song order 24 uses residual Speed, not Trick Room action order.
      // Queue both simultaneous faints before the final winner check.
      const perishOrder=[0,1].filter(seat=>this.#current(seat).hp && this.#current(seat).volatiles.perishsong);
      const speed=seat=>fieldSpeed(this.#current(seat),this.#field,seat);
      perishOrder.sort((a,b)=>speed(b)-speed(a) || (this.#rng.integer(2)?1:-1));
      for(const seat of perishOrder) {
        const target=this.#current(seat),damage=perishSongTurn(target);
        if(damage)this.#damageEvent({target,damage,kind:'condition',id:'perishsong'});
        else if(this.#checkEnd())break;
      }
      this.#checkEnd();
    }
  }
  #expireEndTurnEffects() {
    for (const seat of [0,1]) {
      const conditionBefore=snapshotConditions(this.#current(seat));
      if (this.#current(seat).volatiles) delete this.#current(seat).volatiles.flinch;
      if (this.#current(seat).volatiles) delete this.#current(seat).volatiles.beakblast;
      const locked=this.#current(seat).volatiles?.lockedmove;
      if(locked) {
        if(this.#current(seat).status==='slp')delete this.#current(seat).volatiles.lockedmove;
        else if(--locked.duration<=0){delete this.#current(seat).volatiles.lockedmove;addConfusion(this.#current(seat),this.#rng,{causeMoveId:locked.move});}
      }
      residualActionRestrictions(this.#current(seat));
      endTurnProtection(this.#current(seat),this.#turn);
      if(this.#data && !this.#outcome)endRoost(this.#current(seat));
      notifyConditionChanges(this.#current(seat),conditionBefore);
    }
    const terrainBefore=this.#field.terrain;
    lateFieldUpkeep(this.#field);
    this.#logTerrainChange(terrainBefore);
    if(this.#data && terrainBefore!==this.#field.terrain)for(const pokemon of [this.#current(0),this.#current(1)])updateMimicry(pokemon,this.#field.terrain);
    for(const seat of [0,1]) {
      // Harvest runs after weather expiry and reads the field at its callback.
      if(this.#data && !this.#outcome) {
        const pokemon=this.#current(seat);
        const effect=residualAbility(pokemon,this.#rng,{
          sun:!this.#field.weatherSuppressed && this.#field.weather==='sunnyday',data:this.#data});
        // Publish only after the mechanics have actually restored the berry.
        // Both the message and the two public facts consume this one event.
        if(effect?.restoredBerryId) {
          const event=projectPublicHarvestRestoration({
            holder:pokemon,berryId:effect.restoredBerryId,confirmed:true,
          },{appearanceOf:holder=>this.#appearanceOf(holder)});
          if(event)this.#publishPublicEffect(event);
        }
      }
      this.#current(seat).activeTurns=(this.#current(seat).activeTurns ?? 0)+1;
    }
  }
  #resolveEndTurnSwitches(order,residualForcedSwitches) {
    // runAction handles forceSwitchFlag after the whole residual event, in side
    // order. Entrants must not receive the outgoing Pokemon's residual ticks.
    if(!this.#outcome)for(const {seat,actor} of residualForcedSwitches.sort((a,b)=>a.seat-b.seat)) {
      if(this.#current(seat)!==actor || actor.hp<=0)continue;
      const reserves=this.#teams[seat].filter(p=>p!==actor && p.hp>0);
      if(reserves.length)this.#switch(seat,this.#teams[seat].indexOf(reserves[this.#rng.integer(reserves.length)]));
    }
    if(residualForcedSwitches.length && !this.#outcome)this.#checkEnd();
    if(!this.#outcome) {
      for(const seat of [0,1]) {
        const target=this.#current(seat),candidate=target.pendingEmergencyCandidate;
        if(candidate && target.hp>0 && candidate.beforeHP>target.maxHP/2 && target.hp<=target.maxHP/2)target.pendingEmergencyExit=true;
        delete target.pendingEmergencyCandidate;
      }
      const requiredSeats=[0,1].filter(seat=>this.#current(seat).pendingEmergencyExit && this.#current(seat).hp>0 && this.#teams[seat].some(p=>p!==this.#current(seat)&&p.hp>0));
      for(const seat of [0,1])delete this.#current(seat).pendingEmergencyExit;
      if(requiredSeats.length) {
        this.#suspendedTurn={kind:'pivot',actorSeat:requiredSeats[0],requiredSeats,turnOrder:[...order],startIndex:order.length,actions:[null,null],endTurn:true};
        this.#phase='pivot';this.#pending=[null,null];this.#request++;return true;
      }
    }
    return false;
  }
  #move(seat, action) {
    this.#presentationAction++;
    let result;
    this.#resolvingMove=true;
    try { result=this.#executeMove(seat,action); }
    finally { this.#resolvingMove=false; }
    this.#presentState();
    return result;
  }
  #executeMove(seat, action) {
    const actor = action.actor, foe = this.#current(1-seat);
    // A knocked-out target cannot receive a queued attack before forced replacement.
    const id = action.index === 'struggle' ? 'struggle' : actor.moves[action.index].id;
    if(id!=='destinybond' && actor.volatiles?.destinybond)delete actor.volatiles.destinybond;
    const move = this.#definitions[id];
    // Pinned Charge condition clears on AfterMove or MoveAborted, not only on a hit.
    // MoveAborted observes the selected definition before ModifyType.
    const chargeExpires=id!=='charge' && move.type==='Electric';
    actor.activeMoveActions=(actor.activeMoveActions ?? 0)+1;
    const permission = beforeMoveStatus(actor, move, this.#rng, {
      attack: boostedStat(actor.stats.atk, actor.atkStage), defense: boostedStat(actor.stats.def, actor.defStage),
      restriction:()=>{
        const reason=moveRestriction(actor,id,move);
        return reason==='choicelock' ? '' : reason;
      },
      // The status can end before a move is announced (or even if later blocked).
      // Do not attribute a thaw to an unannounced move or to an already-consumed berry.
      onStatusEnd:status=>{
        const event=projectPublicNaturalStatusEnd({
          target:actor,before:status,after:actor.status,confirmed:true,
        },{appearanceOf:subject=>this.#activeAppearanceOf(subject)});
        if(event)this.#publishPublicEffect(event);
        else this.#message(status==='slp'
          ? `${actor.speciesName} は目を覚ました！`
          : `${actor.speciesName} のこおりがとけた！`);
      },
    });
    if (permission.canMove && moveRestriction(actor,id,move)==='choicelock') {
      permission.canMove=false;permission.reason='choicelock';
    }
    if (!permission.canMove) {
      if(chargeExpires && actor.volatiles)delete actor.volatiles.charge;
      actor.moveLastTurnResult=permission.reason==='recharge'?null:false;
      if(actor.volatiles)delete actor.volatiles.destinybond;
      if(actor.volatiles)delete actor.volatiles.twoturnmove;
      const reasons = {slp:'眠っている',frz:'凍りついている',par:'体がしびれている',flinch:'ひるんでいる',confusion:'混乱して自分を攻撃した',attract:'メロメロで動けない',taunt:'ちょうはつされて技を使えない',disable:'かなしばりで技を封じられている',healblock:'かいふくふうじで回復技を使えない',throatchop:'じごくづきで音の技を使えない',choicelock:'持ち物で技を固定されている',recharge:'反動で動けない'};
      // Confusion's confirmed self-hit is already one causal damage event.
      // No move was announced when this permission check failed: neither
      // the selected move nor a hidden item/ability can be disclosed here.
      if(permission.reason!=='confusion' || !permission.damage) {
        const event=projectPublicUnableToAct({
          actor,reason:permission.reason,confirmed:true,
        },{appearanceOf:subject=>this.#activeAppearanceOf(subject)});
        const privateOptions=permission.reason==='choicelock'
          ?{privateSeat:seat,
            privateText:`${actor.speciesName} は ${itemLabels.choicescarf}で別の技を出せなかった。`}
          :{};
        if(event)this.#publishPublicEffect(event,privateOptions);
        else if(permission.reason==='choicelock')
          this.#message(`${actor.speciesName} は技を出せなかった。`,privateOptions);
        else this.#message(`${actor.speciesName} は${reasons[permission.reason]}。`);
      }
      if(permission.damage)this.#damageEvent({target:actor,damage:permission.damage,beforeHP:permission.beforeHP,
        ...(permission.reason==='confusion'?{kind:'condition',id:'confusion'}:{})});
      else if (actor.hp === 0) { this.#lastFaintSeat = seat; this.#message(`${actor.speciesName} は倒れた！`); }
      return null;
    }
    const continuation=actor.volatiles?.twoturnmove?.move===id;
    const lockedContinuation=actor.volatiles?.lockedmove?.move===id;
    if (id !== 'struggle' && !continuation && !lockedContinuation) {
      const foe=this.#current(1-seat),foeAbility=foe.abilityId;
      const pressureTargets=move.flags?.mustpressure || !['self','allySide','allyTeam','foeSide'].includes(move.target);
      actor.moves[action.index].pp--;
      if(foe.hp && foeAbility==='pressure' && pressureTargets)actor.moves[action.index].pp=Math.max(0,actor.moves[action.index].pp-1);
    }
    registerMoveUsed(actor,id);
    actor.usedMoves ??= [];
    if(!actor.usedMoves.includes(id))actor.usedMoves.push(id);
    this.#publishMoveUse(actor,id);
    if(this.#data && ['solarbeam','electroshot'].includes(id)) {
      const effectiveWeather=actor.abilityId==='megasol'?'sunnyday':this.#field.weatherSuppressed?'':this.#field.weather;
      const charge=id==='solarbeam' ? solarBeamCharge(actor,effectiveWeather) : electroShotCharge(actor,effectiveWeather);
      if(!charge.attack) {
        prepareProtean(actor,{...move,id});
        this.#message(`${actor.speciesName} は${id==='solarbeam'?'光を吸収している。':'エネルギーを集めている。'}`);
        if(chargeExpires && actor.volatiles)delete actor.volatiles.charge;
        actor.moveLastTurnResult=undefined;return null;
      }
    }
    if(this.#data) {
      const result=this.#productionMove(seat,id,actor,false,action.targetIndex??null,action.priority??0);
      actor.moveLastTurnResult=result?.notFail?null:Boolean(result?.success ?? result);
      return result?.moveId ? result : null;
    }
    if (move.heal) {
      const heal = Math.min(actor.maxHP - actor.hp, Math.round(actor.maxHP / 2)); actor.hp += heal;
      this.#message(heal ? `${actor.speciesName} のHPが回復した。` : 'しかし うまく決まらなかった。');
    } else if (move.atkChange) {
      const target = move.target === 'self' ? actor : foe;
      if (target.hp === 0) { this.#message('しかし 相手はいなかった。'); return; }
      const before = target.atkStage;
      changeBoosts(target,{atk:move.atkChange},{source:actor});
      this.#message(before === target.atkStage ? '攻撃はこれ以上変化しない。' : `${target.speciesName} の攻撃が${move.atkChange > 0 ? '上がった' : '下がった'}！`);
    } else {
      if (foe.hp === 0) { this.#message('しかし 相手はいなかった。'); return; }
      const before=this.#frames();
      runDamageHits({move:{id,accuracy:true,flags:{protect:1}},actor,target:foe,turn:this.#turn,rng:this.#rng,
        onHit:hit=>this.#impact(seat,id,'normal',hit,before),
        damageForHit:()=>{
          const crit = this.#rng.integer(24) === 0;
          const damage = physicalDamage({ attack: actor.stats.atk, defense: foe.stats.def,
            power: move.power, atkStage: actor.atkStage, defStage: foe.defStage, crit, roll: this.#rng.integer(16), stab: id !== 'struggle', burned: actor.status === 'brn' });
          if (crit) this.#message('急所に当たった！');
          return damage;
        }});
      this.#message(`${foe.speciesName} にダメージ。`);
      if (foe.hp === 0) { this.#lastFaintSeat = 1-seat; this.#message(`${foe.speciesName} は倒れた！`); }
      if (move.recoil) {
        actor.hp = Math.max(0, actor.hp - Math.max(1, Math.round(actor.maxHP / 4)));
        this.#message(`${actor.speciesName} は反動を受けた。`);
        if (actor.hp === 0) { this.#lastFaintSeat = seat; this.#message(`${actor.speciesName} は倒れた！`); }
      }
      return null;
    }
    return null;
  }
  #productionMove(seat,id,actor,called=false,targetIndex=null,priority=0,delayed=false) {
    const move=this.#definitions[id];
    // AfterMove observes the effective type, including called moves. Delayed
    // attacks are impact callbacks, not a new move use by their saved source.
    const type=id==='aurawheel' && actor.speciesId==='morpekohangry' ? 'Dark' : move.type;
    try { return this.#executeProductionMove(seat,id,actor,called,targetIndex,priority,delayed); }
    finally { if(!delayed && id!=='charge' && type==='Electric' && actor.volatiles)delete actor.volatiles.charge; }
  }
  #executeProductionMove(seat,id,actor,called=false,targetIndex=null,priority=0,delayed=false) {
    this.#statusReflectionCauses.clear();
    const foe=this.#current(1-seat),move=this.#definitions[id];
    const statusBefore=new Map([actor,foe].map(pokemon=>[pokemon,pokemon.status]));
    const weatherBefore=this.#field.weather,terrainBefore=this.#field.terrain;
    // Sleep Talk's selected move is an independently visible move use in the
    // battle log, so retain it in this same appearance's public history.
    if(called) {
      actor.usedMoves ??=[];
      if(!actor.usedMoves.includes(id))actor.usedMoves.push(id);
    }
    const sourceActive=!delayed || this.#current(seat)===actor && actor.hp>0;
    const eventActor=sourceActive?actor:{...actor,abilityId:'noability',itemId:''};
    const continuingLock=actor.volatiles?.lockedmove?.move===id ? actor.volatiles.lockedmove : null;
    if(!delayed && this.#data && actor.abilityId==='stancechange' && actor.baseSpeciesId==='aegislash' && (id==='kingsshield' || move.category!=='Status'))
      changeBattleForm(actor,id==='kingsshield'?'aegislash':'aegislashblade',this.#data);
    if(id==='lastrespects')actor.lastRespectsFainted=this.#totalFainted[seat];
    if(move.selfdestruct==='always' && (actor.hp>0 && actor.abilityId==='damp' || foe.hp>0 && foe.abilityId==='damp' && actor.abilityId!=='moldbreaker')) {
      // Identify the public ability source by seat without revealing a disguised species.
      const dampSeat=actor.hp>0 && actor.abilityId==='damp' ? seat : 1-seat;
      const publicEvent=projectPublicAbilityMoveBlock({
        holder:this.#current(dampSeat),holderSeat:dampSeat,abilityId:'damp',
        moveId:id,moveName:moveLabels[id]??move.name,confirmed:true,
      },{appearanceOf:pokemon=>this.#activeAppearanceOf(pokemon)});
      if(publicEvent)this.#publishPublicEffect(publicEvent);
      return {moveId:id,success:false};
    }
    if(called && ['healblock','throatchop'].includes(moveRestriction(actor,id,move)))return false;
    let result,switchItem=null,stateEventsLogged=false,secondaryStatusCause=null,contactStatusCauses=new Map(),sparklingAriaCured=false;
    const effectLogStart=this.#log.length;
    if(id==='roar' || id==='whirlwind') {
      const resolver=id==='roar' ? resolveRoar : resolveWhirlwind;
      // Magic Bounce's TryHit runs before type/Prankster immunity and reflects
      // once. Resolve against the original user's team without spending foe PP.
      const reflected=move.flags?.reflectable && foe.abilityId==='magicbounce' && actor.abilityId!=='moldbreaker';
      const targetSeat=reflected?seat:1-seat;
      result=resolver({id,actor:reflected?foe:actor,target:reflected?actor:foe,
        opponentTeam:this.#teams[targetSeat],rng:this.#rng,turn:this.#turn});
      if(!result.success) {
        // Good as Gold prevents forced switching; it does not reflect the move.
        // Identify the already public defending seat, not a possibly disguised species.
        if(result.reason==='good-as-gold') {
          const blockerSeat=reflected?seat:1-seat;
          const publicEvent=projectPublicAbilityMoveBlock({
            holder:this.#current(blockerSeat),holderSeat:blockerSeat,abilityId:'goodasgold',
            moveId:id,moveName:moveLabels[id]??move.name,confirmed:true,
          },{appearanceOf:pokemon=>this.#activeAppearanceOf(pokemon)});
          if(publicEvent)this.#publishPublicEffect(publicEvent);
        } else if(!this.#publishNeutralMoveOutcome(id,moveLabels[id]??move.name,'failed'))
          this.#message(`${moveLabels[id]??move.name}：しかし うまく決まらなかった。`);
        return {moveId:id,success:false,reason:result.reason};
      }
      return {...result,moveId:id,forcedSwitchSeat:targetSeat};
    } else if(move.category==='Status') {
      const hpBeforeStatusMove=[actor.hp,foe.hp];
      result=executeStatusMove({id,definition:move,actor,target:foe,field:this.#field,seat,teams:this.#teams,reviveIndex:targetIndex,
        priority,transform:()=>this.#transform(seat,actor,foe),
        active:[this.#current(0),this.#current(1)],chart:this.#data.typeChart,definitions:this.#definitions,rng:this.#rng,turn:this.#turn,
        actorAlreadyTrapped:this.#isShadowTrapped(seat),
        remainingActions:foe.actedTurn!==this.#turn,
        targetWillMove:foe.actedTurn!==this.#turn && this.#pending[1-seat]?.kind==='move',
        overrideAction:moveId=>{
          const pending=this.#pending[1-seat];
          if(pending?.kind==='move' && pending.actor===foe) {
            const nextIndex=foe.moves.findIndex(slot=>slot.id===moveId);
            if(nextIndex>=0) {
              pending.index=nextIndex;
              // Encore replaces the queued action; terrain checks must use
              // the replacement's effective priority, not the old move's.
              pending.priority=this.#movePriority(pending,this.#definitions[moveId]);
            }
          }
        }});
      if(result.success) {
        const directStatusCause=move.status && !result.bounced && ['normal','adjacentFoe','foe','any'].includes(move.target)
          ? {directStatusTarget:foe,directStatusMoveId:id} : {};
        const bouncedStatusCause=move.status && result.bounced && ['normal','adjacentFoe','foe','any'].includes(move.target)
          ? {bouncedStatusTarget:actor,bouncedStatusMoveId:id,bouncedStatusSource:foe} : {};
        this.#logStatusChanges(statusBefore,[actor,foe],id==='rest'?{restActor:actor}:id==='healbell'?{cureStatusTarget:actor,cureStatusMoveId:id}:result.bounced?bouncedStatusCause:directStatusCause);
        this.#logWeatherChange(weatherBefore,id);this.#logTerrainChange(terrainBefore,id);
        // Announce only HP restored by a resolved healing move. Other status
        // moves may trigger items; do not misattribute that healing to the move.
        if(id==='painsplit') {
          // Pain Split changes are determined before either Pokémon eats a berry.
          // A berry must not hide a real loss or be credited to Pain Split.
          // HP redistribution is not attack damage; avoid #damageEvent side effects.
          for(const {target,delta} of result.hpChanges??[]) {
            if(delta<0) {
              if(!this.#publishMoveHpLoss(target,id,moveLabels[id]??move.name,delta))
                this.#message(`${target.speciesName} は ${moveLabels[id]??move.name}で HPが減った！`);
            } else if(!this.#publishMoveHeal(target,id,moveLabels[id]??move.name,delta))
              this.#message(`${target.speciesName} は ${moveLabels[id]??move.name}で HPを回復した！`);
          }
        } else if(move.heal || ['rest','morningsun','moonlight','synthesis','strengthsap'].includes(id)) {
          for(const [index,recipient] of [actor,foe].entries()) {
            if(recipient.hp>hpBeforeStatusMove[index]
              && !this.#publishMoveHeal(recipient,id,moveLabels[id]??move.name,recipient.hp-hpBeforeStatusMove[index]))
              this.#message(`${recipient.speciesName} は ${moveLabels[id]??move.name}で HPを回復した！`);
          }
        }
        stateEventsLogged=true;
      }
      if(result.selfDamage)this.#damageEvent({target:actor,damage:result.selfDamage,kind:'move',moveId:id});
      if(result.calledMove) {
        this.#publishMoveUse(actor,result.calledMove);
        return this.#productionMove(seat,result.calledMove,actor,true,null,priority);
      }
      if(result.switchMode) {
        const switchTransfer=result.switchMode==='copyvolatile'
          ? {boosts:structuredClone(actor.boosts),volatiles:Object.fromEntries(Object.entries(actor.volatiles??{}).filter(([key])=>!this.#data.noCopyVolatileIds.includes(key)))}
          : result.switchMode==='shedtail' ? {volatiles:{substitute:structuredClone(actor.volatiles.substitute)}} : null;
        return {moveId:id,success:result.success,switchMode:result.switchMode,switchTransfer};
      }
    } else {
      if(!foe.hp && move.selfdestruct!=='always')return false;
      if(move.flags?.futuremove && !delayed) {
        requireRule(id==='futuresight','Unsupported delayed move: '+id);
        if(this.#field.futureMoves[1-seat])return {moveId:id,success:false};
        prepareProtean(actor,move);
        this.#field.futureMoves[1-seat]={moveId:id,sourceSeat:seat,sourceId:actor.instanceId,dueTurn:this.#turn+2};
        return {moveId:id,success:true,notFail:true};
      }
      if(called && actor.abilityId==='prankster' && foe.types.includes('Dark'))return {moveId:id,success:false};
      // The pending action remains stored for the whole local turn. Only count
      // it as the target's queued move while it still belongs to the active foe.
      const foePending=this.#pending[1-seat];
      const foeWillMove=foe.actedTurn!==this.#turn && foePending?.kind==='move' && foePending.actor===foe;
      const targetAction=foeWillMove
        ? {kind:'move',move:this.#definitions[foePending.index==='struggle' ? 'struggle' : foe.moves[foePending.index]?.id]}
        : {kind:'switch'};
      const attackEvents=createProductionAttackEvents({
        id,move,actor,target:foe,eventActor,field:this.#field,seat,teams:this.#teams,totalFaintedBySide:this.#totalFainted,data:this.#data,
        rng:this.#rng,delayed,sourceActive,targetHpBefore:foe.hp,
        getActive:activeSeat=>this.#current(activeSeat),
        getFrames:()=>this.#frames(),
         headlessPresentation:this.#headlessPresentation,
        impact:(...args)=>this.#impact(...args),
        damageEvent:event=>this.#damageEvent(event),
        breakIllusion:pokemon=>this.#breakIllusion(pokemon),
        message:text=>this.#message(text),
        publishAbilityActivation:(holder,abilityId)=>{
          // A lethal hit can freeze the defender's public appearance before
          // the confirmed DamagingHit ability callback. Keep that identity.
          const event=projectPublicDamagingHitAbility({holder,abilityId,confirmed:true},{
            appearanceOf:subject=>this.#appearanceOf(subject),
          });
          if(!event)return false;
          this.#publishPublicEffect(event);
          return true;
        },
        publishAbilityLifecycle:fact=>{
          const event=projectPublicAbilityLifecycle(fact,{
            appearanceOf:subject=>this.#appearanceOf(subject),
          });
          if(!event)return false;
          this.#publishPublicEffect(event);
          return true;
        },
        publishEntryAbility:(holder,abilityId,options)=>
          this.#publishEntryAbility(holder,abilityId,{...options,allowFrozen:true}),
        updateItems:()=>this.#updateItems(),
        flushSuppression:()=>this.#flushSuppressionRefresh()
      });
      const attackOptions={id,definition:called ? {...move,sleepUsable:true} : delayed ? {...move,ignoreImmunity:false} : move,actor,target:foe,chart:this.#data.typeChart,turn:this.#turn,rng:this.#rng,priority,
        ...(delayed?{allowFaintedActor:true,sourceActive,deferAfterAttackItems:true}:{}),
        field:this.#field,targetSeat:1-seat,
        effectiveSpeeds:{actor:fieldSpeed(actor,this.#field,seat),target:fieldSpeed(foe,this.#field,1-seat)},
        weather:this.#field.weatherSuppressed?'':this.#field.weather,terrain:this.#field.terrain,targetAlreadyActed:!foeWillMove,
        targetAction,
        onPowerBoost:moveId=>{if(moveId==='ficklebeam')this.#message(`${actor.speciesName} の きまぐレーザーの威力が上がった！`);},
        ...attackEvents.callbacks,
        onScreensBroken:({ids})=>{
          // Public defender-side state changed before Brick Break's damage.
          // Emit only the names of screens that were actually removed.
          for(const name of ids)
            this.#message(`${actor.speciesName}の ${moveLabels.brickbreak}で ${foe.speciesName}側の ${moveLabels[name]??name}が消えた！`);
          this.#presentState();
        },
        onBlocked:()=>{
          const guardId=['protect','kingsshield','silktrap','spikyshield'].find(guard=>foe.volatiles?.[guard]?.turn===this.#turn);
          const event=projectPublicConfirmedGuardBlock({
            holder:foe,moveId:id,moveName:moveLabels[id]??move.name,guardId,confirmed:true,
          },{appearanceOf:subject=>this.#activeAppearanceOf(subject)});
          if(event)this.#publishPublicEffect(event);
          else this.#message(`${foe.speciesName} は ${moveLabels[guardId]??'防御技'}で ${moveLabels[id]??move.name}を防いだ！`);
        },
        onHeal:({target,amount})=>{
          if(amount>0 && !this.#publishMoveHeal(target,id,moveLabels[id]??move.name,amount))
            this.#message(`${target.speciesName} は ${moveLabels[id]??move.name}で HPを回復した！`);
        },
        onSecondaryStatus:({target,status})=>{
          if(target===foe)secondaryStatusCause={target,status,moveId:id};
        },
        onContactStatus:({target,status,causeAbilityId,source})=>{
          if(target && status && causeAbilityId && source)
            contactStatusCauses.set(target,{status,causeAbilityId,source});
        },
        onCrit:()=>this.#message('急所に 当たった！')};
      if(move.selfdestruct==='always') {
        result=resolveSelfdestruct({actor,target:foe.hp>0?foe:null,moveId:id,onDamage:e=>this.#damageEvent(e),executeAttack:options=>resolveAttack({...attackOptions,...options})});
        if(result.selfdestruct && id==='explosion')this.#explosionSeat=seat;
      } else if(id==='struggle')result=resolveStruggle(attackOptions);
      else if(id==='dragontail')result=resolveDragonTail({...attackOptions,opponentTeam:this.#teams[1-seat]});
      else result=resolveAttack(attackOptions);
      switchItem=attackEvents.getSwitchItem();
      if(delayed && this.#current(seat)===actor && actor.hp>0 && actor.itemId==='lifeorb') {
        const item=afterAttackItems(move,actor,result.totalDamage,{hitOccurred:result.hits>0});
        if(item<0)this.#damageEvent({target:actor,damage:-item,kind:'item',itemId:'lifeorb'});
      }
      for(const affected of [actor,foe]) {
        const candidate=affected.pendingEmergencyCandidate;
        if(candidate && affected.hp>0 && candidate.beforeHP>affected.maxHP/2 && affected.hp<=affected.maxHP/2)affected.pendingEmergencyExit=true;
        delete affected.pendingEmergencyCandidate;
      }
      if(id==='sparklingaria') {
        if(result.hits && actor.hp>0 && actor.abilityId!=='sheerforce' && foe.volatiles?.sparklingaria && foe.status==='brn')sparklingAriaCured=clearStatus(foe);
        delete foe.volatiles.sparklingaria;
      }
      if(result.hits && move.flags?.recharge && actor.hp){actor.volatiles.recharge=true;}
      if(id==='beakblast')delete actor.volatiles.beakblast;
    }
    // A rampage restarts only the reference's outer duration, not its sampled
    // true duration. A failed continuation ends early without fatigue unless
    // this was already its final turn; successful final hits confuse immediately.
    if(continuingLock && (continuingLock.duration<=1 || !(result.success??result.hits))) {
      delete actor.volatiles.lockedmove;
      if(continuingLock.duration<=1)addConfusion(actor,this.#rng,{causeMoveId:continuingLock.move});
    }
    if(this.#data)updateDynamicForms([this.#current(0),this.#current(1)],this.#field,this.#data);
    if(result.immune) {
      if(result.absorbedAbilityId==='flashfire' && foe.abilityId==='flashfire' && foe.volatiles?.flashfire) {
        // The engine has confirmed a real Fire absorption. The same published
        // event owns the Japanese log and the actual defender's public record.
        const publicOutcome=projectPublicAbilityMoveOutcome({
          holder:foe,abilityId:'flashfire',outcome:'absorbed',
          moveId:id,moveName:moveLabels[id]??move.name,moveType:'Fire',
          charged:true,confirmed:true,
        },{appearanceOf:holder=>this.#activeAppearanceOf(holder)});
        if(publicOutcome)this.#publishPublicEffect(publicOutcome);
      }
      if(!this.#publishNeutralMoveOutcome(id,moveLabels[id]??move.name,'immune'))
        this.#message(`${moveLabels[id]??move.name}：相手には効果がないようだ。`);
    }
    else if(result.missed) {
      if(!this.#publishNeutralMoveOutcome(id,moveLabels[id]??move.name,'missed'))
        this.#message(`${moveLabels[id]??move.name}：技は外れた。`);
    }
    else if(!(result.success ?? result.hits) && this.#log.length===effectLogStart) {
      const reason=id==='trick'?'trick':['taunt','encore','disable'].includes(id)?'status-fail':'generic';
      if(!this.#publishNeutralMoveOutcome(id,moveLabels[id]??move.name,'failed',reason))
        this.#message(id==='trick' ? '持ち物の交換は成立しなかった。' : ['taunt','encore','disable'].includes(id) ? `${moveLabels[id]}は効かなかった。` : `${moveLabels[id]??move.name}：しかし うまく決まらなかった。`);
    }
    const success=Boolean(result.success ?? result.hits);
    if(success && !stateEventsLogged) {
      this.#logStatusChanges(statusBefore,[actor,foe],{
        contactStatusCauses,
        ...(sparklingAriaCured ? {cureStatusTarget:foe,cureStatusMoveId:id}
          : secondaryStatusCause && foe.status===secondaryStatusCause.status
            ?{secondaryStatusTarget:foe,secondaryStatus:foe.status,secondaryStatusMoveId:secondaryStatusCause.moveId}:{}),
      });

      this.#logWeatherChange(weatherBefore);this.#logTerrainChange(terrainBefore,id);
    }
    const notFail=result.notFail || result.blocked==='protect';
    if(id==='dragontail')return {...result,moveId:id,success,switchItem,...(notFail?{notFail:true}:{})};
    if(['uturn','voltswitch','flipturn'].includes(id)) {
      const hitResult=!success ? result.immune ? 'immune' : result.missed ? 'miss' : 'failed' : result.substituteHits ? 'substitute' : 'hit';
      return {moveId:id,success,hitResult,switchItem,...(notFail?{notFail:true}:{})};
    }
    return switchItem || notFail ? {moveId:id,success,...(switchItem?{switchItem}:{}),...(notFail?{notFail:true}:{})} : success;
  }
  #checkEnd() {
    const living = this.#teams.map(team => team.some(p => p.hp > 0));
    const explosionSeat=this.#explosionSeat;this.#explosionSeat=null;
    const winner = (explosionSeat===null ? undefined : explosionWinner(living,explosionSeat)) ?? winnerAfterFaints(living, this.#lastFaintSeat);
    if (winner === undefined) return false;
    this.#finish(winner, 'knockout'); return true;
  }
  #finish(winner, reason) {
    this.#phase = 'finished';
    this.#suspendedTurn = null;
    this.#outcome = { winner, reason, survivors: this.#teams.map((team,seat) => team.filter(p => p.hp > 0 && (reason !== 'surrender' || seat === winner))
      .map(p => ({ instanceId: p.instanceId, alive: true, entered: p.entered }))) };
    this.#message(winner === null ? '引き分け！' : `プレイヤー${winner+1} の勝利！`);
  }
  surrender(seat, request) {
    this.#seat(seat); requireRule(this.#phase !== 'finished' && request === this.#request, 'Stale battle action');
    this.#message(`プレイヤー${seat+1} は降参した。`);
    this.#actions.push({ seat, request, action: { kind: 'surrender' } });
    this.#pending = [null,null]; this.#request++; this.#finish(1-seat, 'surrender');
  }
  result() { return structuredClone(this.#outcome); }
  // Server-only record for replay/debugging; never part of view().
  record() { return structuredClone({ seed: this.#seed, actions: this.#actions }); }
  checkpoint() {
    return structuredClone({ version: 1, production:!!this.#data, sets: this.#sets, teams: this.#teams, field: this.#field, seed: this.#seed,
      rng: this.#rng.checkpoint(), megaUsed:this.#megaUsed, totalFainted:this.#totalFainted, active: this.#active, phase: this.#phase, turn: this.#turn,
      request: this.#request, log: this.#log, seatLogOverrides:this.#seatLogOverrides, entrySequence:this.#entrySequence, observations:this.#observations, faintAppearances:this.#faintAppearances, audioEvents:this.#audioEvents, audioSequence:this.#audioSequence,
      presentationEvents:this.#presentationEvents,presentationSequence:this.#presentationSequence,presentationAction:this.#presentationAction,
      outcome: this.#outcome, lastFaintSeat: this.#lastFaintSeat,
      actions: this.#actions, suspendedTurn:this.#suspendedTurn, pending: this.#pending.map(p => p ? {kind:p.kind,index:p.index,...(p.mega?{mega:true}:{}),...(p.targetIndex!==undefined?{targetIndex:p.targetIndex}:{}),...(p.priority!==undefined?{priority:p.priority}:{}),...(p.resolved?{resolved:true}:{}),actorId:p.actor.instanceId} : null) });
  }
  static restore(saved,{data=null}={}) {
    requireRule(saved?.version === 1, 'Unsupported saved battle');
    requireRule(!saved.production || data,'Production battle data required');
    const b = new Battle(saved.sets, saved.seed,{data:saved.production ? data : null});
    b.#teams = structuredClone(saved.teams); b.#active = [...saved.active];
    b.#bindHeldItemEvents();
    b.#observations=structuredClone(saved.observations);
    b.#faintAppearances=structuredClone(saved.faintAppearances??[null,null]);
    b.#entrySequence=saved.entrySequence ? [...saved.entrySequence] : b.#teams.map(team=>team.filter(p=>p.entered).length);
    b.#megaUsed=[...saved.megaUsed];
    b.#field = structuredClone(saved.field);
    b.#rng = new SeededRandom(saved.rng); b.#phase = saved.phase; b.#turn = saved.turn;
    b.#request = saved.request; b.#log = [...saved.log];
    b.#seatLogOverrides=Array.isArray(saved.seatLogOverrides)
      ?structuredClone(saved.seatLogOverrides):[];
    b.#outcome = structuredClone(saved.outcome);
    b.#audioEvents = structuredClone(saved.audioEvents);
    b.#audioSequence = saved.audioSequence;
    b.#presentationEvents=structuredClone(saved.presentationEvents);
    // Rebuild the transient category index from the most recent persisted
    // public presentation frame. Saved version-1 snapshots without this
    // optional frame metadata still restore through the old text fallback.
    // No new top-level checkpoint field, log format, or public identity data.
    for(const event of [...b.#presentationEvents].reverse()) {
      const frames=event.after;
      const frame=Array.isArray(frames)?frames[0]:null;
      if(!frame || !Number.isInteger(frame.publicMessageStart)
        || !frame.publicMessageKinds)continue;
      for(const [relative,kind] of Object.entries(frame.publicMessageKinds)) {
        const index=frame.publicMessageStart+Number(relative);
        if(Number.isSafeInteger(index) && index>=Math.max(0,b.#log.length-80)
          && index<b.#log.length && typeof kind==='string')
          b.#publicMessageKinds.set(index,kind);
      }
      break;
    }
    b.#presentationSequence=saved.presentationSequence;
    b.#presentationAction=saved.presentationAction;
    b.#totalFainted=Array.isArray(saved.totalFainted) && saved.totalFainted.length===2
      ? [...saved.totalFainted]
      : saved.teams.map(team=>team.filter(member=>member.hp<=0).length);
    b.#lastFaintSeat = saved.lastFaintSeat; b.#actions = structuredClone(saved.actions);
    b.#pending = saved.pending.map((p,seat) => p ? { kind:p.kind,index:p.index,mega:p.mega,targetIndex:p.targetIndex,priority:p.priority,resolved:p.resolved,actor:b.#teams[seat].find(member=>member.instanceId===p.actorId) } : null);
    b.#suspendedTurn = structuredClone(saved.suspendedTurn);
    return b;
  }
  // Public command-selection snapshot. Excludes presentation/audio history; only
  // internal progression tests should use it, not the room's full view protocol.
  decisionView(seat) { return this.#viewState(seat); }
  // Minimal public-information projection for deterministic headless AI.
  // Unlike view()/decisionView(), this does not build logs, presentation state,
  // hidden observations or full private battle objects on every decision.
  // Every mutable field is freshly allocated: callers cannot mutate the battle.
  // Do not add opponent members, moves, items, identities or private sets here.
  decisionViewForAI(seat) {
    this.#seat(seat);
    const active=this.#current(seat),foe=this.#current(1-seat);
    const pivotSeats=this.#phase==='pivot'
      ?this.#suspendedTurn?.requiredSeats??[this.#suspendedTurn?.actorSeat]:null;
    const requestType=pivotSeats
      ?(pivotSeats.includes(seat)?'pivot':'wait'):this.#phase;
    const hasFaintedReserve=this.#hasFaintedReserve(seat);
    const team=this.#teams[seat].map(p=>{
      // The rule engine must still decide whether the ACTIVE moves are legal.
      // Reserve move slots are only used for switch scoring by this AI.
      const legal=p===active
        ?legalMoveSlots(p,this.#definitions,{hasRevivalTarget:hasFaintedReserve})
        :null;
      return {
        hp:p.hp,maxHP:p.maxHP,status:p.status,speciesId:p.speciesId,
        itemId:p.itemId,abilityId:p.abilityId,
        // Own team information only: reserve Mega candidates are not sent
        // to the opposing AI or any public room view.
        megaCandidate:Boolean(p.mega)&&!p.megaEvolved&&
          p.itemId===p.megaStoneId,
        stats:p.stats?{...p.stats}:p.stats,
        types:p.types?[...p.types]:p.types,
        boosts:p.boosts?{...p.boosts}:p.boosts,
        moves:p.moves.map((slot,index)=>({
          id:slot.id,...(legal?{disabled:legal[index].disabled}:{})
        })),
      };
    });
    const visibleTypes=this.#faintAppearances[1-seat]?.types
      ??foe.illusionTypes??foe.types;
    // A-2: Only facts already published in this seat's Battle.view.revealed
    // may enter the server-only compact CPU input. Use an appearance key,
    // never a hidden roster/instance ID (Illusion can mask identities).
    const publicAppearance=[...this.#publicHistory(1-seat)].reverse()
      .find(row=>row?.active===true);
    const publicObservation=publicAppearance?{
      appearanceId:publicAppearance.publicInstanceId,
      usedMoves:[...(publicAppearance.usedMoves??[])],
      observedItemIds:[...(publicAppearance.observedItemIds??[])],
      observedAbilityIds:[...(publicAppearance.observedAbilityIds??[])],
    }:null;
    return {
      phase:this.#phase,requestType,turn:this.#turn,request:this.#request,
      // The visible arena state is public to both seats. Never expose the
      // opponent's unseen team, moves, items, ability, or exact HP here.
      field:{
        weather:this.#field.weather??null,
        terrain:this.#field.terrain??null,
        trickRoomTurns:this.#field.trickRoomTurns??0,
        sideEffects:[seat,1-seat].map(i=>[
          ...['stealthrock','spikes','toxicspikes','stickyweb']
            .filter(id=>this.#field.sides?.[i]?.[id])
            .map(id=>({id,layers:Number(this.#field.sides[i][id])})),
          ...['tailwind','reflect','lightscreen','auroraveil']
            .filter(id=>this.#field.sides?.[i]?.[id])
            .map(id=>({id,duration:this.#field.sides[i][id].duration})),
        ]),
      },
      self:{
        active:this.#active[seat],team,
        confirmed:Boolean(this.#pending[seat])
          ||Boolean(pivotSeats&&!pivotSeats.includes(seat)),
        needsSwitch:Boolean(pivotSeats?.includes(seat)),
        canSwitch:Boolean(pivotSeats?.includes(seat))||active.hp===0
          ||(!active.volatiles?.twoturnmove&&!active.volatiles?.lockedmove
            &&!this.#isShadowTrapped(seat)),
        canMega:this.#phase==='turn'&&!this.#pending[seat]
          &&!this.#megaUsed[seat]&&active.hp>0&&!!active.mega
          &&!active.megaEvolved&&active.itemId===active.megaStoneId,
      },
      opponent:{
        // Public faint count only: never reveal hidden sets or team identities.
        remaining:this.#teams[1-seat].filter(p=>p.hp>0).length,
        speciesId:this.#faintAppearances[1-seat]?.speciesId
          ??foe.illusionSpeciesId??foe.speciesId,
        types:visibleTypes?[...visibleTypes]:visibleTypes,
        hpPercent:foe.hp===0?0:Math.max(1,Math.ceil(foe.hp/foe.maxHP*100)),
        status:foe.status,
        publicObservation,
        stages:Object.fromEntries(BOOST_STATS.map(stat=>[stat,boostStage(foe,stat)])),
      },
    };
  }
  view(seat) {
    this.#seat(seat);
    return {...this.#viewState(seat),audioEvents:structuredClone(this.#audioEvents),audioSequence:this.#audioSequence,
      presentationSequence:this.#presentationSequence,
      presentationEvents:this.#presentationEvents.map(({before,after,damageFrame,...event})=>structuredClone({
        ...event,...(before ? {before:before[seat]} : {}),...(damageFrame?{damageFrame:damageFrame[seat]}:{}),after:after[seat]}))};
  }
  #viewState(seat) {
    this.#seat(seat);
    const foe = this.#current(1-seat),active=this.#current(seat);
    const hasFaintedReserve=this.#hasFaintedReserve(seat);
    const publicConditions=p=>PUBLIC_BATTLE_CONDITIONS.filter(id=>Boolean(p.volatiles?.[id]));
    // Server-owned remaining turns; no client-side timers or inferred duration.
    const publicConditionDurations=p=>Object.fromEntries(['encore','taunt','perishsong'].flatMap(id=>{
      const duration=p.volatiles?.[id]?.duration;
      return Number.isInteger(duration)&&duration>0?[[id,duration]]:[];
    }));
    const pivotSeats=this.#phase==='pivot' ? this.#suspendedTurn?.requiredSeats ?? [this.#suspendedTurn?.actorSeat] : null;
    const requestType=pivotSeats ? (pivotSeats.includes(seat) ? 'pivot' : 'wait') : this.#phase;
    return structuredClone({ phase: this.#phase, requestType, turn: this.#turn, request: this.#request,
      field: { weather: ({sunnyday:'ひざしがつよい',raindance:'雨',sandstorm:'砂嵐',snowscape:'雪'})[this.#field.weather] ?? 'なし',
        terrain: ({electricterrain:'エレキフィールド',grassyterrain:'グラスフィールド',mistyterrain:'ミストフィールド',psychicterrain:'サイコフィールド'})[this.#field.terrain] ?? 'なし',
        ...(this.#field.weather && this.#field.weatherTurns > 0 ? {weatherDuration:this.#field.weatherTurns} : {}),
        ...(this.#field.terrain && this.#field.terrainTurns > 0 ? {terrainDuration:this.#field.terrainTurns} : {}),
        effects: this.#field.trickRoomTurns ? ['トリックルーム'] : [],
        ...(this.#field.trickRoomTurns > 0 ? {trickRoomDuration:this.#field.trickRoomTurns} : {}),
        sideEffects:[seat,1-seat].map(i=>[
          ...['stealthrock','spikes','toxicspikes','stickyweb'].filter(id=>this.#field.sides?.[i]?.[id]).map(id=>({id,layers:Number(this.#field.sides[i][id])})),
          ...['tailwind','reflect','lightscreen','auroraveil'].filter(id=>this.#field.sides?.[i]?.[id]).map(id=>({id,duration:this.#field.sides[i][id].duration})),
          ...(this.#field.wishes?.[i] ? [{id:'wish'}] : []),
        ]) },
      revealed: this.#publicHistory(1-seat),
      self: { megaUsed:this.#megaUsed[seat],canMega:this.#phase==='turn' && !this.#pending[seat] && !this.#megaUsed[seat] && this.#current(seat).hp>0 && !!this.#current(seat).mega && !this.#current(seat).megaEvolved && this.#current(seat).itemId===this.#current(seat).megaStoneId,needsSwitch:Boolean(pivotSeats?.includes(seat)), team: this.#teams[seat].map(p=>{
        const availability=legalMoveSlots(p,this.#definitions,{hasRevivalTarget:p!==active||hasFaintedReserve});
        const {trueSpeciesName,illusionInstanceId,illusionSpeciesId,illusionSpeciesName,illusionTypes,publicInstanceId,publicIdentityKnown,...visible}=p;
        return {...visible,abilityId:p.abilityId||p.transformedAbilityId,speciesName:trueSpeciesName??p.speciesName,conditions:publicConditions(p),conditionDurations:publicConditionDurations(p),moves:p.moves.map((slot,index)=>{
          const needsRevivalTarget=p===active&&slot.id==='revivalblessing'&&!hasFaintedReserve;
          return {...slot,disabled:availability[index].disabled,disabledReason:needsRevivalTarget?'瀕死の控えがいないため使用できません':availability[index].reason};
        })};
      }), active: this.#active[seat], confirmed: Boolean(this.#pending[seat]) || Boolean(pivotSeats && !pivotSeats.includes(seat)),
        canSwitch:Boolean(pivotSeats?.includes(seat)) || this.#current(seat).hp===0 || !this.#current(seat).volatiles?.twoturnmove && !this.#current(seat).volatiles?.lockedmove && !this.#isShadowTrapped(seat) },
      opponent: { speciesId:this.#faintAppearances[1-seat]?.speciesId??foe.illusionSpeciesId??foe.speciesId, speciesName:this.#faintAppearances[1-seat]?.speciesName??foe.illusionSpeciesName??foe.speciesName, ...((this.#faintAppearances[1-seat]?.types??foe.illusionTypes??foe.types) ? {types:[...(this.#faintAppearances[1-seat]?.types??foe.illusionTypes??foe.types)]} : {}), level: 50,
        hpPercent: foe.hp === 0 ? 0 : Math.max(1, Math.ceil(foe.hp / foe.maxHP * 100)), status: foe.status, atkStage: foe.atkStage, defStage: foe.defStage,
        stages:Object.fromEntries(BOOST_STATS.map(stat=>[stat,boostStage(foe,stat)])),
        conditions:publicConditions(foe),conditionDurations:publicConditionDurations(foe),
        confirmed: Boolean(this.#pending[1-seat]) || Boolean(pivotSeats && !pivotSeats.includes(1-seat)) }, log: this.#viewLog(seat) });
  }
}