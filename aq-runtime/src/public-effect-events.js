import { visibleBoostAbilityCauses as visibleBoostAbilities, visibleBoostItemCauses as visibleBoostItems } from './battle-boost-public-causes.js';
// Pure projection of already-confirmed HP loss. Never apply damage, consume RNG,
// inspect log text, or include an internal Pokemon/party instance ID.
import { moveLabels, statusLabels, abilityLabels, itemLabels } from '../data/catalog/labels-ja.js';

const knownConditions=new Set(['leechseed','curse','partiallytrapped','saltcure','perishsong','confusion','destinybond']);
const knownHazards=new Set(['stealthrock','spikes']);
const knownWeather=new Set(['sandstorm']);

function publicAppearance(pokemon,appearanceOf) {
  if(!pokemon || typeof appearanceOf!=='function')return null;
  const ref=appearanceOf(pokemon);
  return ref && (ref.seat===0 || ref.seat===1)
    && typeof ref.publicInstanceId==='string' && ref.publicInstanceId
    ? {seat:ref.seat,publicInstanceId:ref.publicInstanceId} : null;
}

function knownCause(kind,id) {
  if(typeof id!=='string'||!id)return false;
  if(kind==='ability')return Boolean(abilityLabels[id]);
  if(kind==='item')return Boolean(itemLabels[id]);
  if(kind==='move')return Boolean(moveLabels[id]);
  if(kind==='status')return Boolean(statusLabels[id]);
  if(kind==='condition')return knownConditions.has(id);
  if(kind==='hazard')return knownHazards.has(id);
  if(kind==='weather')return knownWeather.has(id);
  return false;
}

export function projectPublicDamageEvents(effect,{appearanceOf=()=>null,canDisclose=()=>false}={}) {
  if(!(effect?.damage>0))return [];
  const target=publicAppearance(effect.target,appearanceOf);
  // The name is already the battle's public (possibly Illusion-masked) name.
  const targetName=effect.target?.speciesName??'ポケモン';
  const emit=(displayKind,causeKind,causeId,source,{contact=false}={})=>{
    const sourceRef=publicAppearance(source,appearanceOf);
    const known=knownCause(causeKind,causeId);
    const cause=known?{kind:causeKind,id:causeId,...(sourceRef?{source:sourceRef}:{})}:undefined;
    const disclosed=known && sourceRef && (causeKind==='ability'||causeKind==='item')
      && canDisclose({kind:causeKind,id:causeId,source,effect})===true;
    return {
      kind:'damage',
      ...(target?{target}:{}),
      targetName,
      ...(cause?{cause}:{}),
      result:{kind:'hp-loss',displayKind,...(contact?{contact:true}:{})},
      disclosures:disclosed?[{kind:causeKind,id:causeId,subject:sourceRef}]:[],
    };
  };
  if(effect.kind==='contact' && Array.isArray(effect.causes) && effect.causes.length) {
    // A mixed recognized/unrecognized batch keeps the original neutral one-line
    // fallback instead of inventing a partially attributed source.
    if(effect.causes.every(c=>['ability','item'].includes(c.kind)&&knownCause(c.kind,c.id)))
      return effect.causes.map(c=>emit('contact',c.kind,c.id,effect.source,{contact:true}));
  }
  const kind=effect.kind;
  if(kind==='ability')return [emit(kind,'ability',effect.abilityId,
    effect.abilityId==='liquidooze'?effect.source:effect.target)];
  if(kind==='disguise')return [emit(kind,'ability','disguise',effect.target)];
  if(kind==='liquidooze'||kind==='innardsout')
    return [emit(kind,'ability',kind,effect.source)];
  if(kind==='item')return [emit(kind,'item',effect.itemId,effect.target)];
  if(kind==='status')return [emit(kind,'status',effect.status??effect.target?.status,null)];
  if(kind==='hazard')return [emit(kind,'hazard',effect.id,null)];
  if(kind==='weather')return [emit(kind,'weather',effect.id,null)];
  if(kind==='condition')return [emit(kind,'condition',effect.id,null)];
  if(kind==='move'||kind==='selfdestruct'||kind==='recoil'||kind==='crash'||kind==='spikyshield')
    return [emit(kind,'move',kind==='spikyshield'?'spikyshield':effect.moveId,effect.source)];
  return [emit(kind,null,null,null)];
}


// The scheduled delayed effect is now due and its source is known. This is an
// arrival announcement, NOT evidence that the following attack hits or deals
// damage. The original move-use already disclosed the move; do not append
// another usedMoves entry or infer a different active source after switching.
export function projectPublicDelayedMoveArrival({
  source,moveId,moveName,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !source || typeof moveId!=='string' || !moveLabels[moveId]
    || typeof moveName!=='string' || !moveName)return null;
  const sourceRef=publicAppearance(source,appearanceOf);
  return {
    kind:'delayed-move-arrival',
    sourceName:source.speciesName??'ポケモン',
    cause:{kind:'move',id:moveId,...(sourceRef?{source:sourceRef}:{})},
    result:{kind:'due',moveName},
    disclosures:[],
  };
}

// Only the battle's confirmed "used a move" point may call this projector.
// An attempted but prevented move does not disclose. Called moves are separate
// uses; a delayed hit does not count as a second use.
export function projectPublicMoveUse({actor,moveId,moveName},{appearanceOf=()=>null}={}) {
  const subject=publicAppearance(actor,appearanceOf);
  const publicId=typeof moveId==='string'&&moveId?moveId:null;
  return {
    kind:'move-used',
    ...(subject?{actor:subject}:{}),
    actorName:typeof actor?.speciesName==='string'?actor.speciesName:'ポケモン',
    ...(publicId&&subject?{cause:{kind:'move',id:publicId,source:subject}}:{}),
    result:{kind:'announced',moveName:typeof moveName==='string'?moveName:'技'},
    disclosures:publicId&&subject?[{kind:'move',id:publicId,subject}]:[],
  };
}


// Projection of a completed stat-stage change. The battle-moves callback
// carries the actual source object; this boundary retains only its public
// appearance. Neither the current opposing slot nor message text identifies it.
export function projectPublicBoostChange({target,source,change},{appearanceOf=()=>null}={}) {
  const outcome=change?.kind;
  const isOutcome=['blocked','reflected','limit'].includes(outcome);
  if(!target || !change || (!isOutcome && outcome!=null)
    || !['atk','def','spa','spd','spe','accuracy','evasion'].includes(change.stat)
    || ((!isOutcome || outcome==='limit')
      && (!Number.isInteger(change.amount) || change.amount===0)))return null;
  const targetRef=publicAppearance(target,appearanceOf);
  // A blocked/reflected/capped change belongs to the affected holder, never
  // to the attempted move's attacker or an inferred current opposing slot.
  const sourceRef=publicAppearance(isOutcome?target:source,appearanceOf);
  const abilityId=change.causeAbilityId;
  const visibleAbility=typeof abilityId==='string' && Boolean(abilityLabels[abilityId])
    && (visibleBoostAbilities.has(abilityId)
      || abilityId==='intimidate' && typeof change.causeSpeciesName==='string'
        && Boolean(change.causeSpeciesName));
  // The observer can report a positive boost on a different Pokemon (Gooey
  // or Intimidate), but a failed change can only reveal its real defender's
  // confirmed block/reflect ability or its own attempted activation at cap.
  const outcomeAbilityAllowed=!isOutcome
    || (target.abilityId===abilityId && (
      outcome==='blocked' && ['clearbody','mirrorarmor'].includes(abilityId)
      || outcome==='reflected' && abilityId==='mirrorarmor'
      || outcome==='limit' && visibleBoostAbilities.has(abilityId)));
  const abilityVisible=visibleAbility&&outcomeAbilityAllowed;
  const moveId=typeof change.causeMoveId==='string' && moveLabels[change.causeMoveId]
    ?change.causeMoveId:null;
  const itemId=visibleBoostItems.has(change.causeItemId) && itemLabels[change.causeItemId]
    ?change.causeItemId:null;
  // A public block/reflect names the defender's ability as the direct cause.
  // The attempted move is a separate, already confirmed public fact.
  const cause=isOutcome && outcome!=='limit' && abilityVisible
    ?{kind:'ability',id:abilityId} : abilityId==='intimidate' && abilityVisible
      ?{kind:'ability',id:abilityId} : moveId
        ?{kind:'move',id:moveId} : abilityVisible
          ?{kind:'ability',id:abilityId} : itemId
            ?{kind:'item',id:itemId} : null;
  const result={kind:isOutcome?outcome:'stages-changed',stat:change.stat,
    ...(!isOutcome || outcome==='limit'?{amount:change.amount}:{})};
  if(isOutcome) {
    // Block/reflect logs historically show both the attempted move and the
    // blocking ability. A cap uses the same move > ability > item precedence.
    if(moveId)result.causeMoveId=moveId;
    if(abilityVisible)result.causeAbilityId=abilityId;
    if(itemId)result.causeItemId=itemId;
  } else {
    if(cause?.kind==='move')result.causeMoveId=cause.id;
    if(cause?.kind==='ability') {
      result.causeAbilityId=cause.id;
      if(cause.id==='intimidate')result.causeSpeciesName=change.causeSpeciesName;
    }
    if(cause?.kind==='item')result.causeItemId=cause.id;
  }
  return {
    kind:'boost-changed',
    ...(targetRef?{target:targetRef}:{}),
    targetName:typeof target.speciesName==='string'?target.speciesName:'ポケモン',
    // Indirect move effects may not retain the setter. Never invent one.
    ...(cause?{cause:{...cause,...(sourceRef&&cause.kind!=='move'?{source:sourceRef}:{})}}:{}),
    result,
    disclosures:sourceRef && (cause?.kind==='ability'||cause?.kind==='item')
      ?[{kind:cause.kind,id:cause.id,subject:sourceRef}]:[],
  };
}

// Item consumption, transfer, and exchange are already resolved by the engine.
// The event captures only the publicly announced item and the exact public
// appearances at publication time; no current private inventory is inspected.
export function projectPublicHeldItemEvent({holder,item,kind,other=null,otherItem=null},{appearanceOf=()=>null}={}) {
  const holderRef=publicAppearance(holder,appearanceOf);
  const otherRef=publicAppearance(other,appearanceOf);
  // Accept only own label-table entries; inherited Object prototype names
  // (e.g. constructor) are not publicly recognized item IDs.
  const labeled=id=>typeof id==='string' && Object.hasOwn(itemLabels,id)
    && typeof itemLabels[id]==='string' && itemLabels[id].length>0;
  const itemId=labeled(item)?item:null;
  const otherItemId=labeled(otherItem)?otherItem:null;
  // An unmapped nonempty item must not be formatted as an empty inventory.
  // Do not carry the private raw ID into the public event. A successful
  // exchange is public, but the unrecognized item's identity is not.
  const exchangeHasUnmappedItem=kind==='exchange' && (
    Boolean(item) && !itemId || Boolean(otherItem) && !otherItemId);
  if(kind!=='exchange'&&!itemId)return null;
  const disclosures=[];
  const disclose=(id,subject)=>{
    if(id&&subject&&!disclosures.some(entry=>entry.id===id
      && entry.subject.seat===subject.seat
      && entry.subject.publicInstanceId===subject.publicInstanceId))
      disclosures.push({kind:'item',id,subject});
  };
  disclose(itemId,holderRef);
  if(kind==='transfer')disclose(itemId,otherRef);
  if(kind==='exchange')disclose(otherItemId,otherRef);
  return {
    kind:'held-item',
    holderName:holder?.speciesName??'ポケモン',
    ...(other?{otherName:other.speciesName??'ポケモン'}:{}),
    ...(holderRef?{holder:holderRef}:{}),
    ...(otherRef?{other:otherRef}:{}),
    ...(['activated','healed','statreset'].includes(kind)&&itemId&&holderRef
      ?{cause:{kind:'item',id:itemId,source:holderRef}}:{}),
    result:{kind,itemId,...(kind==='exchange'?{otherItemId,
      ...(exchangeHasUnmappedItem?{unmappedItem:true}:{})}:{})},
    disclosures,
  };
}

// Mummy's replacement and Wandering Spirit's exchange disclose abilities
// that the engine explicitly announced. Frozen before-change IDs are captured
// by the mechanics callback; they are never reverse-engineered from log text.
export function projectPublicAbilityLifecycle(fact,{appearanceOf=()=>null}={}) {
  if(fact?.confirmed!==true)return null;
  const disclosures=[];
  const add=(id,mon)=>{
    const subject=publicAppearance(mon,appearanceOf);
    if(!subject || !abilityLabels[id])return;
    if(!disclosures.some(d=>d.id===id && d.subject.seat===subject.seat
      && d.subject.publicInstanceId===subject.publicInstanceId))
      disclosures.push({kind:'ability',id,subject});
  };
  const publicId=id=>typeof id==='string' && abilityLabels[id]?id:null;
  if(fact.kind==='replaced' && fact.holder && fact.source
    && fact.newAbilityId==='mummy') {
    const holder=publicAppearance(fact.holder,appearanceOf);
    const source=publicAppearance(fact.source,appearanceOf);
    add('mummy',fact.source);
    add(fact.oldAbilityId,fact.holder);
    add('mummy',fact.holder);
    return {
      kind:'ability-lifecycle',
      holderName:fact.holder.speciesName??'ポケモン',
      sourceName:fact.source.speciesName??'ポケモン',
      ...(holder?{holder}:{}),
      cause:{kind:'ability',id:'mummy',...(source?{source}:{})},
      result:{kind:'replaced',beforeId:publicId(fact.oldAbilityId),afterId:'mummy'},
      disclosures,
    };
  }
  if(fact.kind==='swapped' && fact.first && fact.second
    && fact.secondAbilityId==='wanderingspirit') {
    const source=publicAppearance(fact.second,appearanceOf);
    const first=publicAppearance(fact.first,appearanceOf);
    const second=source;
    add(fact.firstAbilityId,fact.first);
    add('wanderingspirit',fact.second);
    add('wanderingspirit',fact.first);
    add(fact.firstAbilityId,fact.second);
    return {
      kind:'ability-lifecycle',
      firstName:fact.first.speciesName??'ポケモン',
      secondName:fact.second.speciesName??'ポケモン',
      ...(first?{first}:{}),
      ...(second?{second}:{}),
      cause:{kind:'ability',id:'wanderingspirit',...(source?{source}:{})},
      result:{kind:'swapped',firstBeforeId:publicId(fact.firstAbilityId),
        secondBeforeId:'wanderingspirit'},
      disclosures,
    };
  }
  return null;
}

// The combat hit callback reports a completed, publicly announced ability
// activation. Its frozen pre-change ID belongs to the defender, not to the
// attacker that dealt the hit; the formatter cannot grant the disclosure.
export function projectPublicDamagingHitAbility({holder,abilityId,confirmed},
  {appearanceOf=()=>null}={}) {
  if(confirmed!==true || !holder || typeof abilityId!=='string'
    || !abilityLabels[abilityId])return null;
  const subject=publicAppearance(holder,appearanceOf);
  return {
    kind:'ability-activated',
    holderName:holder.speciesName??'ポケモン',
    ...(subject?{holder:subject}:{}),
    cause:{kind:'ability',id:abilityId,...(subject?{source:subject}:{})},
    result:{kind:'announced'},
    disclosures:subject?[{kind:'ability',id:abilityId,subject}]:[],
  };
}

// Entry ability notifications are emitted only after a confirmed effect.
// A Frisk event discloses two independently attributed facts: the actual
// activating ability and the other Pokémon's publicly announced held item.
export function projectPublicEntryAbilityEvent({
  holder,abilityId,confirmed,friskTarget=null,friskedItemId=null,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !holder || typeof abilityId!=='string' || !abilityLabels[abilityId])return null;
  const frisk=abilityId==='frisk';
  if(frisk && !friskTarget)return null;
  if(!frisk && (friskTarget || friskedItemId))return null;
  const holderRef=publicAppearance(holder,appearanceOf);
  const friskRef=frisk?publicAppearance(friskTarget,appearanceOf):null;
  // An actual Frisk activation discloses its ability even if the confirmed
  // held-item ID cannot safely be rendered as a known public item.
  const publicFriskItem=frisk && typeof friskedItemId==='string'
    && Boolean(itemLabels[friskedItemId]);
  return {
    kind:'entry-ability',
    holderName:holder.speciesName??'ポケモン',
    ...(holderRef?{holder:holderRef}:{}),
    ...(publicFriskItem?{friskTargetName:friskTarget.speciesName??'ポケモン'}:{}),
    cause:{kind:'ability',id:abilityId,...(holderRef?{source:holderRef}:{})},
    result:{kind:publicFriskItem?'frisked':'announced',
      ...(publicFriskItem?{itemId:friskedItemId}:{})},
    disclosures:[
      ...(holderRef?[{kind:'ability',id:abilityId,subject:holderRef}]:[]),
      ...(publicFriskItem&&friskRef?[{kind:'item',id:friskedItemId,subject:friskRef}]:[]),
    ],
  };
}

// Air Balloon is revealed only after an actual, surviving entry. Its public
// appearance is fixed at announcement time, including an Illusion mask.
export function projectPublicEntryItemReveal({holder,itemId,confirmed},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !holder || itemId!=='airballoon')return null;
  const subject=publicAppearance(holder,appearanceOf);
  return {
    kind:'entry-item-revealed',
    holderName:holder.speciesName??'ポケモン',
    ...(subject?{holder:subject}:{}),
    cause:{kind:'item',id:'airballoon',...(subject?{source:subject}:{})},
    result:{kind:'floating',itemId:'airballoon'},
    disclosures:subject?[{kind:'item',id:'airballoon',subject}]:[],
  };
}

// Move-caused status transitions are published only after the final status
// is committed. A move announcement is public, but it does not authorize
// disclosure of a private ability, item, or the actor's party identity.
export function projectPublicMoveStatusEvent({
  target,moveId,moveName,statusId,change,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !target || typeof moveId!=='string' || !moveLabels[moveId]
    || typeof moveName!=='string' || !moveName
    || !['applied','cleared'].includes(change))return null;
  if(change==='applied' && !statusLabels[statusId])return null;
  if(change==='cleared' && statusId!=='')return null;
  const targetRef=publicAppearance(target,appearanceOf);
  return {
    kind:'status-move',
    ...(targetRef?{target:targetRef}:{}),
    targetName:target.speciesName??'ポケモン',
    cause:{kind:'move',id:moveId},
    result:{kind:change,...(change==='applied'?{statusId}:{}),moveName},
    disclosures:[],
  };
}

// Pain Split's resolved HP redistribution is not attack damage. Publish
// its actual negative HP delta without invoking any damage side effects or
// guessing the other Pokemon's private health.
export function projectPublicMoveHpLoss({target,moveId,moveName,delta,confirmed},
  {appearanceOf=()=>null}={}) {
  if(confirmed!==true || !target || !Number.isFinite(delta) || delta>=0
    || typeof moveId!=='string' || !moveLabels[moveId]
    || typeof moveName!=='string' || !moveName)return null;
  const subject=publicAppearance(target,appearanceOf);
  return {
    kind:'hp-loss-move',
    targetName:target.speciesName??'ポケモン',
    ...(subject?{target:subject}:{}),
    cause:{kind:'move',id:moveId},
    result:{kind:'hp-loss',moveName},
    disclosures:[],
  };
}

// This projector runs only after the final status has changed. The active-only
// appearance resolver preserves the pre-migration disclosure boundary.
export function projectPublicStatusAbilityEvent({
  target,source,statusId,abilityId,variant,moveId=null,moveName=null,
},{appearanceOf=()=>null}={}) {
  if(!target || !source || !statusLabels[statusId] || !abilityLabels[abilityId])return null;
  if(variant==='synchronize' && abilityId!=='synchronize')return null;
  if(variant==='magicbounce' && (abilityId!=='magicbounce' || !moveId || !moveName))return null;
  if(!['synchronize','magicbounce','contact'].includes(variant))return null;
  const targetRef=publicAppearance(target,appearanceOf);
  const sourceRef=publicAppearance(source,appearanceOf);
  return {
    kind:'status',
    ...(targetRef?{target:targetRef}:{}),
    targetName:target.speciesName??'ポケモン',
    cause:{kind:'ability',id:abilityId,...(sourceRef?{source:sourceRef}:{})},
    result:{kind:'applied',statusId,variant,sourceName:source.speciesName??'ポケモン',
      ...(moveId&&moveName?{moveId,moveName}:{})},
    disclosures:sourceRef?[{kind:'ability',id:abilityId,subject:sourceRef}]:[],
  };
}


// Only confirmed Cursed Body Disable and Oblivious condition cures use this
// transitional event. The cause holder is captured by the effect callback,
// never inferred from a log string or the opposing active slot.
export function projectPublicConditionAbilityEvent({
  target,source,conditionId,abilityId,ended=false,confirmed=false,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !target || !abilityLabels[abilityId])return null;
  const cursedBody=abilityId==='cursedbody' && conditionId==='disable' && !ended;
  const oblivious=abilityId==='oblivious' && ['taunt','attract'].includes(conditionId) && ended;
  // Own Tempo is disclosed only after its own confirmed confusion cure.
  const ownTempo=abilityId==='owntempo' && conditionId==='confusion' && ended && source===target;
  if(!cursedBody&&!oblivious&&!ownTempo)return null;
  const targetRef=publicAppearance(target,appearanceOf);
  const sourceRef=publicAppearance(source,appearanceOf);
  return {
    kind:'condition-ability',
    ...(targetRef?{target:targetRef}:{}),
    targetName:target.speciesName??'ポケモン',
    cause:{kind:'ability',id:abilityId,...(sourceRef?{source:sourceRef}:{})},
    result:{kind:ended?'ended':'started',conditionId,ended},
    disclosures:sourceRef?[{kind:'ability',id:abilityId,subject:sourceRef}]:[],
  };
}

// The same confirmed HP gain has one projection rule regardless of whether
// a move or a publicly activated ability was its source. Keep the legacy
// event.kind and Japanese formatters until the UI consumer can migrate.
// A move's amount is validated here; ability callbacks only arrive after
// positive healing and must not infer a private ability from current state.
function projectPublicConfirmedHealing({target,causeKind,causeId,moveName,amount,confirmed},
  {appearanceOf=()=>null}={}) {
  if(confirmed!==true || !target)return null;
  if(causeKind==='move') {
    if(!(amount>0) || !Number.isFinite(amount)
      || typeof causeId!=='string' || !moveLabels[causeId]
      || typeof moveName!=='string' || !moveName)return null;
  } else if(causeKind==='ability') {
    if(typeof causeId!=='string' || !abilityLabels[causeId])return null;
  } else return null;
  const targetRef=publicAppearance(target,appearanceOf);
  const abilityCause=causeKind==='ability';
  return {
    kind:abilityCause?'heal':'heal-move',
    ...(targetRef?{target:targetRef}:{}),
    targetName:target.speciesName??'ポケモン',
    cause:{kind:causeKind,id:causeId,...(abilityCause&&targetRef?{source:targetRef}:{})},
    result:{kind:'hp-gain',...(!abilityCause?{moveName}:{})},
    disclosures:abilityCause&&targetRef?[{kind:'ability',id:causeId,subject:targetRef}]:[],
  };
}

export function projectPublicMoveHealEvent({
  target,moveId,moveName,amount,confirmed,
},{appearanceOf=()=>null}={}) {
  return projectPublicConfirmedHealing({
    target,causeKind:'move',causeId:moveId,moveName,amount,confirmed,
  },{appearanceOf});
}

export function projectPublicAbilityHeal({target,abilityId,confirmed},{appearanceOf=()=>null}={}) {
  return projectPublicConfirmedHealing({
    target,causeKind:'ability',causeId:abilityId,confirmed,
  },{appearanceOf});
}


// After Update has actually cured a major status, the activating ability
// is already publicly announced. Freeze the current appearance at that point;
// neither a previous log line nor the player's current private item can disclose it.
export function projectPublicAbilityStatusCure({
  holder,abilityId,previousStatus,afterStatus,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !holder || !abilityLabels[abilityId]
    || !statusLabels[previousStatus] || afterStatus!=='')return null;
  const subject=publicAppearance(holder,appearanceOf);
  return {
    kind:'status-cured-ability',
    holderName:holder.speciesName??'ポケモン',
    ...(subject?{holder:subject}:{}),
    cause:{kind:'ability',id:abilityId,...(subject?{source:subject}:{})},
    result:{kind:'status-cleared'},
    disclosures:subject?[{kind:'ability',id:abilityId,subject}]:[],
  };
}

// Harvest's successful restoration is one confirmed public event with two
// independently disclosed facts. The restored berry is not the ability cause.
export function projectPublicHarvestRestoration({holder,berryId,confirmed},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !holder || typeof berryId!=='string' || !berryId)return null;
  const subject=publicAppearance(holder,appearanceOf);
  const knownBerry=Boolean(itemLabels[berryId]);
  return {
    kind:'harvest-restored',
    holderName:holder.speciesName??'ポケモン',
    cause:{kind:'ability',id:'harvest',...(subject?{source:subject}:{})},
    result:{kind:'berry-restored',itemId:knownBerry?berryId:null},
    disclosures:subject?[
      {kind:'ability',id:'harvest',subject},
      ...(knownBerry?[{kind:'item',id:berryId,subject}]:[]),
    ]:[],
  };
}

// The field transition is already committed by the engine. Attribute only a
// matching public move or a named confirmed setter; do not guess who set an
// expiring or overwritten condition from today's active Pokemon.
const visibleFieldConditions={
  weather:new Set(['','sunnyday','raindance','sandstorm','snowscape']),
  terrain:new Set(['','electricterrain','grassyterrain','mistyterrain','psychicterrain']),
};
export function projectPublicFieldChange({
  type,before,after,sourceMoveId=null,sourceMoveName=null,sourceAbility=null,
},{appearanceOf=()=>null}={}) {
  if(!visibleFieldConditions[type]?.has(before)||!visibleFieldConditions[type]?.has(after)||before===after)return null;
  const setter=after && sourceMoveId===after && typeof sourceMoveName==='string'
    ?{kind:'move',id:sourceMoveId}:null;
  const ability=after && sourceAbility?.[type]===after && sourceAbility?.speciesName
    && abilityLabels[sourceAbility.abilityId] ?sourceAbility:null;
  const ref=ability?publicAppearance(ability.pokemon,appearanceOf):null;
  const causedBySpinner=type==='terrain' && before && !after && sourceMoveId==='icespinner';
  const cause=setter ?? (ability?{
    kind:'ability',id:ability.abilityId,...(ref?{source:ref}:{})
  }:causedBySpinner?{kind:'move',id:'icespinner'}:null);
  return {
    kind:'field-change',
    ...(cause?{cause}:{}),
    result:{type,before,after,
      ...(setter?{sourceMoveName}:{}),
      ...(ability?{sourceName:ability.speciesName}:{}),
      ...(causedBySpinner?{sourceMoveName:sourceMoveName??moveLabels.icespinner??'アイススピナー'}:{}),
    },
    disclosures:ability && ref?[{kind:'ability',id:ability.abilityId,subject:ref}]:[],
  };
}


// A successful Natural Cure switch-out is public after the outgoing
// appearance has been frozen, but before the authored ability is restored.
// The caller supplies the confirmed transition; no log parsing or live
// opponent ability lookup may authorize the disclosure.
export function projectPublicNaturalCure({
  holder,abilityId,previousStatus,afterStatus,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !holder || abilityId!=='naturalcure'
    || typeof previousStatus!=='string' || !previousStatus || afterStatus!=='')return null;
  const subject=publicAppearance(holder,appearanceOf);
  return {
    kind:'switchout-ability-cure',
    holderName:holder.speciesName??'ポケモン',
    ...(subject?{holder:subject}:{}),
    cause:{kind:'ability',id:'naturalcure',...(subject?{source:subject}:{})},
    result:{kind:'status-cleared'},
    disclosures:subject?[{kind:'ability',id:'naturalcure',subject}]:[],
  };
}


// Project confirmed ability-caused move outcomes from one server-owned result.
// A failed move alone never permits inferring the defender's private ability.
// Keep legacy blocked event.kind while the UI still consumes string logs.
const publicMoveAbilityOutcomeCauses=Object.freeze({
  blocked:new Set(['damp','goodasgold']),
  absorbed:new Set(['flashfire']),
});
export function projectPublicAbilityMoveOutcome({
  holder,holderSeat,abilityId,moveId,moveName,moveType=null,charged=false,
  outcome,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !holder || holder.abilityId!==abilityId
    || !publicMoveAbilityOutcomeCauses[outcome]?.has(abilityId)
    || typeof moveId!=='string' || !moveId)return null;
  if(outcome==='blocked' && ![0,1].includes(holderSeat))return null;
  // This branch is reached only after the engine has committed Flash Fire's
  // actual immunity/charge. A prior charge or ordinary immunity is insufficient.
  if(outcome==='absorbed' && (moveType!=='Fire' || charged!==true))return null;
  const subject=publicAppearance(holder,appearanceOf);
  return {
    kind:outcome==='blocked'?'move-blocked-ability':'move-absorbed-ability',
    ...(outcome==='blocked'
      ?{holderSeat}
      :{holderName:holder.illusionSpeciesName??holder.speciesName??'ポケモン'}),
    ...(subject?{holder:subject}:{}),
    cause:{kind:'ability',id:abilityId,...(subject?{source:subject}:{})},
    result:{
      kind:outcome==='blocked'?'blocked-move':'absorbed-move',
      moveId,moveName:typeof moveName==='string'?moveName:'技',
    },
    disclosures:subject?[{kind:'ability',id:abilityId,subject}]:[],
  };
}

// Compatibility entrypoint for already-migrated Damp / Good as Gold callers.
export function projectPublicAbilityMoveBlock(input,context) {
  return projectPublicAbilityMoveOutcome({...input,outcome:'blocked'},context);
}


// Confirmed status changes without publicly attributable cause must still be
// projected from the same resolved-fact boundary as named move/ability cures.
// No current ability, item, move selection or Japanese log is consulted to
// infer a disclosure.
export function projectPublicNeutralStatusChange({
  target,before,after,confirmed,causeConditionId=null,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !target || before===after
    || (before!=='' && !statusLabels[before])
    || (after!=='' && !statusLabels[after]))return null;
  const subject=publicAppearance(target,appearanceOf);
  const toxicSpikes=(causeConditionId==='toxicspikes' && ['psn','tox'].includes(after));
  return {
    kind:'status-transition',
    targetName:target.speciesName??'ポケモン',
    ...(subject?{target:subject}:{}),
    ...(toxicSpikes?{cause:{kind:'condition',id:'toxicspikes'}}:{}),
    result:{kind:after?'applied':'cleared',statusId:after},
    disclosures:[],
  };
}

// Neutral move outcomes reveal only the move that was already announced.
// "Immune"/"missed"/"failed" cannot authorize guesses about a hidden
// defending ability or held item. Keep existing Japanese messages and ordering.
export function projectPublicNeutralMoveOutcome({
  moveId,moveName,outcome,reason='generic',confirmed,
}={}) {
  if(confirmed!==true || typeof moveId!=='string' || !moveId
    || (typeof moveName!=='string' || !moveName)
    || !['immune','missed','failed'].includes(outcome)
    || (outcome!=='failed' && reason!=='generic')
    || !['generic','trick','status-fail'].includes(reason))return null;
  return {
    kind:'move-result',
    cause:{kind:'move',id:moveId},
    result:{kind:outcome,moveName,...(outcome==='failed'?{reason}:{})},
    disclosures:[],
  };
}


// Residual and delayed HP gains without an activating item or ability remain
// anonymous to their setter. The engine supplies a confirmed positive delta
// and one accepted condition ID; never derive a source from log text or the
// currently active opponent.
const publicRecoveryConditions=new Set(['wish','grassyterrain','leechseed']);
export function projectPublicConfirmedConditionHeal({
  target,conditionId,amount,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !target || !publicRecoveryConditions.has(conditionId)
    || !Number.isFinite(amount) || amount<=0)return null;
  const subject=publicAppearance(target,appearanceOf);
  return {
    kind:'heal-condition',
    targetName:target.speciesName??'ポケモン',
    ...(subject?{target:subject}:{}),
    cause:{kind:conditionId==='grassyterrain'?'field':'condition',id:conditionId},
    result:{kind:'hp-gain',conditionId},
    disclosures:[],
  };
}


// The guard's successful block callback is already committed. A protected
// move does not prove a secret ability or item and never adds a disclosure.
// Guard IDs must be public guard moves; unknown guards use neutral wording.
const publicGuardMoves=new Set(['protect','kingsshield','silktrap','spikyshield']);
export function projectPublicConfirmedGuardBlock({
  holder,moveId,moveName,guardId,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !holder || typeof moveId!=='string'||!moveId
    || typeof moveName!=='string'||!moveName)return null;
  const target=publicAppearance(holder,appearanceOf);
  return {
    kind:'move-result',
    targetName:holder.illusionSpeciesName??holder.speciesName??'ポケモン',
    ...(target?{target}:{}),
    cause:{kind:'move',id:moveId},
    result:{kind:'guarded',moveName,
      ...(publicGuardMoves.has(guardId)?{guardId}:{}),
    },
    disclosures:[],
  };
}


// The existing condition observer fires after the mutable volatile was
// started/ended. Only its confirmed transition and explicit publicly used
// move cause may be projected; hidden abilities, held items, and the current
// move selection must never be inferred here.
const publicConditionIds=new Set([
  'taunt','encore','disable','healblock','throatchop','confusion',
  'saltcure','partiallytrapped','trapped','jawlock','octolock',
  'substitute','yawn','leechseed','curse','perishsong','destinybond',
  'recharge','glaiverush','lockedmove',
]);
const publicConditionMoveCauses=new Set([
  'taunt','encore','disable','throatchop','saltcure','glaiverush',
  'leechseed','yawn','octolock','perishsong','curse','jawlock',
  'firespin','infestation','whirlpool','spiritshackle',
  'outrage','ragingfury',
]);
export function projectPublicConfirmedConditionTransition({
  target,id,ended=false,causeMoveId=null,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !target || !publicConditionIds.has(id)
    || typeof ended!=='boolean')return null;
  const subject=publicAppearance(target,appearanceOf);
  const publicMoveId=!ended && publicConditionMoveCauses.has(causeMoveId)
    && moveLabels[causeMoveId]?causeMoveId:null;
  return {
    kind:'condition-transition',
    targetName:target.speciesName??'ポケモン',
    ...(subject?{target:subject}:{}),
    ...(publicMoveId?{cause:{kind:'move',id:publicMoveId}}:{}),
    result:{id,ended,...(publicMoveId?{causeMoveId:publicMoveId}:{})},
    disclosures:[],
  };
}


// Reset/copy callbacks describe a completed boost transition, but not a new
// ability/item activation. Reuse the existing boost formatter and never infer
// causes from mutable inventories or old text.
export function projectPublicNeutralBoostNotice({
  target,kind,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !target || !['reset','copy'].includes(kind))return null;
  const subject=publicAppearance(target,appearanceOf);
  return {
    kind:'boost-changed',
    targetName:target.speciesName??'ポケモン',
    ...(subject?{target:subject}:{}),
    result:{kind},
    disclosures:[],
  };
}



// Waking and thawing are already completed by beforeMoveStatus before an
// action is announced; a hidden held berry must never be inferred from them.
export function projectPublicNaturalStatusEnd({
  target,before,after,confirmed,
},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !target || !['slp','frz'].includes(before) || after!=='')return null;
  const subject=publicAppearance(target,appearanceOf);
  return {
    kind:'status-transition',
    targetName:target.speciesName??'ポケモン',
    ...(subject?{target:subject}:{}),
    result:{kind:'natural-end',previousStatus:before,statusId:''},
    disclosures:[],
  };
}

// An action denied before move announcement has no publicly used move ID.
// The cause of a Choice lock remains owner-private: even the result reason
// must not distinguish that held item in the opponent's event metadata.
const publicUnableReasons=new Set([
  'slp','frz','par','flinch','confusion','attract','taunt','disable',
  'healblock','throatchop','choicelock','recharge',
]);
export function projectPublicUnableToAct({actor,reason,confirmed},{appearanceOf=()=>null}={}) {
  if(confirmed!==true || !actor || !publicUnableReasons.has(reason))return null;
  const subject=publicAppearance(actor,appearanceOf);
  return {
    kind:'action-result',
    actorName:actor.speciesName??'ポケモン',
    ...(subject?{actor:subject}:{}),
    result:{kind:'unable',reason:reason==='choicelock'?'blocked':reason},
    disclosures:[],
  };
}
