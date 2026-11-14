import { visibleBoostMoveCauses, visibleBoostAbilityCauses, visibleBoostItemCauses } from './battle-boost-public-causes.js';
import { abilityLabels, moveLabels, itemLabels, statusLabels } from '../data/catalog/labels-ja.js';
// Public text only: callers report an already resolved effect, never predict it.
const observers=new WeakMap();
export function observeEffectEvents(pokemon,observer) {
  if(typeof observer==='function')observers.set(pokemon,observer);
  else observers.delete(pokemon);
}
export function notifyEffectEvent(pokemon,event) { observers.get(pokemon)?.(event); }
export function endCondition(pokemon,id,{causeAbilityId=null}={}) {
  if(!pokemon.volatiles?.[id])return false;
  delete pokemon.volatiles[id];
  notifyEffectEvent(pokemon,{id,ended:true,...(causeAbilityId==='oblivious'?{causeAbilityId}: {})});
  return true;
}
const observedConditions=new Set(['taunt','encore','disable','healblock','throatchop','confusion']);
export function snapshotConditions(pokemon) {
  return Object.fromEntries(Object.keys(conditionNames).filter(id=>!observedConditions.has(id)).map(id=>[id,Boolean(pokemon.volatiles?.[id])]));
}
export function notifyConditionChanges(pokemon,before,{causeMoveId=null}={}) {
  for(const [id,present] of Object.entries(before))if(present!==Boolean(pokemon.volatiles?.[id]))
    notifyEffectEvent(pokemon,{id,ended:present,
      ...(!present && causeMoveId && (id===causeMoveId || id==='partiallytrapped' || (id==='trapped' && causeMoveId==='spiritshackle') || (id==='lockedmove' && ['outrage','ragingfury'].includes(causeMoveId))) ? {causeMoveId} : {})});
}
export const statNames={atk:'攻撃',def:'防御',spa:'特攻',spd:'特防',spe:'素早さ',accuracy:'命中率',evasion:'回避率'};
export const conditionNames={taunt:'ちょうはつ',encore:'アンコール',disable:'かなしばり',healblock:'かいふくふうじ',throatchop:'じごくづき',confusion:'こんらん',saltcure:'しおづけ',partiallytrapped:'しめつけ',trapped:'交代できない状態',jawlock:'くらいつく',octolock:'たこがため',substitute:'みがわり',yawn:'ねむけ',leechseed:'やどりぎのタネ',curse:'のろい',perishsong:'ほろびのうた',destinybond:'みちづれ',recharge:'反動で休む状態',glaiverush:'きょけんとつげきの隙',lockedmove:'技を出し続ける状態'};
export function boostMessage(pokemon,{stat,amount,kind,causeAbilityId,causeSpeciesName,causeMoveId,causeItemId}) {
  const name=pokemon.speciesName;
  const moveCause=visibleBoostMoveCauses.has(causeMoveId)?`（${moveLabels[causeMoveId]}）`:'';
  const publicItemCause=visibleBoostItemCauses.has(causeItemId)?`（${itemLabels[causeItemId]}）`:'';
  const publicAbilityCause=visibleBoostAbilityCauses.has(causeAbilityId)?`（${abilityLabels[causeAbilityId]}）`:'';
  if(kind==='reset')return `${name} の能力変化が元に戻った！`;
  if(kind==='copy')return `${name} は能力変化を引き継いだ！`;
  if(kind==='reflected')return `${name} は ${statNames[stat]}の低下をはね返した！${moveCause}${publicAbilityCause}`;
  if(kind==='blocked')return `${name} の ${statNames[stat]}は下がらなかった！${moveCause}${publicAbilityCause}`;
  if(kind==='limit')return `${name} の ${statNames[stat]}は これ以上${amount>0?'上がらない':'下がらない'}！${moveCause||publicAbilityCause||publicItemCause}`;
  const direction=amount>0 ? Math.abs(amount)>=2?'ぐーんと上がった':'上がった' : Math.abs(amount)>=2?'がくっと下がった':'下がった';
  // Keep confirmed public Intimidate, Moxie, Speed Boost, Moody, Stamina, Weak Armor, Justified, Thermal Exchange, Defiant, Competitive, Guard Dog, Steadfast, Sticky Web and Octolock causes; other
  // stat changes remain neutral rather than inferring hidden sources.
  const cause=causeAbilityId==='intimidate' && causeSpeciesName
    ? `（${causeSpeciesName}の ${abilityLabels.intimidate}）` : moveCause||publicAbilityCause||publicItemCause;
  return `${name} の ${statNames[stat]}が ${direction}！${cause}`;
}
export function conditionMessage(pokemon,{id,ended=false,causeMoveId=null,causeAbilityId=null}) {
  const label=conditionNames[id];
  // Only a resolved move can explicitly identify its own newly applied restriction.
  const moveCause=!ended && causeMoveId===id && ['taunt','encore','disable','throatchop','saltcure','glaiverush','leechseed','yawn','octolock','perishsong','curse'].includes(id) && moveLabels[id]
    ? `（${moveLabels[id]}）` : '';
  const abilityCause=!ended && id==='disable' && causeAbilityId==='cursedbody'
    ? `（${abilityLabels.cursedbody}）` : '';
  // A trapping move's name is public only after its volatile condition starts.
  // The fixed runtime catalog connects these three direct trapping moves.
  const trappingCause=!ended
    && ((id==='partiallytrapped' && ['firespin','infestation','whirlpool'].includes(causeMoveId))
      || (id==='trapped' && causeMoveId==='spiritshackle')) && moveLabels[causeMoveId]
    ? `（${moveLabels[causeMoveId]}）` : '';
  // The rampage move was publicly used before its self-lock was confirmed.
  const lockedMoveCause=!ended && id==='lockedmove' && ['outrage','ragingfury'].includes(causeMoveId) && moveLabels[causeMoveId]
    ? `（${moveLabels[causeMoveId]}）` : '';
  // Locked rampage moves are already public when their self-confusion resolves.
  if(!ended && id==='jawlock' && causeMoveId==='jawlock' && moveLabels.jawlock)
    return `${pokemon.speciesName} は ${moveLabels.jawlock}で 交代できなくなった！`;
  if(!ended && id==='confusion' && causeMoveId && moveLabels[causeMoveId])
    return `${pokemon.speciesName} は ${moveLabels[causeMoveId]}で こんらんした！`;
  if(!ended && id==='taunt')return `${pokemon.speciesName} は ちょうはつを受けた！${moveCause}`;
  if(!ended && id==='encore')return `${pokemon.speciesName} は アンコールされた！${moveCause}`;
  // Attract is not a timed condition in the shared lookup, but an observed cure
  // is public and should not disappear from the event log.
  if(ended && id==='attract')return `${pokemon.speciesName} の メロメロが解けた！${causeAbilityId==='oblivious'?`（${abilityLabels.oblivious}）`:''}`;
  return label ? `${pokemon.speciesName} の ${label}${ended?'が解けた':'が始まった'}！${moveCause||abilityCause||trappingCause||lockedMoveCause}${ended&&causeAbilityId==='oblivious'?`（${abilityLabels.oblivious}）`:''}` : '';
}


// Japanese text is a projection of the already-public move-use event. Whether
// a move is disclosed to an appearance is decided elsewhere, not by this text.
// Arrival notification uses the frozen confirmed event, not a fresh move
// declaration. It must not update public knowledge a second time.
export function formatPublicDelayedMoveArrival(event) {
  if(event?.kind!=='delayed-move-arrival' || event.result?.kind!=='due')return [];
  return [`${event.sourceName??'ポケモン'} の ${event.result.moveName}！`];
}

export function formatPublicMoveMessages(event) {
  if(event?.kind!=='move-used')return [];
  return [String(event.actorName??'ポケモン')+' の '+String(event.result?.moveName??'技')+'！'];
}


// Text subscribes to the same confirmed boost event as public knowledge.
// An empty or different formatter must never change which cause is disclosed.
export function formatPublicBoostMessages(event) {
  if(event?.kind!=='boost-changed')return [];
  return [boostMessage({speciesName:event.targetName??'ポケモン'},event.result)];
}

// A formatter of an already-public item event. Disclosure is decided before
// this function runs, and an empty message must not revoke the observation.
export function formatPublicHeldItemMessages(event) {
  if(event?.kind!=='held-item')return [];
  const {kind,itemId,otherItemId}=event.result??{};
  const holderName=event.holderName??'ポケモン';
  const name=itemLabels[itemId];
  // A mapped empty inventory is different from an unmapped nonempty item.
  // A confirmed exchange may be announced without guessing a secret item ID.
  if(kind==='exchange' && event.result?.unmappedItem)
    return [`${holderName} と ${event.otherName??'ポケモン'} は持ち物を入れ替えた！`];
  if(kind==='exchange')return [`${holderName} の持ち物は ${name??'なし'}、${event.otherName??'ポケモン'} の持ち物は ${itemLabels[otherItemId]??'なし'}になった！`];
  if(!name)return [];
  if(kind==='transfer'&&event.otherName)return [`${event.otherName} は ${holderName}から ${name}を受け取った！`];
  if(kind==='removed')return [`${holderName} の ${name}は はたきおとされた！`];
  if(kind==='healed')return [`${holderName} は ${name}で HPを回復した！`];
  if(kind==='statreset')return [`${holderName} は ${name}で 下がった能力を元に戻した！`];
  if(kind==='recycled')return [`${holderName} は ${name}を リサイクルした！`];
  if(kind==='activated')return [`${holderName} の ${name}が発動した！`];
  return [`${holderName} は ${name}を使った。`];
}

// The lifecycle names and both before-change IDs are confirmed by mechanics.
// A formatter never looks at the mutable current abilities of either holder.
export function formatPublicAbilityLifecycle(event) {
  if(event?.kind!=='ability-lifecycle')return [];
  const ability=id=>abilityLabels[id]??'とくせい';
  const result=event.result??{};
  if(result.kind==='replaced' && event.cause?.id==='mummy')return [
    `${event.sourceName??'ポケモン'} の ${ability('mummy')}！`,
    `${event.holderName??'ポケモン'} の ${ability(result.beforeId)} は ${ability(result.afterId)} になった！`,
  ];
  if(result.kind==='swapped' && event.cause?.id==='wanderingspirit')return [
    `${event.secondName??'ポケモン'} の ${ability('wanderingspirit')}！ ${event.firstName??'ポケモン'} の ${ability(result.firstBeforeId)} と ${event.secondName??'ポケモン'} の ${ability(result.secondBeforeId)} を入れ替えた！`,
  ];
  return [];
}

// One already-confirmed activation is rendered without re-reading the
// current ability: contact can exchange the holder's ability in the same hit.
export function formatPublicDamagingHitAbility(event) {
  if(event?.kind!=='ability-activated' || event.result?.kind!=='announced')return [];
  const label=abilityLabels[event.cause?.id];
  return label?[ `${event.holderName??'ポケモン'} の ${label}！` ]:[];
}

// A confirmed entry ability and the Frisk-observed item are independently
// disclosed by the same event, whether or not either text renders.
export function formatPublicEntryAbilityMessages(event) {
  if(event?.kind!=='entry-ability')return [];
  const abilityId=event.cause?.id,abilityName=abilityLabels[abilityId];
  if(!abilityName)return [];
  const holder=event.holderName??'ポケモン';
  if(event.result?.kind==='frisked' && abilityId==='frisk' && itemLabels[event.result.itemId])
    return [`${holder} の おみとおし！ ${event.friskTargetName??'ポケモン'} は ${itemLabels[event.result.itemId]} を持っている！`];
  if(event.result?.kind==='announced')return [`${holder} の ${abilityName}！`];
  return [];
}

// The item disclosure is independent of whether this public message renders.
export function formatPublicEntryItemMessages(event) {
  if(event?.kind!=='entry-item-revealed' || event.result?.kind!=='floating'
    || event.result.itemId!=='airballoon')return [];
  return [`${event.holderName??'ポケモン'} は ${itemLabels.airballoon}で浮いている！`];
}

// Message rendering is independent of the already-confirmed move cause and
// public disclosure decision. Preserve the existing status log text exactly.
export function formatPublicMoveStatusMessages(event) {
  if(event?.kind!=='status-move')return [];
  const {kind,statusId,moveName}=event.result??{};
  if(typeof moveName!=='string' || !moveName)return [];
  const targetName=event.targetName??'ポケモン';
  if(kind==='applied' && statusLabels[statusId])
    return [`${targetName} は ${moveName}で ${statusLabels[statusId]}状態になった！`];
  if(kind==='cleared')
    return [`${targetName} は ${moveName}で 状態異常が治った！`];
  return [];
}

// HP redistribution is already resolved by Pain Split. This text does not
// re-apply attack damage or infer the target's private exact HP.
export function formatPublicMoveHpLoss(event) {
  if(event?.kind!=='hp-loss-move' || event.result?.kind!=='hp-loss'
    || typeof event.result.moveName!=='string'||!event.result.moveName)return [];
  return [`${event.targetName??'ポケモン'} は ${event.result.moveName}で HPが減った！`];
}

// Only announced and confirmed ability-caused status changes use this path.
// Legacy neutral and cured-status messages remain outside this initial scope.
export function formatPublicStatusAbilityMessages(event) {
  if(event?.kind!=='status' || event.result?.kind!=='applied')return [];
  const {statusId,variant,sourceName,moveName}=event.result;
  const targetName=event.targetName??'ポケモン';
  const label=statusLabels[statusId],abilityId=event.cause?.id;
  if(!label || !abilityLabels[abilityId])return [];
  if(variant==='synchronize')
    return [`${targetName} は ${sourceName}の ${abilityLabels.synchronize}で ${label}状態になった！`];
  if(variant==='magicbounce' && moveName)
    return [`${targetName} は ${sourceName}の ${abilityLabels.magicbounce}で ${moveName}をはね返され、${label}状態になった！`];
  if(variant==='contact')
    return [`${targetName} は ${abilityLabels[abilityId]}で ${label}状態になった！`];
  return [];
}


// Condition messages and observed ability history subscribe independently to
// the same confirmed event. No text scanning or live ability lookup is needed.
export function formatPublicConditionAbilityMessages(event) {
  if(event?.kind!=='condition-ability')return [];
  const {conditionId,ended}=event.result??{};
  const abilityId=event.cause?.id;
  if(abilityId==='owntempo' && conditionId==='confusion' && ended)
    return [`${event.targetName??'ポケモン'} は ${abilityLabels.owntempo}で こんらんが解けた！`];
  if(!['cursedbody','oblivious'].includes(abilityId))return [];
  const text=conditionMessage({speciesName:event.targetName??'ポケモン'},{
    id:conditionId,ended,causeAbilityId:abilityId,
  });
  return text?[text]:[];
}

// The move's already-confirmed HP gain and its Japanese text are independent
// of observations. A held item or private set is never inferred from the heal.
export function formatPublicMoveHealMessages(event) {
  if(event?.kind!=='heal-move'||event.result?.kind!=='hp-gain'
    || typeof event.result.moveName!=='string'||!event.result.moveName)return [];
  return [`${event.targetName??'ポケモン'} は ${event.result.moveName}で HPを回復した！`];
}

// Public event text is a separate projection from the ability disclosure.
// The caller has already confirmed healing and its cause; no HP is read here.
export function formatPublicAbilityHealMessages(event) {
  if(event?.kind!=='heal' || event.result?.kind!=='hp-gain')return [];
  const name=abilityLabels[event.cause?.id];
  if(!name)return [];
  return [`${event.targetName??'ポケモン'} は ${name}で HPを回復した！`];
}


// A confirmed status cure and the observing player's knowledge subscribe to
// this event independently. Preserve the existing Japanese cure message.
export function formatPublicAbilityStatusCureMessages(event) {
  if(event?.kind!=='status-cured-ability'||event.result?.kind!=='status-cleared')return [];
  const ability=abilityLabels[event.cause?.id];
  return ability?[`${event.holderName??'ポケモン'} の ${ability}で状態異常が治った！`]:[];
}

// Harvest's message and public knowledge consume the same confirmed event.
export function formatPublicHarvestRestorationMessages(event) {
  if(event?.kind!=='harvest-restored' || event.result?.kind!=='berry-restored')return [];
  return [`${event.holderName??'ポケモン'} は ${abilityLabels.harvest}で ${itemLabels[event.result.itemId]??'きのみ'}を取り戻した！`];
}


// Field messages are a text-only projection of a confirmed transition.
// In particular, expiry cannot be attributed to an old setter, and neither
// an emitted line nor its absence decides ability knowledge.
export function formatPublicFieldChangeMessages(event) {
  if(event?.kind!=='field-change')return [];
  const {type,after,sourceMoveName,sourceName}=event.result??{};
  const cause=event.cause;
  if(type==='weather') {
    const description={
      sunnyday:'日差しが強くなった！',raindance:'雨が降り始めた！',
      sandstorm:'砂あらしが吹き始めた！',snowscape:'雪が降り始めた！',
    }[after];
    if(!description)return ['天候がおさまった！'];
    if(cause?.kind==='move' && sourceMoveName)return [`${sourceMoveName}で ${description}`];
    if(cause?.kind==='ability' && sourceName && abilityLabels[cause.id])
      return [`${sourceName}の ${abilityLabels[cause.id]}で ${description}`];
    return [description];
  }
  if(type==='terrain') {
    const label={
      electricterrain:'エレキフィールド',grassyterrain:'グラスフィールド',
      mistyterrain:'ミストフィールド',psychicterrain:'サイコフィールド',
    }[after];
    if(!label)
      return [cause?.kind==='move'&&cause.id==='icespinner'
        ?`${sourceMoveName??moveLabels.icespinner??'アイススピナー'}で フィールドの効果が消えた！`
        :'フィールドの効果が消えた！'];
    if(cause?.kind==='move' && sourceMoveName)return [`${sourceMoveName}で ${label}が広がった！`];
    if(cause?.kind==='ability' && sourceName && abilityLabels[cause.id])
      return [`${sourceName}の ${abilityLabels[cause.id]}で ${label}が広がった！`];
    return [`${label}が広がった！`];
  }
  return [];
}


// Natural Cure's switch-out announcement consumes the same confirmed event
// as the frozen outgoing appearance's observed-ability history.
export function formatPublicNaturalCureMessages(event) {
  if(event?.kind!=='switchout-ability-cure' || event.result?.kind!=='status-cleared'
    || event.cause?.id!=='naturalcure')return [];
  return [`${event.holderName??'ポケモン'} の ${abilityLabels.naturalcure}で状態異常が治った！`];
}


// A confirmed blocking ability and its public knowledge are consumers of a
// single confirmed public event; do not display a disguised holder species.
export function formatPublicAbilityMoveBlockMessages(event) {
  const abilityId=event?.cause?.id;
  if(event?.kind==='move-absorbed-ability' && event.result?.kind==='absorbed-move'
    && abilityLabels[abilityId])
    return [`${event.holderName??'ポケモン'} の ${abilityLabels[abilityId]}！`];
  if(event?.kind!=='move-blocked-ability' || !['damp','goodasgold'].includes(abilityId)
    || event.result?.kind!=='blocked-move' || ![0,1].includes(event.holderSeat))return [];
  const action=abilityId==='damp'?'は使えなかった！':'を防いだ！';
  return [`プレイヤー${event.holderSeat+1}の ${abilityLabels[abilityId]}で ${event.result.moveName}${action}`];
}


// Render an already-confirmed anonymous status transition. Unattributed
// changes stay anonymous; a public hazard can be named only when the caller
// supplies the confirmed hazard cause, never by inspecting text or inventory.
export function formatPublicNeutralStatusMessages(event) {
  if(event?.kind!=='status-transition')return [];
  const {kind,statusId}=event.result??{};
  const holder=event.targetName??'ポケモン';
  if(kind==='natural-end' && statusId==='') {
    if(event.result.previousStatus==='slp')return [`${holder} は目を覚ました！`];
    if(event.result.previousStatus==='frz')return [`${holder} のこおりがとけた！`];
    return [];
  }
  if(kind==='cleared' && statusId==='')return [`${holder} の状態異常が治った！`];
  if(kind==='applied' && statusLabels[statusId]) {
    const hazard=event.cause?.kind==='condition'&&event.cause.id==='toxicspikes'?'どくびしで ':'';
    return [`${holder} は ${hazard}${statusLabels[statusId]}状態になった！`];
  }
  return [];
}

// A committed move result is public. It does not identify the private
// ability or item that may have caused the failure.
export function formatPublicNeutralMoveOutcomeMessages(event) {
  if(event?.kind!=='move-result' || event.cause?.kind!=='move')return [];
  const {kind,moveName,reason}=event.result??{};
  if(typeof moveName!=='string'||!moveName)return [];
  if(kind==='immune')return [`${moveName}：相手には効果がないようだ。`];
  if(kind==='missed')return [`${moveName}：技は外れた。`];
  if(kind==='guarded')
    return [`${event.targetName??'ポケモン'} は ${moveLabels[event.result?.guardId]??'防御技'}で ${moveName}を防いだ！`];
  if(kind==='failed') {
    if(reason==='trick')return ['持ち物の交換は成立しなかった。'];
    if(reason==='status-fail')return [`${moveName}は効かなかった。`];
    if(reason==='generic')return [`${moveName}：しかし うまく決まらなかった。`];
  }
  return [];
}


// Unattributed positive healing from an already-public field/condition.
// Naming a setter would invent information that the confirmed result does
// not carry; item and ability healing have their own public event causes.
export function formatPublicConfirmedConditionHeal(event) {
  if(event?.kind!=='heal-condition' || event.result?.kind!=='hp-gain')return [];
  const label={
    wish:'ねがいごとの効果',
    grassyterrain:'グラスフィールド',
    leechseed:'やどりぎのタネの吸収',
  }[event.result.conditionId];
  if(!label)return [];
  return [`${event.targetName??'ポケモン'} は ${label}で HPを回復した！`];
}


// The event fixes public cause/target at the server boundary. Rendering never
// grants disclosure and uses the already-established legacy Japanese phrasing.
export function formatPublicConfirmedConditionTransition(event) {
  if(event?.kind!=='condition-transition')return [];
  const {id,ended,causeMoveId}=event.result??{};
  if(typeof id!=='string' || typeof ended!=='boolean')return [];
  const text=conditionMessage({speciesName:event.targetName??'ポケモン'},{
    id,ended,
    // A named cause is already confirmed and allowlisted by the projector.
    ...(event.cause?.kind==='move' && event.cause.id===causeMoveId
      ?{causeMoveId}:{}),
  });
  return text?[text]:[];
}

// Render an authoritative pre-move denial without reconstructing hidden
// inventory, a selected move, or the ability that might have caused it.
const publicUnableText=Object.freeze({
  slp:'眠っている',frz:'凍りついている',par:'体がしびれている',
  flinch:'ひるんでいる',confusion:'混乱して自分を攻撃した',
  attract:'メロメロで動けない',taunt:'ちょうはつされて技を使えない',
  disable:'かなしばりで技を封じられている',
  healblock:'かいふくふうじで回復技を使えない',
  throatchop:'じごくづきで音の技を使えない',
  recharge:'反動で動けない',
});
export function formatPublicUnableToAct(event) {
  if(event?.kind!=='action-result' || event.result?.kind!=='unable')return [];
  const name=event.actorName??'ポケモン';
  if(event.result.reason==='blocked')return [name+' は技を出せなかった。'];
  const cause=publicUnableText[event.result.reason];
  return cause?[name+' は'+cause+'。']:[];
}
