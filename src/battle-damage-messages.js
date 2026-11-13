// Format resolved public HP-loss events. Never infer a source from UI state.
import { moveLabels, statusLabels, abilityLabels, itemLabels } from '../data/catalog/labels-ja.js';

const siteNames={stealthrock:'ステルスロック',spikes:'まきびし'};
const weatherNames={sandstorm:'すなあらし'};
const conditionNames={leechseed:'やどりぎのタネ',curse:'のろい',partiallytrapped:'拘束状態',saltcure:'しおづけ',perishsong:'ほろびのうた'};

export function damageMessage(event) {
  const name=event.target.speciesName;
  // Both players share the same public log. Raw HP deltas would expose the
  // opposing Pokémon's private max-HP/stat values; only the cause is public.
  const loss=cause=>`${name} は ${cause}で ダメージを受けた！`;
  if(event.kind==='status')return loss(statusLabels[event.status??event.target.status]??'状態異常');
  if(event.kind==='hazard'&&siteNames[event.id])return loss(siteNames[event.id]);
  if(event.kind==='weather'&&weatherNames[event.id])return loss(weatherNames[event.id]);
  if(event.kind==='ability'&&abilityLabels[event.abilityId])return loss(abilityLabels[event.abilityId]);
  // Disguise's confirmed one-eighth HP cost is caused by the holder's ability,
  // not by the direct attack that was blocked by its intact disguise.
  if(event.kind==='disguise')return loss(abilityLabels.disguise);
  // Destiny Bond causes an immediate faint, not ordinary HP-loss damage.
  if(event.kind==='condition'&&event.id==='destinybond')return `${name} は みちづれにされた！`;
  if(event.kind==='condition'&&event.id==='confusion')return `${name} は混乱して自分を攻撃した。`;
  if(event.kind==='condition'&&conditionNames[event.id])return loss(conditionNames[event.id]);
  if(event.kind==='move'&&event.moveId&&moveLabels[event.moveId])return loss(moveLabels[event.moveId]);
  // Explosion and Self-Destruct use a distinct faint-first damage path.
  // Report their actual public move ID, rather than guessing which one was used.
  if(event.kind==='selfdestruct'&&event.moveId&&moveLabels[event.moveId])return loss(moveLabels[event.moveId]);
  if(event.kind==='contact')return `${name} にダメージ。`;
  if(event.kind==='recoil'&&event.moveId)return loss(`${moveLabels[event.moveId]??event.moveName??'技'}の反動`);
  if(event.kind==='crash'&&event.moveId)return loss(`${moveLabels[event.moveId]??event.moveName??'技'}の失敗時の反動`);
  if(event.kind==='item'&&event.itemId==='lifeorb')return loss('いのちのたまの反動');
  // These event kinds are emitted only after confirmed HP loss. Do not infer
  // an ability or move from an unconfirmed action or reveal the source Pokemon.
  if(event.kind==='liquidooze')return loss(abilityLabels.liquidooze);
  if(event.kind==='innardsout')return loss(abilityLabels.innardsout);
  if(event.kind==='spikyshield')return loss(moveLabels.spikyshield);
  return `${name} にダメージ。`;
}

// Formatter accepts only a sanitized, structured public event. It never
// decides disclosure or queries private Pokemon state.
export function formatPublicMessages(event) {
  if(event?.result?.silent===true)return [];
  if(event?.kind!=='damage')return [];
  const name=event.targetName??'ポケモン';
  const cause=event.cause;
  const kind=event.result?.displayKind;
  if(kind==='contact' && event.result?.contact && cause) {
    const label=cause.kind==='ability'?abilityLabels[cause.id]
      :cause.kind==='item'?itemLabels[cause.id]:null;
    if(label)return [name+' は 相手の '+label+'で ダメージを受けた！'];
  }
  const input={kind,target:{speciesName:name},
    id:cause && ['condition','hazard','weather'].includes(cause.kind)?cause.id:undefined,
    abilityId:cause?.kind==='ability'?cause.id:undefined,
    itemId:cause?.kind==='item'?cause.id:undefined,
    status:cause?.kind==='status'?cause.id:undefined,
    moveId:cause?.kind==='move'?cause.id:undefined};
  // For mapped direct ability/condition damage this is the existing formatter.
  // A missing/unknown cause falls back to the neutral public message.
  return [damageMessage(input)];
}

// Stable legacy text API, kept for callers not yet migrated to public events.
export function damageMessages(event) {
  if(event.kind==='contact'&&Array.isArray(event.causes)&&event.causes.length) {
    const names=event.causes.map(c=>c.kind==='ability'?abilityLabels[c.id]
      :c.kind==='item'?itemLabels[c.id]:null);
    if(names.every(Boolean))return names.map(name=>event.target.speciesName
      +' は 相手の '+name+'で ダメージを受けた！');
  }
  return [damageMessage(event)];
}
