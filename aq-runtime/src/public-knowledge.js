import { abilityLabels, itemLabels } from '../data/catalog/labels-ja.js';

// A malformed or future unsupported disclosure must not put an arbitrary
// private ability/item ID in opponent-visible observation history. This is
// defense in depth: the confirmed event projector still owns disclosure
// authorization and the actual effect timing.
function publiclyNamed(kind,id) {
  const labels=kind==='ability'?abilityLabels:kind==='item'?itemLabels:null;
  return labels && Object.hasOwn(labels,id) && typeof labels[id]==='string'
    && labels[id].length>0;
}

// Apply only explicitly disclosed public facts to their recorded appearance.
// No log formatting, current active Pokemon, private ability or item lookup.
export function applyPublicKnowledge(observations,event) {
  for(const disclosure of event?.disclosures??[]) {
    const {kind,id,subject}=disclosure;
    const field=kind==='ability'?'observedAbilityIds':kind==='item'?'observedItemIds'
      :kind==='move'?'usedMoves':null;
    if(!field || typeof id!=='string' || !id || !subject
      || ((kind==='ability'||kind==='item') && !publiclyNamed(kind,id))
      || (subject.seat!==0 && subject.seat!==1)
      || typeof subject.publicInstanceId!=='string')continue;
    const record=observations?.[subject.seat]?.find(
      entry=>entry.publicInstanceId===subject.publicInstanceId);
    if(!record)continue;
    // A confirmed post-faint source (e.g. Innards Out) can add an observation
    // to its already-frozen appearance without reidentifying the Pokemon.
    record[field]??=[];
    if(!record[field].includes(id))record[field].push(id);
  }
}
