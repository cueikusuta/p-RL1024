// Consumer-side projection: accept only fields already published by Battle.view(seat).
// Never import Battle checkpoints or inspect view.self/secret opposing rosters.
import {createOpponentBelief,observeOpponentBelief} from './opponent-belief.js';

const strings=value=>Array.isArray(value)
  ?value.filter(v=>typeof v==='string'&&v.length>0):[];
const shown=value=>typeof value==='string'&&value.length>0?value:null;

// Retain a per-appearance belief; a new appearance cannot inherit observations
// from a different opponent even when its displayed species is the same.
export function projectPublicOpponentBelief(publicBattleView,catalog,previous=null) {
  const opponent=publicBattleView?.opponent;
  if(!opponent||typeof opponent.speciesName!=='string'||!opponent.speciesName)
    throw new TypeError('Expected Battle.view(seat) with a visible opponent');
  const revealed=Array.isArray(publicBattleView.revealed)
    ?publicBattleView.revealed:[];
  const row=[...revealed].reverse().find(event=>event?.active===true);
  const appearanceId=shown(row?.publicInstanceId);
  const publicSpeciesName=shown(row?.rosterSpeciesName)
    ??shown(row?.baseSpeciesName)
    ??opponent.speciesName;
  const update={
    usedMoves:strings(row?.usedMoves),
    observedItemIds:strings(row?.observedItemIds),
    observedAbilityIds:strings(row?.observedAbilityIds),
  };
  // Only a publicly announced, unmasked transformation confirms the Mega form.
  if(shown(row?.baseSpeciesName)&&shown(row?.speciesName)
    &&row.speciesName!==row.baseSpeciesName) {
    // Battle exposes the evolved form ID after the public Mega announcement.
    // Catalog Mega set names still identify their original species; compare
    // canonical IDs instead of two independently authored display strings.
    if(shown(row?.speciesId))update.revealedMegaSpeciesId=row.speciesId;
    else update.revealedMegaSpeciesName=row.speciesName;
  }
  const same=previous?.appearanceId===appearanceId
    &&previous?.belief?.publicSpeciesName===publicSpeciesName
    &&appearanceId!==null;
  const belief=same
    ?observeOpponentBelief(previous.belief,update,catalog)
    :observeOpponentBelief(createOpponentBelief(publicSpeciesName,catalog),update,catalog);
  return {appearanceId,belief};
}

// Compact CPU projections expose the visible speciesId, but do not include
// full move/item observation history. Do not reconstruct unseen events.
export function projectCompactOpponentBelief(decisionView,catalog,previous=null) {
  const publicSpeciesId=shown(decisionView?.opponent?.speciesId);
  if(!publicSpeciesId||!Array.isArray(catalog))
    throw new TypeError('Expected authorized compact AI view and catalog');
  const direct=catalog.find(set=>!set.mega&&set.speciesId===publicSpeciesId);
  const evolved=catalog.find(set=>set.mega?.speciesId===publicSpeciesId);
  // Mega form identity is public after the form has actually appeared.
  const source=direct??catalog.find(set=>
    !set.mega&&set.speciesId===evolved?.speciesId);
  const publicSpeciesName=source?.speciesNameJa??
    evolved?.displaySpeciesNameJa??publicSpeciesId;
  // Accept only appearance-local observations already copied from
  // Battle.view(seat).revealed; never private opponent fields or party IDs.
  const row=decisionView.opponent.publicObservation;
  const appearanceId=shown(row?.appearanceId);
  const same=appearanceId!==null&&
    previous?.appearanceId===appearanceId&&
    previous?.publicSpeciesId===publicSpeciesId&&
    previous?.belief?.publicSpeciesName===publicSpeciesName;
  const initial=same?previous.belief:
    createOpponentBelief(publicSpeciesName,catalog);
  const observed=observeOpponentBelief(initial,{
    usedMoves:strings(row?.usedMoves),
    observedItemIds:strings(row?.observedItemIds),
    observedAbilityIds:strings(row?.observedAbilityIds),
  },catalog);
  const belief=evolved
    ?observeOpponentBelief(observed,{
      revealedMegaSpeciesId:publicSpeciesId,
    },catalog)
    :observed;
  return {publicSpeciesId,appearanceId,belief};
}
