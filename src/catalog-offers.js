import { budgetFor, megaOfferRule, offerCostGuarantees, requireRule, validateOffer } from './draft.js';

export function seededShuffle(values, seed) {
  let state = seed >>> 0 || 1;
  const result = [...values];
  for (let index = result.length - 1; index > 0; index--) {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    const other = state % (index + 1);
    [result[index], result[other]] = [result[other], result[index]];
  }
  return result;
}

// Shuffle each species clause once, then shuffle the variants belonging to
// that clause. An extra registered Mega form must not multiply its species'
// chance of being considered.
export function seededSpeciesGroups(sets, speciesSeed, variantSeed = speciesSeed) {
  const bySpecies = new Map();
  for (const set of sets) {
    const key = set.speciesClauseId ?? set.speciesId;
    if (!bySpecies.has(key)) bySpecies.set(key, []);
    bySpecies.get(key).push(set);
  }
  return seededShuffle([...bySpecies.keys()], speciesSeed).map((key, index) =>
    seededShuffle(bySpecies.get(key), variantSeed ^ Math.imul(index + 1, 0xc2b2ae35)));
}

// Bounded independent search over the caller's eligible catalog pool.
export function findOfferWitness(sets, costs, blockedItems = [], seed = 1, maxNodes = 100000, {megaCount=null}={}) {
  blockedItems=blockedItems.filter(Boolean);
  // Choose a randomized species order independently within each cost tier.
  // Only then randomize that species' variants: extra sets must not grant
  // extra priority merely because the species has more registered variants.
  const options = new Map([...new Set(costs)].map(cost => [
    cost, seededSpeciesGroups(
      sets.filter(set => set.cost === cost && !blockedItems.includes(set.itemId)),
      seed ^ Math.imul(cost, 0x9e3779b9),
      seed ^ Math.imul(cost, 0x85ebca6b),
    ).flat(),
  ]));
  const remaining = new Map([...options.keys()].map(cost => [cost, costs.filter(c => c === cost).length]));
  const species = new Set(); const items = new Set(blockedItems); const chosen = [];
  let nodes = 0;
  function search() {
    if (++nodes > maxNodes) throw new Error(`Witness search limit exceeded (${maxNodes} nodes)`);
    const selectedMegas=chosen.filter(s=>s.mega).length;
    if(megaCount!==null && (selectedMegas>megaCount || selectedMegas+costs.length-chosen.length<megaCount))return null;
    if (chosen.length === costs.length) return megaCount===null || selectedMegas===megaCount ? [...chosen] : null;
    let next; let candidates;
    let possibleMegas=selectedMegas;
    let requiredMegas=selectedMegas;
    // Reserve shared items for higher-cost tiers first. Backtrack only when
    // needed to retain the full required tier distribution.
    for (const [cost, count] of [...remaining].sort((a, b) => b[0] - a[0])) {
      if (!count) continue;
      const available = options.get(cost).filter(set => !species.has(set.speciesClauseId ?? set.speciesId) && (!set.itemId || !items.has(set.itemId)) && !(set.mega && megaCount!==null && selectedMegas>=megaCount));
      if (new Set(available.map(s => s.speciesClauseId ?? s.speciesId)).size < count || new Set(available.filter(s=>s.itemId).map(s => s.itemId)).size+available.filter(s=>!s.itemId).length < count) return null;
      possibleMegas+=Math.min(count,new Set(available.filter(s=>s.mega).map(s=>s.speciesClauseId ?? s.speciesId)).size);
      requiredMegas+=Math.max(0,count-new Set(available.filter(s=>!s.mega).map(s=>s.speciesClauseId ?? s.speciesId)).size);
      if (!candidates) { next = cost; candidates = available; }
    }
    if(megaCount!==null && (possibleMegas<megaCount || requiredMegas>megaCount))return null;
    for (const set of candidates) {
      chosen.push(set); species.add(set.speciesClauseId ?? set.speciesId); if(set.itemId)items.add(set.itemId);
      remaining.set(next, remaining.get(next) - 1);
      const result = search();
      if (result) return result;
      remaining.set(next, remaining.get(next) + 1);
      chosen.pop(); species.delete(set.speciesClauseId ?? set.speciesId); if(set.itemId)items.delete(set.itemId);
    }
    return null;
  }
  return { offer: search(), nodes };
}

function hasAffordableTeam(offer, carry, round, requireMega=false) {
  const pool=[...carry,...offer],selected=[],species=new Set(),items=new Set();
  const carryIds=new Set(carry.map(p=>p.instanceId));
  function visit(start,cost,hasMega) {
    if(selected.length===6)return cost<=budgetFor(round) && (!requireMega||hasMega);
    for(let i=start;i<pool.length;i++) {
      const p=pool[i],key=p.speciesClauseId??p.speciesId;
      if(species.has(key)||p.itemId&&items.has(p.itemId))continue;
      const next=cost+(carryIds.has(p.instanceId)?0:p.cost);
      if(next>budgetFor(round))continue;
      species.add(key);if(p.itemId)items.add(p.itemId);selected.push(p);
      if(visit(i+1,next,hasMega||Boolean(p.mega)))return true;
      selected.pop();species.delete(key);if(p.itemId)items.delete(p.itemId);
    }
    return false;
  }
  return visit(0,0,false);
}

// Offer generation is server-owned. Never expose this audit seed or an
// opponent's fresh candidates in a public Match view.
export function generateCatalogOffer(sets, facts, {round,seat,carry=[],seed,allowCpuSpecialCarry=false,reservedItems=[]}) {
  requireRule(Array.isArray(reservedItems)&&reservedItems.every(id=>typeof id==='string'),'Invalid reserved items');
  budgetFor(round);
  requireRule(seat===0||seat===1,'Invalid seat');
  requireRule(Number.isInteger(seed)&&seed>0&&seed<=0xffffffff,'Invalid seed');
  const megaCount=megaOfferRule(round,carry.length).min;
  const rule=offerCostGuarantees(round);
  let state=seed;
  const extras=[];
  for(let i=0;i<rule.draws;i++) {
    state^=state<<13;state^=state>>>17;state^=state<<5;state>>>=0;
    extras.push(rule.drawFrom[Math.floor(state/0x100000000*rule.drawFrom.length)]);
  }
  const baseCosts=Object.entries(rule).filter(([cost])=>/^\d$/.test(cost)).flatMap(([cost,count])=>Array(count).fill(Number(cost)));
  const blockedSpecies=new Set(carry.map(p=>p.speciesClauseId??p.speciesId));
  const blockedItems=new Set([...carry.map(p=>p.itemId).filter(Boolean),...reservedItems]);
  state^=state<<13;state^=state>>>17;state^=state<<5;state>>>=0;
  const megaSpeciesGroups=seededSpeciesGroups(sets.filter(s=>Boolean(s.mega)&&s.cost<=(round===1?3:7)
    &&!blockedSpecies.has(s.speciesClauseId??s.speciesId)&&(!s.itemId||!blockedItems.has(s.itemId))),
    state ^ 0x9e3779b9, state ^ 0x85ebca6b);
  const normalPool=sets.filter(s=>!s.mega&&!blockedSpecies.has(s.speciesClauseId??s.speciesId));
  const alternatives=rule.draws===2?rule.drawFrom.flatMap(a=>rule.drawFrom.map(b=>[a,b])):rule.drawFrom.map(c=>[c]);
  const schedules=[extras,...seededShuffle(alternatives,state).filter(xs=>xs.join(',')!==extras.join(','))];
  let offer=null,nodes=0,selectedExtras=extras;const attempts=[];
  outer:for(const extra of schedules) {
    // Try differently seeded Mega combinations when a held item or species
    // competes with the twelve normal slots; never silently shrink either pool.
    for(let offset=0;offset<Math.min(megaSpeciesGroups.length,32);offset++) {
      const megas=[],megaSpecies=new Set(blockedSpecies),megaItems=new Set(blockedItems);
      for(let index=0;index<megaSpeciesGroups.length&&megas.length<megaCount;index++) {
        const group=megaSpeciesGroups[(index+offset)%megaSpeciesGroups.length];
        const s=group.find(candidate=>!candidate.itemId||!megaItems.has(candidate.itemId));
        if(!s)continue;
        const key=s.speciesClauseId??s.speciesId;
        if(megaSpecies.has(key))continue;
        megas.push(s);megaSpecies.add(key);if(s.itemId)megaItems.add(s.itemId);
      }
      if(megas.length!==megaCount)continue;
      const normalCandidates=normalPool.filter(s=>!megaSpecies.has(s.speciesClauseId??s.speciesId)
        &&(!s.itemId||!megaItems.has(s.itemId)));
      const maxNodes=100000-nodes;
      if(maxNodes<=0)break outer;
      const result=findOfferWitness(normalCandidates,[...baseCosts,...extra],[...megaItems],(state^Math.imul(offset+1,2654435761))>>>0,maxNodes,{megaCount:0});
      nodes+=result.nodes;
      attempts.push({extraCosts:extra,megaSetIds:megas.map(s=>s.setId),nodes:result.nodes,found:Boolean(result.offer)});
      if(result.offer&&hasAffordableTeam([...result.offer,...megas],carry,round,round===1)) {
        offer=[...result.offer,...megas];selectedExtras=extra;break outer;
      }
    }
  }
  requireRule(offer,'Catalog cannot supply a complete legal offer');
  const fresh=offer.map(set=>{
    const species=facts.species[set.speciesId];
    requireRule(species,'Missing species facts');
    const publicSpeciesName=sets.find(candidate=>candidate.speciesId===set.speciesId&&!candidate.mega)?.speciesNameJa;
    requireRule(!set.mega||publicSpeciesName,'Missing public base species name');
    return {...structuredClone(set),instanceId:'r'+round+'-p'+seat+'-'+set.setId,
      speciesName:set.speciesNameJa,publicSpeciesName:publicSpeciesName??set.speciesNameJa,
      baseStats:structuredClone(species.baseStats),types:[...species.types]};
  });
  validateOffer(fresh,carry,round,{allowCpuSpecialCarry});
  return {fresh,audit:{seed,round,seat,extraCosts:selectedExtras,initialExtraCosts:extras,attempts,
    megaCount,nodes,setIds:offer.map(s=>s.setId)}};
}
