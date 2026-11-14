import {budgetFor,requireRule,speciesKey,validateOffer,RuleError} from './draft.js';
import {generateCatalogOffer} from './catalog-offers.js';
import {cpuCandidateWeight,cpuSetStrength,resolveCpuHandicap} from './cpu-handicap.js';
import {CPU_SPECIAL_SETS} from './cpu-special-sets.js';

function randomStream(seed){
  let state=(seed>>>0)||1;
  return ()=>{
    state^=state<<13;state^=state>>>17;state^=state<<5;state>>>=0;
    return state/0x100000000;
  };
}
function weightedIndex(weights,random){
  const total=weights.reduce((sum,w)=>sum+w,0);
  if(!(total>0))throw new Error('No weighted CPU candidate');
  let point=random()*total;
  for(let i=0;i<weights.length;i++){
    point-=weights[i];
    if(point<0)return i;
  }
  return weights.length-1;
}
// Group before sampling, so a species with several registered sets does not
// get more opportunities merely by having multiple variants.
export function drawCpuCandidate(sets,usedSpecies,usedItems,handicap,random){
  const bySpecies=new Map();
  for(const set of sets){
    if(set.mega||usedSpecies.has(speciesKey(set))||
       (set.itemId&&usedItems.has(set.itemId)))continue;
    const key=speciesKey(set);
    if(!bySpecies.has(key))bySpecies.set(key,[]);
    bySpecies.get(key).push(set);
  }
  if(!bySpecies.size)return null;
  const groups=[...bySpecies.entries()].sort(([a],[b])=>String(a).localeCompare(String(b),'en'));
  const bestWeight=groups.map(([,rows])=>
    Math.max(...rows.map(row=>cpuCandidateWeight(row,handicap))));
  const group=groups[weightedIndex(bestWeight,random)][1]
    .slice().sort((a,b)=>a.setId.localeCompare(b.setId,'en'));
  return group[weightedIndex(group.map(row=>cpuCandidateWeight(row,handicap)),random)];
}
function drawExtraMega(sets,usedSpecies,usedItems,round,random){
  const groups=new Map();
  for(const set of sets){
    if(!set.mega||set.cost>(round===1?3:7)||
       usedSpecies.has(speciesKey(set))||
       (set.itemId&&usedItems.has(set.itemId)))continue;
    // Kyurem Black and White are separately weighted candidate forms, but
    // they retain the same Kyurem species clause when building a team.
    const key=set.speciesClauseId==='arceus'?'arceus':set.speciesId;
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push(set);
  }
  if(!groups.size)return null;
  const ordered=[...groups.entries()].sort(([a],[b])=>String(a).localeCompare(String(b),'en'));
  const species=ordered[Math.floor(random()*ordered.length)][1]
    .slice().sort((a,b)=>a.setId.localeCompare(b.setId,'en'));
  return species[Math.floor(random()*species.length)];
}
function drawCpuLegendary(usedSpecies,usedItems,random){
  // Draw species before a variant, so Arceus's many plates do not make
  // that species disproportionately likely to appear in the single Lv4 slot.
  const groups=new Map();
  for(const set of CPU_SPECIAL_SETS){
    if(usedSpecies.has(speciesKey(set))||
      (set.itemId&&usedItems.has(set.itemId)))continue;
    const key=speciesKey(set);
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push(set);
  }
  if(!groups.size)return null;
  const speciesOptions=[...groups.entries()]
    .sort(([a],[b])=>a.localeCompare(b,'en'));
  const group=speciesOptions[Math.floor(random()*speciesOptions.length)][1]
    .sort((a,b)=>a.setId.localeCompare(b.setId,'en'));
  return group[Math.floor(random()*group.length)];
}
function materialize(set,sets,facts,round,seat){
  const species=facts.species[set.speciesId];
  requireRule(species,'Missing species facts for CPU candidate');
  const name=sets.find(s=>s.speciesId===set.speciesId&&!s.mega)?.speciesNameJa;
  return {...structuredClone(set),
    instanceId:'r'+round+'-p'+seat+'-'+set.setId,
    speciesName:set.speciesNameJa,publicSpeciesName:set.baseSpeciesNameJa??name??set.speciesNameJa,
    baseStats:structuredClone(species.baseStats),types:[...species.types]};
}
// The offer has already been validated: species/items are globally distinct.
// The strength proxy is additive (catalog cost), so a bounded knapsack DP
// finds the exact best affordable six-member roster without an exponential
// scan over every six-of-N choice. Carryovers spend zero, but retain strength.
export function bestCpuOfferRosterScore(fresh,carry,credits){
  if(!Number.isSafeInteger(credits)||credits<0)return null;
  const carryIds=new Set(carry.map(set=>set.instanceId));
  const cap=Math.min(credits,6*8);
  const best=Array.from({length:7},()=>Array(cap+1).fill(-Infinity));
  best[0][0]=0;
  for(const set of [...fresh,...carry]){
    const cost=carryIds.has(set.instanceId)?0:set.cost;
    const value=6*cpuSetStrength(set);
    for(let count=5;count>=0;count--){
      for(let spent=cap-cost;spent>=0;spent--){
        const prior=best[count][spent];
        if(prior===-Infinity)continue;
        best[count+1][spent+cost]=Math.max(best[count+1][spent+cost],prior+value);
      }
    }
  }
  const highest=Math.max(...best[6]);
  return Number.isFinite(highest)?highest/6:null;
}

// CPU-only path. The ordinary catalog generator remains identical for PvP
// and level 0. All candidate pools are generated from reproducible seeds.
export function generateCpuCatalogOffer(sets,facts,{round,seat,carry=[],seed,difficulty=0,credits=null}){
  const h=resolveCpuHandicap(difficulty);
  if(difficulty===0)return generateCatalogOffer(sets,facts,{round,seat,carry,seed});
  requireRule(seat===1,'CPU handicap must only apply to seat 1');
  const availableCredits=credits??(budgetFor(round)+(round===1?h.initialCreditBonus:h.perRoundCreditBonus));
  const drafts=[];
  let attemptedDraws=0;
  // Invalid pools do not count as one of the promised candidate sets.
  // Retry with deterministic independent seeds, with a strict finite cap.
  for(let draw=0;draw<h.candidatePoolDraws+8&&drafts.length<h.candidatePoolDraws;draw++){
    attemptedDraws++;
    const drawSeed=draw===0?seed:((seed^Math.imul(draw,0x9e3779b9))>>>0)||1;
    // Reserve a special item BEFORE producing the base offer. This guarantees
    // a legal dedicated slot even when special items would
    // otherwise collide with the large normal/Mega candidate pool.
    const earlySpecial=h.specialLegendaryCandidates?
      drawCpuLegendary(new Set(carry.map(speciesKey)),
        new Set(carry.map(set=>set.itemId).filter(Boolean)),
        randomStream(drawSeed^0x19b4ea51)):null;
    if(h.specialLegendaryCandidates&&!earlySpecial)continue;
    let base;
    try {base=generateCatalogOffer(sets,facts,{round,seat,carry,seed:drawSeed,
      allowCpuSpecialCarry:h.specialLegendaryCandidates>0,
      reservedItems:earlySpecial?.itemId?[earlySpecial.itemId]:[]});}
    catch(error){
      if(error instanceof RuleError&&error.message==='Catalog cannot supply a complete legal offer')continue;
      if(error instanceof Error&&/^Witness search limit exceeded/.test(error.message))continue;
      throw error;
    }
    const random=randomStream(drawSeed^0x47c0ffee);
    const normals=base.fresh.filter(p=>!p.mega);
    const costOne=normals.filter(p=>p.cost===1);
    // At difficulty 3+ release up to two guaranteed cost-1 slots into
    // weighted draws. R7+ has no such fixed slots, so none are released.
    const released=h.costOneWeightMultiplier<1?Math.min(2,costOne.length):0;
    const releasedIds=new Set((released?costOne.slice(-released):[]).map(p=>p.instanceId));
    const fresh=base.fresh.filter(p=>!releasedIds.has(p.instanceId));
    const usedSpecies=new Set([...fresh,...carry].map(speciesKey));
    const usedItems=new Set([...fresh,...carry].map(p=>p.itemId).filter(Boolean));
    if(earlySpecial){
      usedSpecies.add(speciesKey(earlySpecial));
      if(earlySpecial.itemId)usedItems.add(earlySpecial.itemId);
    }
    let complete=true;
    for(let i=0;i<released+h.extraNormalCandidates;i++){
      const candidate=drawCpuCandidate(sets,usedSpecies,usedItems,h,random);
      if(!candidate){complete=false;break;}
      usedSpecies.add(speciesKey(candidate));
      if(candidate.itemId)usedItems.add(candidate.itemId);
      fresh.push(materialize(candidate,sets,facts,round,seat));
    }
    if(!complete)continue;
    const extraMegaSetIds=[];
    for(let i=0;i<h.extraMegaCandidates;i++){
      const candidate=drawExtraMega(sets,usedSpecies,usedItems,round,random);
      if(!candidate){complete=false;break;}
      usedSpecies.add(speciesKey(candidate));
      if(candidate.itemId)usedItems.add(candidate.itemId);
      extraMegaSetIds.push(candidate.setId);
      fresh.push(materialize(candidate,sets,facts,round,seat));
    }
    if(!complete)continue;
    const specialSetId=earlySpecial?.setId??null;
    if(earlySpecial)fresh.push(materialize(earlySpecial,CPU_SPECIAL_SETS,facts,round,seat));
    try {validateOffer(fresh,carry,round,{cpuHandicap:h});}
    catch {continue;}
    const rosterScore=bestCpuOfferRosterScore(fresh,carry,availableCredits);
    if(rosterScore===null)continue;
    drafts.push({fresh,rosterScore,draw,baseAudit:base.audit,
      extraMegaSetIds,specialSetId});
  }
  requireRule(drafts.length===h.candidatePoolDraws,'CPU catalog cannot supply a complete legal offer');
  drafts.sort((a,b)=>b.rosterScore-a.rosterScore||a.draw-b.draw);
  const best=drafts[0];
  return {fresh:best.fresh,audit:{
    ...best.baseAudit,cpuDifficulty:difficulty,drawsTried:attemptedDraws,
    selectedDraw:best.draw,legalRosterScore:best.rosterScore,
    normalCandidateCount:12+h.extraNormalCandidates,
    megaCandidateCount:best.fresh.filter(set=>set.mega&&!set.cpuSpecial).length,
    extraMegaSetIds:best.extraMegaSetIds,
    specialSetId:best.specialSetId,
  }};
}
