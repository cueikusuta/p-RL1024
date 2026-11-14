// I737 P3: deterministic draft and lead/three-member ranking.
// These functions accept only the caller's own Match.view() and the public
// opponent species list; never accept an opponent's true team/selected sets.
import {validateTeam,speciesKey} from '../draft.js';
import {calculateChampionsStats} from '../champions-stats.js';
import {analyzeSetCapabilities} from './set-capabilities.js';
import {createOpponentBelief} from './opponent-belief.js';
import {assessMatchup} from './matchup-assessment.js';

const finite=n=>typeof n==='number'&&Number.isFinite(n);
const compareIds=(a,b)=>a.join('|').localeCompare(b.join('|'),'en');
function requireRoster(team) {
  if(!Array.isArray(team)||team.length!==6||
     team.some(p=>!p?.instanceId||!p?.setId))
    throw new TypeError('Expected six confirmed own instances');
  if(new Set(team.map(p=>p.instanceId)).size!==6)
    throw new TypeError('Duplicate own instance IDs');
}
function statsFor(p,data) {
  if(p.stats?.hp>0)return p.stats;
  const base=data.species?.[p.speciesId]?.baseStats;
  if(!base)return null;
  return calculateChampionsStats(base,p.statPoints??{},data.natures?.[p.nature]??{});
}
function typesFor(p,data) {
  return p.types??data.species?.[p.speciesId]?.types??[];
}
function basicStrength(p,data) {
  const stats=statsFor(p,data);
  if(!stats)return 0;
  const attacks=(p.moves??[]).map(id=>data.moves?.[id])
    .filter(m=>m&&m.category!=='Status'&&m.basePower>0);
  const power=attacks.length?Math.max(...attacks.map(m=>
    m.basePower*(typesFor(p,data).includes(m.type)?1.25:1))):0;
  return (stats.hp+stats.def+stats.spd)/55+
    Math.max(stats.atk,stats.spa)/13+stats.spe/24+Math.min(14,power/14);
}
// The opposing hidden selection is an unknown three-of-six subset, not a
// factual roster. Give all 20 public-species triples equal prior standing,
// then protect against the least covered hypotheses. These are utility
// scores, never probability estimates.
export function scorePublicOpponentTrioHypotheses(coverage) {
  if(!Array.isArray(coverage)||coverage.length!==6||
    coverage.some(value=>!finite(value)))
    throw new TypeError('Expected six finite public-species coverage scores');
  const scenarios=[];
  for(let i=0;i<6;i++)for(let j=i+1;j<6;j++)for(let k=j+1;k<6;k++) {
    const ordered=[coverage[i],coverage[j],coverage[k]].sort((a,b)=>a-b);
    scenarios.push(ordered[0]*0.6+ordered[1]*0.25+ordered[2]*0.15);
  }
  scenarios.sort((a,b)=>a-b);
  const mean=scenarios.reduce((sum,score)=>sum+score,0)/scenarios.length;
  const tail=scenarios.slice(0,4).reduce((sum,score)=>sum+score,0)/4;
  return mean*0.8+tail*0.2;
}

// Only one of multiple Mega candidates may transform in a single battle.
// Evaluate the best available incremental form value, NOT their sum. The
// other candidates can still fight in their untransformed forms.
export function estimateMegaOpportunityValue(team,data) {
  if(!Array.isArray(team)||!data?.species)
    throw new TypeError('Expected own team and battle data');
  return Math.max(0,...team.map(p=>{
    if(!p.mega?.speciesId)return 0;
    const mega=data.species[p.mega.speciesId];
    if(!mega?.baseStats)return 0;
    const megaStats=calculateChampionsStats(mega.baseStats,p.statPoints??{},
      data.natures?.[p.nature]??{});
    return Math.max(0,basicStrength({...p,stats:megaStats,types:mega.types},data)
      -basicStrength(p,data));
  }));
}

function roleBonus(team,capabilities,data) {
  const tags=team.map(p=>new Set(capabilities.get(p.instanceId).tags));
  const has=tag=>tags.some(s=>s.has(tag));
  const complementary=(enabler,receiver)=>tags.some((a,i)=>a.has(enabler)&&
    tags.some((b,j)=>i!==j&&b.has(receiver)));
  let score=0;
  if(complementary('screen','setup'))score+=10;
  // Passing boosts only has team value when a separate attacking receiver
  // exists; requiring a setup move on that receiver reverses the dependency.
  if(complementary('batonPass','attack'))score+=5;
  // A Trick Room setter plus an arbitrary fast attacker is not synergy.
  if(tags.some((a,i)=>a.has('trickRoom')&&team.some((p,j)=>
    i!==j&&tags[j].has('attack')&&finite(statsFor(p,data)?.spe)&&
    finite(statsFor(team[i],data)?.spe)&&
    statsFor(p,data).spe<statsFor(team[i],data).spe)))score+=3;
  if(has('hazard')&&has('phazing'))score+=4;
  if(has('hazardRemoval'))score+=2;
  if(has('recovery'))score+=2;
  return score;
}
function teamDiversity(team,data) {
  const types=new Set(team.flatMap(p=>typesFor(p,data)));
  const attacking=new Set(team.flatMap(p=>(p.moves??[]).map(id=>data.moves?.[id])
    .filter(m=>m&&m.category!=='Status').map(m=>m.type)));
  return types.size*1.2+attacking.size*1.5;
}
function capabilitiesFor(team,data) {
  return new Map(team.map(p=>[p.instanceId,analyzeSetCapabilities(p,data)]));
}

// The same non-additive scorer is available to a SMALL independent full-
// enumeration oracle. The caller must validate legal rosters separately.
function scoreValidDraftRoster(chosen,{scored,caps,megaById,data,credits,spent,carried}) {
  return chosen.reduce((n,p)=>n+scored.get(p.instanceId),0)+
    roleBonus(chosen,caps,data)+teamDiversity(chosen,data)+
    Math.max(0,...chosen.map(p=>megaById.get(p.instanceId)??0))*0.2+
    Math.max(0,credits-spent)*0.15+
    chosen.filter(p=>carried.has(p.instanceId)).length*0.6;
}
export function scoreStrategyDraftRoster(chosen,{data,credits,carry=[]}={}) {
  if(!Array.isArray(chosen)||chosen.length!==6||!data?.moves||
     !Number.isSafeInteger(credits)||!Array.isArray(carry)||
     chosen.some(p=>!p?.instanceId)||new Set(chosen.map(p=>p.instanceId)).size!==6)
    throw new TypeError('Expected six distinct own sets, catalog data and budget');
  const carried=new Set(carry.map(p=>p.instanceId));
  const scored=new Map(chosen.map(p=>[p.instanceId,basicStrength(p,data)]));
  const caps=capabilitiesFor(chosen,data);
  const megaById=new Map(chosen.map(p=>[
    p.instanceId,estimateMegaOpportunityValue([p],data),
  ]));
  const spent=chosen.reduce((n,p)=>n+(carried.has(p.instanceId)?0:p.cost),0);
  return scoreValidDraftRoster(chosen,{
    scored,caps,megaById,data,credits,spent,carried,
  });
}

// Full enumeration is intentional: the strategy contains non-additive
// synergies, so the old additive branch-and-bound upper bound is not sound.
export function chooseStrategyTeam(view,{data,forcedSetId=null,
  fastDraftValidation=false,cpuHandicap=null,maxCandidateCount=null}={}) {
  if(!data?.moves)throw new TypeError('Expected battle definitions');
  if(typeof fastDraftValidation!=='boolean')
    throw new TypeError('Invalid fast draft validation option');
  const fresh=view?.self?.fresh??[],carry=view?.self?.carry??[];
  const completeOffer=[...carry,...fresh],credits=view?.self?.credits,round=view?.round;
  // CPU-only fast path: evaluate a bounded candidate shortlist when the
  // Ascension offer is large. Keep the cheapest six for affordable completion,
  // then the highest standalone strengths and all eligible Mega/special sets.
  // Final roster is still fully validated against the complete original offer.
  let options=completeOffer;
  if(maxCandidateCount!==null){
    if(!Number.isInteger(maxCandidateCount)||maxCandidateCount<10)
      throw new TypeError('Invalid CPU shortlist size');
    if(options.length>maxCandidateCount){
      const spent=p=>carry.some(q=>q.instanceId===p.instanceId)?0:p.cost;
      const cheapest=options.slice().sort((a,b)=>spent(a)-spent(b)||
        a.setId.localeCompare(b.setId,'en')).slice(0,6);
      const high=options.slice().sort((a,b)=>basicStrength(b,data)-basicStrength(a,data)||
        a.setId.localeCompare(b.setId,'en'));
      const selected=new Set(cheapest.map(p=>p.instanceId));
      // Retain the explicitly offered CPU-only legendary in the shortlist.
      for(const p of options.filter(p=>p.cpuSpecial)){
        selected.add(p.instanceId);
      }
      for(const p of high){
        if(selected.size>=maxCandidateCount)break;
        selected.add(p.instanceId);
      }
      options=completeOffer.filter(p=>selected.has(p.instanceId));
    }
  }
  if(!Number.isSafeInteger(credits)||!Number.isInteger(round))
    throw new TypeError('Expected authorized draft view');
  if(forcedSetId&&!options.some(p=>p.setId===forcedSetId))
    throw new Error('Forced set absent from public own offer');
  const carried=new Set(carry.map(p=>p.instanceId));
  // I877 P3/P4: the opt-in simulator path precomputes invariant candidate
  // constraints. Full server validation still runs once on the winning roster.
  // Never cache this across offers/rounds: carry/credits are Match-specific.
  const optionInfo=fastDraftValidation?options.map(p=>({
    id:p.instanceId,species:speciesKey(p),item:p.itemId,
    cost:carried.has(p.instanceId)?0:p.cost,mega:Boolean(p.mega),
  })):null;
  const selectedInfo=[];
  const scored=new Map(options.map(p=>[p.instanceId,basicStrength(p,data)]));
  const caps=capabilitiesFor(options,data);
  // A Mega gain belongs to the own set, not to each roster combination.
  // Cache it once; full six-of-offer enumeration visits many legal rosters.
  const megaById=new Map(options.map(p=>[
    p.instanceId,estimateMegaOpportunityValue([p],data),
  ]));
  let best=null,bestScore=-Infinity;
  const chosen=[];
  function visit(start,remainingCost,megaCount) {
    if(chosen.length===6) {
      if(forcedSetId&&!chosen.some(p=>p.setId===forcedSetId))return;
      const ids=chosen.map(p=>p.instanceId);
      // The legacy branch validates (and deep-clones) every legal leaf.
      // The fast branch checks the exact candidate invariants during DFS,
      // then invokes authoritative validateTeam once on the selected winner.
      // No alternate rule interpretation or opponent truth is introduced.
      if(!fastDraftValidation) {
        try {validateTeam(ids,fresh,carry,round,credits);}
        catch {return;}
      }
      const score=scoreValidDraftRoster(chosen,{
        scored,caps,megaById,data,credits,spent:remainingCost,carried,
      });
      if(score>bestScore+1e-9||
         (Math.abs(score-bestScore)<=1e-9&&
          (!best||compareIds(ids,best)<0))) {
        best=ids;bestScore=score;
      }
      return;
    }
    if(options.length-start<6-chosen.length)return;
    for(let i=start;i<=options.length-(6-chosen.length);i++) {
      const p=options[i],info=optionInfo?.[i];
      const cost=info?info.cost:(carried.has(p.instanceId)?0:p.cost);
      const mega=info?info.mega:Boolean(p.mega);
      if(!Number.isInteger(cost)||remainingCost+cost>credits||
         megaCount+mega>3)continue;
      if(info
        ?selectedInfo.some(q=>q.id===info.id||q.species===info.species||
          (info.item&&q.item===info.item))
        :chosen.some(q=>speciesKey(q)===speciesKey(p)||
          (p.itemId&&q.itemId===p.itemId)))continue;
      chosen.push(p);
      if(info)selectedInfo.push(info);
      visit(i+1,remainingCost+cost,megaCount+mega);
      chosen.pop();
      if(info)selectedInfo.pop();
    }
  }
  visit(0,0,0);
  if(!best)throw new Error('No legal affordable six-member strategy roster');
  // Fail closed if the underlying authoritative rule becomes stricter.
  // The ordinary production path (fastDraftValidation=false) is unchanged.
  if(fastDraftValidation)validateTeam(best,fresh,carry,round,credits,{cpuHandicap});
  return best;
}

function publicThreats(names,catalog,data) {
  return names.map(name=>{
    const belief=createOpponentBelief(name,catalog);
    const representative=catalog.find(set=>!set.mega&&set.speciesNameJa===name);
    const species=representative&&data.species?.[representative.speciesId];
    return {name,belief,opponentPublic:{
      types:species?.types??[],hpPercent:100,stages:{},status:'',
    }};
  });
}
function pairAssessment(p,threat,catalog,data) {
  const stats=statsFor(p,data);
  if(!stats)return {score:0,attackKnown:false,immune:false};
  const own={...p,stats,types:typesFor(p,data),hp:stats.hp,
    maxHP:stats.hp,boosts:{},status:''};
  const model=assessMatchup({ownSet:p,own,
    opponentBelief:threat.belief,opponentPublic:threat.opponentPublic,
    catalog,data,field:{sideEffects:[[],[]],weather:null,terrain:null}});
  const axis=name=>model[name]?.status==='evaluated'&&
    finite(model[name].score)?model[name].score:0;
  const attacking=(p.moves??[]).filter(id=>data.moves?.[id]?.category!=='Status');
  const estimates=attacking.map(id=>model.moveEstimates.find(m=>m.id===id));
  // Known zero for every supported attack is distinct from an unmodelled
  // attack, or a status-only set (which may still win indirectly).
  const attackKnown=attacking.length>0&&estimates.every(m=>
    m&&finite(m.expectedPercent)&&finite(m.maxPercent));
  const immune=attackKnown&&estimates.every(m=>m.maxPercent===0);
  const score=32*axis('attack')+15*axis('defense')+
    8*axis('moveOrder')+8*axis('switchIn')-
    9*axis('breakthroughRisk');
  return {score,attackKnown,immune};
}

// Public species names alone are used. Opponent private roster, selection,
// set IDs, item/ability/move internals are deliberately ignored.
export function rankStrategySelections(view,{catalog,data,forcedSetId=null,
  forceLead=false,excludedSpeciesKey=null,excludedItemId=null}={}) {
  if(!Array.isArray(catalog)||!data?.moves||!data?.typeChart)
    throw new TypeError('Expected current public catalog and battle data');
  const team=view?.self?.team;requireRoster(team);
  const foeNames=view?.opponent?.species;
  if(!Array.isArray(foeNames)||foeNames.some(x=>typeof x!=='string'))
    throw new TypeError('Expected opponent public six-species list');
  if(foeNames.length!==0&&foeNames.length!==6)
    throw new TypeError('Opponent preview must be absent or contain six names');
  if(forceLead&&!forcedSetId)throw new Error('Force lead requires forced set');
  if(forcedSetId&&!team.some(p=>p.setId===forcedSetId))
    throw new Error('Forced set absent from own roster');
  const caps=capabilitiesFor(team,data);
  const base=team.map(p=>basicStrength(p,data));
  const threats=publicThreats(foeNames,catalog,data);
  const pairs=team.map(p=>threats.map(t=>pairAssessment(p,t,catalog,data)));
  const rows=[];
  for(let i=0;i<6;i++)for(let j=i+1;j<6;j++)for(let k=j+1;k<6;k++) {
    const positions=[i,j,k],trio=positions.map(n=>team[n]);
    if(forcedSetId&&!trio.some(p=>p.setId===forcedSetId))continue;
    if(trio.some(p=>(excludedSpeciesKey&&speciesKey(p)===excludedSpeciesKey)||
      (excludedItemId&&p.itemId===excludedItemId)))continue;
    const synergy=roleBonus(trio,caps,data)+teamDiversity(trio,data)*0.35;
    const cover=threats.map((_,t)=>{
      const scored=positions.map(n=>pairs[n][t]);
      const ordered=[...scored].sort((a,b)=>b.score-a.score);
      const allImmune=scored.every(p=>p.immune);
      return ordered[0].score+ordered[1].score*0.2-
        (allImmune?125:0);
    });
    const coverScore=cover.length?
      scorePublicOpponentTrioHypotheses(cover):0;
    const megaValue=estimateMegaOpportunityValue(trio,data);
    const coreScore=positions.reduce((n,p)=>n+base[p],0)*0.35+
      synergy+coverScore+megaValue*0.35;
    for(const lead of positions) {
      const reserve=positions.filter(i=>i!==lead);
      // Preserve all 120 legal ordered possibilities. First slot is the
      // actual lead used by Match.confirmSelection.
      for(const order of [reserve,[...reserve].reverse()]) {
        const indexes=[lead,...order];
        if(forceLead&&team[lead].setId!==forcedSetId)continue;
        const matchups=pairs[lead].map(row=>row.score);
        const leadScore=matchups.length?
          matchups.reduce((n,v)=>n+v,0)/matchups.length*0.25+
          Math.min(...matchups)*0.12:0;
        rows.push({selection:indexes.map(n=>team[n].instanceId),
          score:coreScore+leadScore,
          breakdown:{core:coreScore,lead:leadScore,
            coverage:coverScore,synergy,megaOpportunity:megaValue}});
      }
    }
  }
  rows.sort((a,b)=>b.score-a.score||compareIds(a.selection,b.selection));
  return rows;
}
export function chooseStrategySelection(view,options) {
  const best=rankStrategySelections(view,options)[0];
  if(!best)throw new Error('No legal strategy three-member selection');
  return best.selection;
}
