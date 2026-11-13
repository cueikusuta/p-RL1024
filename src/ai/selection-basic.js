// I737 Phase A: the literal legacy Basic scoring and enumeration contract.
// This module takes only own confirmed members and opponent public species.
import {speciesKey} from '../draft.js';
import {effectiveness} from '../battle-mechanics.js';

export function setValue(set, data) {
  const s=set.stats??set.baseStats??{};
  const bulk=(s.hp??80)*0.18+(s.def??80)*0.12+(s.spd??80)*0.12;
  const offense=Math.max(s.atk??80,s.spa??80)*0.22+(s.spe??80)*0.16;
  const moves=(set.moves??[]).map(id=>data.moves[id]).filter(Boolean);
  const attacks=moves.filter(m=>m.category!=='Status'&&m.basePower>0);
  const pressure=attacks.length?Math.max(...attacks.map(m=>
    Math.min(150,m.basePower)*0.12*(set.types?.includes(m.type)?1.35:1)*(m.accuracy===true?1:(m.accuracy??100)/100))):0;
  const utility=moves.filter(m=>m.category==='Status').length*1.6;
  return bulk+offense+pressure+utility;
}
function teamSynergy(team) {
  let score=0;
  for(let i=0;i<team.length;i++)for(let j=i+1;j<team.length;j++) {
    const a=team[i].types??[],b=team[j].types??[];
    const shared=a.filter(t=>b.includes(t)).length;
    score-=shared*4.5;
    if(!shared)score+=0.65;
  }
  return score;
}
// Public catalog identity stays fixed throughout one simulation. Rebuilding
// this lookup for every selection and scanning the full catalog for each Mega
// was a major avoidable cost in large whole-catalog runs.
const publicTypesCache=new WeakMap();
export function publicRosterTypes(names,sets,data) {
  let cache=publicTypesCache.get(sets);
  if(!cache || cache.data!==data || cache.size!==sets.length) {
    const namesToTypes=new Map();
    const ordinaryByClause=new Map();
    for(const set of sets)if(!set.mega && !ordinaryByClause.has(set.speciesClauseId))
      ordinaryByClause.set(set.speciesClauseId,set.speciesNameJa);
    for(const set of sets) {
      const species=data.species[set.speciesId];
      if(!species)continue;
      const label=set.mega
        ?ordinaryByClause.get(set.speciesClauseId)
        :set.speciesNameJa;
      if(label && !namesToTypes.has(label))namesToTypes.set(label,species.types);
    }
    cache={data,size:sets.length,namesToTypes};
    publicTypesCache.set(sets,cache);
  }
  return (names??[]).map(name=>cache.namesToTypes.get(name)??[]);
}
export function rankBasicSelectionCandidates(view,{data,sets,rng=null,forcedSetId=null,
  forceLead=false,excludedSpeciesKey=null,excludedItemId=null}={}) {
  if(forceLead&&!forcedSetId)throw new Error('Force lead requires a specified set');
  const team=view.self.team??[];
  if(team.length!==6)throw new Error('Selection requires a confirmed six-member roster');
  const forced=forcedSetId?team.find(p=>p.setId===forcedSetId):null;
  if(forcedSetId&&!forced)throw new Error('Forced selection is absent from confirmed roster: '+forcedSetId);
  const foeTypes=publicRosterTypes(view.opponent.species,sets,data);
  // Consume exactly one draw for every own member BEFORE filtering trios.
  // Floating-point evaluation order and > (first-tie wins) are unchanged.
  const scores=team.map(p=>{
    const attacks=(p.moves??[]).map(id=>data.moves[id]).filter(m=>m&&m.category!=='Status');
    const spread=foeTypes.length?foeTypes.reduce((sum,types)=>{
      const best=attacks.reduce((max,m)=>{
        if(!m.basePower || !types.length)return max;
        return Math.max(max,Math.min(4,effectiveness(m.type,types,data.typeChart))*(p.types?.includes(m.type)?1.35:1)*m.basePower);
      },0);
      return sum+best/foeTypes.length;
    },0):0;
    return setValue(p,data)+spread*0.16+(rng?rng.integer(1000)/10000:0);
  });
  const rows=[];
  for(let i=0;i<4;i++)for(let j=i+1;j<5;j++)for(let k=j+1;k<6;k++) {
    const chosen=[team[i],team[j],team[k]];
    if(forced&&!chosen.includes(forced))continue;
    if(chosen.some(p=>(excludedSpeciesKey&&speciesKey(p)===excludedSpeciesKey)||
        (excludedItemId&&p.itemId===excludedItemId)))continue;
    const score=scores[i]+scores[j]+scores[k]+teamSynergy(chosen)*1.5;
    let selection=chosen.map(p=>p.instanceId);
    if(forceLead){
      const targetId=forced.instanceId;
      selection=[targetId,...selection.filter(id=>id!==targetId)];
    }
    rows.push({score,selection});
  }
  return rows;
}

export function chooseBasicSelection(view,options={}) {
  const rows=rankBasicSelectionCandidates(view,options);
  let best=null,bestScore=-Infinity;
  for(const row of rows) {
    if(row.score>bestScore){bestScore=row.score;best=row.selection;}
  }
  if(!best)throw new Error('No valid three-member selection');
  return best;
}
