import {playableSets} from '../../aq-runtime/src/production-runtime.js';
import {calculateChampionsStats} from '../../aq-runtime/src/champions-stats.js';
import {estimateAttackPercent} from '../../aq-runtime/src/ai/matchup-assessment.js';
import data from '../../aq-runtime/data/catalog/battle-data.json' with {type:'json'};

const clamp=(n,lo,hi)=>Math.min(hi,Math.max(lo,n));
const stages=p=>p?.spe??0;
const speed=(p,s)=>p.stats.spe*(s>=0?(2+s)/2:2/(2-s))*(p.itemId==='choicescarf'?1.5:1);
const hp=p=>typeof p.hp==='number'?p.hp*100/p.maxHP:p.hpPercent;
const hit=m=>m.accuracy===true?1:clamp((m.accuracy??0)/100,0,1);
const attack=(a,d,m,field)=>{
  if(!m||m.category==='Status')return null;
  const dmg=estimateAttackPercent(a,d,m,data,field);
  return dmg===null?null:{dmg,acc:hit(m),priority:m.priority??0};
};
const canModel=m=>m&&m.category!=='Status'&&['normal','allAdjacent','allAdjacentFoes'].includes(m.target)&&
  !m.selfSwitch&&!m.forceSwitch&&!m.recoil&&!m.drain&&!m.selfdestruct&&
  !m.damage&&!m.damageCallback&&!m.basePowerCallback&&!m.flags?.charge;
const score=(d,h)=>d.acc*(Math.min(d.dmg,h)/100+.30*Number(d.dmg>=h));
// Opponent-response choice is frozen per world, before comparing alternatives;
// the opponent cannot respond to our hidden action and our action cannot vary by world.
function utility(world,own,reserve,move,field){
  const foeMoves=world.moves.map((id,i)=>{
    const m=data.moves[id];
    if(!canModel(m))return null;
    const threat=attack(world,own,m,field);
    return threat&&{m,threat,i};
  }).filter(Boolean).sort((a,b)=>score(b.threat,hp(own))-score(a.threat,hp(own))||a.i-b.i);
  if(!foeMoves.length)return null;
  const {m,threat}=foeMoves[0];
  const other=attack(world,reserve,m,field);
  const outgoing=attack(own,world,move,field);
  if(!other||!outgoing)return null;
  const order=outgoing.priority-threat.priority||speed(own,stages(own.boosts))-speed(world,stages(world.boosts));
  const first=order>0?1:order<0?0:.5;
  const stay=score(outgoing,hp(world))*(1-(1-first)*threat.acc*Number(threat.dmg>=hp(own)))-
    score(threat,hp(own))*(1-first*outgoing.acc*Number(outgoing.dmg>=hp(world)));
  const reserveMoves=reserve.moves.map(x=>data.moves[x.id]).filter(canModel)
    .map(x=>attack(reserve,world,x,field)).filter(Boolean);
  const best=reserveMoves.length?Math.max(...reserveMoves.map(x=>score(x,hp(world)))):0;
  const change=-score(other,hp(reserve))+.20*best*(1-other.acc*Number(other.dmg>=hp(reserve)))-
    .20*score(outgoing,hp(world));
  return {stay,change};
}
const plainTypes=(a,b)=>Array.isArray(a)&&Array.isArray(b)&&a.length===b.length&&a.every(t=>b.includes(t));
const catalog=new Map();
for(const s of playableSets){
  // Refrain from pre-Mega inference when the compact view cannot establish it.
  if(!catalog.has(s.speciesId))catalog.set(s.speciesId,[]);
  catalog.get(s.speciesId).push(s);
}
for(const sets of catalog.values())sets.sort((a,b)=>a.setId<b.setId?-1:a.setId>b.setId?1:0);

export function worldsFor(raw,publicFrame,width,useReveals){
  const foe=raw.opponent, species=data.species[foe.speciesId];
  if(!species||!plainTypes(foe.types,species.types))return [];
  const row=publicFrame?.revealed?.findLast(v=>v.active===true);
  // Move ids are from the authorized public appearance projection, never
  // from an opponent roster. An inconsistent/missing active provenance falls back.
  if(publicFrame&&(!row||row.speciesId!==foe.speciesId))return [];
  const revealed=useReveals&&Array.isArray(row?.usedMoves)?row.usedMoves:[];
  const sets=(catalog.get(foe.speciesId)??[]).filter(s=>revealed.every(id=>s.moves.includes(id)));
  if(sets.length<2)return [];
  const selected=Array.from({length:Math.min(width,sets.length)},(_,i)=>
    sets[Math.floor(i*sets.length/Math.min(width,sets.length))]);
  return selected.map(s=>{const stats=calculateChampionsStats(species.baseStats,s.statPoints,
    data.natures[s.nature]??{});
    return {stats,maxHP:stats.hp,hpPercent:foe.hpPercent,types:foe.types,
      status:foe.status,boosts:foe.stages,abilityId:s.abilityId,itemId:s.itemId,moves:s.moves};
  });
}
export function createSelectiveReranker({mode,model}){
  if(!['Prior','History'].includes(mode))throw Error('Unexpected selective search arm');
  const stats={turns:0,eligible:0,twoWorlds:0,publicReveals:0,changed:0,worlds:0,histogram:Array(10).fill(0)};
  const edges=[.05,.1,.2,.5,1,2,5,10,25];
  const rerank=({observation,legalActions,logits,index,publicContext})=>{
    if(observation.phase!=='battle'||observation.requestType!=='turn')return index;
    stats.turns++;
    const base=legalActions[index];
    if(!base||!['move','switch'].includes(base.kind)||base.mega||
      base.targetIndex!==undefined)return index;
    const otherKind=base.kind==='move'?'switch':'move';
    let other=-1;
    for(let i=0;i<legalActions.length;i++){
      const a=legalActions[i];
      if(a.kind===otherKind&&!a.mega&&a.targetIndex===undefined&&
        (other<0||logits[i]>logits[other]))other=i;
    }
    if(other<0||logits[index]-logits[other]>model.maxLogitGap||!publicContext)return index;
    const raw=publicContext.compact;
    const field=raw?.field;
    if(!field||field.weather||field.terrain||field.trickRoomTurns||
      field.sideEffects?.some(rows=>rows.length))return index;
    const own=raw.self.team[raw.self.active],reserve=raw.self.team[legalActions[base.kind==='switch'?index:other].index];
    const move=data.moves[own?.moves?.[legalActions[base.kind==='move'?index:other].index]?.id];
    if(!own?.stats||!reserve?.stats||own.status||reserve.status||raw.opponent.status||
       own.hp<=0||reserve.hp<=0||!canModel(move))return index;
    stats.eligible++;
    const started=performance.now();
    try{
      const frame=publicContext.publicFrame();
      if(!frame)return index;
      if(frame?.revealed?.findLast(v=>v.active===true)?.usedMoves?.length)stats.publicReveals++;
      const worlds=worldsFor(raw,frame,model.worlds,mode==='History');
      if(worlds.length<model.minimumRealWorlds)return index;
      const rows=worlds.map(w=>utility(w,own,reserve,move,field));
      if(rows.some(x=>!x))return index;
      stats.twoWorlds++; stats.worlds+=worlds.length;
      const deltas=rows.map(r=>base.kind==='move'?r.change-r.stay:r.stay-r.change);
      const mean=deltas.reduce((a,b)=>a+b,0)/deltas.length;
      const worst=Math.min(...deltas);
      if(worst < -model.maxWorstWorldLoss ||
        mean+.25*(worst-mean)<model.minimumMeanUtilityGain)return index;
      stats.changed++;
      return other;
    } finally {
      const elapsed=performance.now()-started;
      const bin=edges.findIndex(x=>elapsed<=x);
      stats.histogram[bin<0?edges.length:bin]++;
    }
  };
  return {rerank,stats};
}
