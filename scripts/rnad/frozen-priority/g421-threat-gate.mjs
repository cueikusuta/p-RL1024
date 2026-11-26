const stage=n=>n>=0?(2+n)/2:2/(2-n);
const hit=m=>m?.accuracy===true||m?.accuracy>=90;
const damage=m=>m&&m.basePower>0&&['Physical','Special'].includes(m.category);
export function firstThreatGate(o,acts,logits,ix,{data,estimateAttackPercent:est,calculateChampionsStats:calc}){
 const v=o.self,f=o.opponent,p=v.team[v.active],m=data.moves[p.moves[acts[ix]?.index]?.id];
 if(acts[ix]?.kind!=='move'||acts[ix].mega||!m||(m.priority??0)>0||m.flags?.charge||
   !p.stats||!p.maxHP||!p.hp||!f.hpPercent||!f.types?.length||
   o.field?.trickRoomTurns||['sunnyday','raindance','sandstorm','hail','snowscape'].includes(o.field?.weather)||
   ['par','slp','frz'].includes(p.status)||['par','slp','frz'].includes(f.status)||
   o.field?.sideEffects?.some(side=>side.some(x=>x.id==='tailwind')))return null;
 const seen=f.publicObservation,sd=data.species[f.speciesId];
 if(!sd?.baseStats||!seen?.usedMoves?.length)return null;
 const q={types:f.types,stats:calc(sd.baseStats,{spe:16,atk:16,spa:16}),boosts:f.stages,status:f.status,
   itemId:seen.observedItemIds?.length===1?seen.observedItemIds[0]:null,
   abilityId:seen.observedAbilityIds?.length===1?seen.observedAbilityIds[0]:null};
 q.maxHP=q.stats.hp;q.hpPercent=f.hpPercent;
 const speed=x=>x.stats.spe*stage(x.boosts?.spe??0)*(x.itemId==='choicescarf'?1.5:1);
 if(speed(q)<speed(p)*1.05)return null;
 const threat=seen.usedMoves.map(id=>({id,m:data.moves[id]}))
   .filter(x=>damage(x.m)&&hit(x.m)&&(x.m.priority??0)===0&&!x.m.flags?.charge)
   .map(x=>({...x,d:est(q,p,x.m,data,o.field)}))
   .filter(x=>x.d!==null&&x.d*.9>=100*p.hp/p.maxHP)
   .sort((a,b)=>b.d-a.d||a.id.localeCompare(b.id));
 if(!threat.length)return null;
 const choices=acts.map((a,i)=>({a,i,gap:logits[ix]-logits[i]}))
   .filter(x=>x.i!==ix&&x.gap<=4&&!x.a.mega&&x.a.targetIndex===undefined)
   .sort((a,b)=>a.gap-b.gap||a.i-b.i);
 const pri=choices.find(({a})=>{
   const alt=data.moves[p.moves[a.index]?.id];
   return a.kind==='move'&&damage(alt)&&hit(alt)&&(alt.priority??0)>0&&
     (est(p,q,alt,data,o.field)??-1)*.85>=f.hpPercent;
 });
 const hazards=o.field?.sideEffects?.[0]??[];
 const sw=!hazards.some(x=>['stealthrock','spikes','toxicspikes','stickyweb'].includes(x.id))?
   choices.find(({a})=>{
     if(a.kind!=='switch')return false;
     const t=v.team[a.index];if(!t?.stats||!t.hp||!t.maxHP)return false;
     return threat.every(({m})=>{
       const pct=est(q,t,m,data,o.field);
       return pct!==null&&pct<=.55*100*t.hp/t.maxHP;
     });
   }):null;
 const take=x=>x?{action:x.a,gap:x.gap}:null;
 return {ourSpecies:p.speciesId,oppSpecies:f.speciesId,ownHpPct:100*p.hp/p.maxHP,
   ourSpeed:speed(p),oppSpeedEstimate:speed(q),threat:threat.map(x=>({id:x.id,damage:x.d})),
   original:acts[ix],priority:take(pri),switch:take(sw)};
}
