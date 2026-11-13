import { effectiveness } from '../../../src/battle-mechanics.js';

const HEAL=new Set('recover roost softboiled slackoff synthesis moonlight morningsun wish milkdrink healorder shoreup lifedew rest strengthsap'.split(' '));
const SETUP=new Set('swordsdance nastyplot dragondance quiverdance calmmind bulkup shellsmash agility irondefense amnesia coil workup shiftgear honeclaws'.split(' '));

function stage(n=0){return n>=0?(2+n)/2:2/(2-n);}
function typeFactor(type,types,data) {
  return types?.length?effectiveness(type,types,data.typeChart):1;
}
function attackEstimate(p,foe,m,data) {
  if(!m || m.category==='Status')return 0;
  const power=m.basePower|| (m.ohko?150:0);
  if(!power)return 0;
  const special=m.category==='Special',attack=Number(special?p.stats?.spa:p.stats?.atk)||90;
  const species=data.species[foe.speciesId];
  const base=Number(special?species?.baseStats?.spd:species?.baseStats?.def)||90;
  const defense=base+55;
  const foeHP=(species?.baseStats?.hp??85)+80;
  const atkStage=stage(p.boosts?.[special?'spa':'atk']??0);
  const defStage=stage(foe.stages?.[special?'spd':'def']??0);
  const stab=p.types?.includes(m.type)?1.5:1;
  const effective=typeFactor(m.type,foe.types,data);
  const accuracy=m.accuracy===true?1:Math.min(1,Math.max(0,Number(m.accuracy??100)/100));
  const raw=(((22*power*attack*atkStage/(defense*defStage))/50)+2)*stab*effective*accuracy;
  return Math.max(0,Math.min(150,100*raw/foeHP));
}
function incomingThreat(p,foe,data) {
  const species=data.species[foe.speciesId];
  const theirTypes=foe.types??species?.types??[];
  return Math.max(1,...theirTypes.map(type=>typeFactor(type,p.types,data)));
}
function switchScore(p,foe,data) {
  if(p.hp<=0)return -Infinity;
  const hpRatio=p.hp/Math.max(1,p.maxHP);
  const moves=p.moves.map(slot=>data.moves[slot.id]).filter(Boolean);
  const offense=Math.max(0,...moves.map(m=>attackEstimate(p,foe,m,data)));
  return offense*0.8+hpRatio*22-incomingThreat(p,foe,data)*13;
}
function chooseSwitch(state,data,rng) {
  const me=state.self,foe=state.opponent;
  const options=me.team.map((p,index)=>({p,index}))
    .filter(({p,index})=>index!==me.active&&p.hp>0);
  if(!options.length)return null;
  options.sort((a,b)=>switchScore(b.p,foe,data)-switchScore(a.p,foe,data) || a.index-b.index);
  return {kind:'switch',index:options[0].index};
}
// Never pass opponent-private candidates, team members or actions to this AI.
export function chooseBattleAction(state,{data,rng,memory={}}={}) {
  const {self,opponent:foe}=state;
  if(self.confirmed||['wait','finished'].includes(state.requestType))return null;
  const active=self.team[self.active];
  if(state.requestType==='pivot'||state.requestType==='replacement'||active.hp<=0) {
    if(!self.needsSwitch && active.hp>0)return null;
    const next=chooseSwitch(state,data,rng);
    if(!next)throw new Error('Required replacement without a living reserve');
    memory.lastSwitchTurn=state.turn;
    return next;
  }
  if(state.requestType!=='turn')return null;
  const hpRatio=active.hp/Math.max(1,active.maxHP);
  const legal=active.moves.map((slot,index)=>({slot,index,m:data.moves[slot.id]}))
    .filter(({slot,m})=>!slot.disabled&&m);
  if(!legal.length)return {kind:'move',index:'struggle'};
  const enemyStatus=Boolean(foe.status);
  let best=null,bestScore=-Infinity,attackPeak=0;
  for(const entry of legal)attackPeak=Math.max(attackPeak,attackEstimate(active,foe,entry.m,data));
  for(const {slot,index,m} of legal) {
    let score=attackEstimate(active,foe,m,data);
    if(m.category==='Status') {
      if(m.heal||m.flags?.heal||HEAL.has(slot.id)) {
        score=hpRatio<0.82 ? (1-hpRatio)*102 : 1;
        if(memory.lastHealId===slot.id&&state.turn-(memory.lastHealTurn??-100)<=2)score*=0.2;
        if(slot.id==='rest'&&hpRatio>0.3)score*=0.55;
      } else if(SETUP.has(slot.id)||m.boosts||m.self?.boosts) {
        const boosts=active.boosts??{};
        const stageTotal=Math.max(boosts.atk??0,boosts.spa??0,boosts.spe??0);
        score=stageTotal<2 && hpRatio>0.55 ? 22+Math.min(12,attackPeak*0.18) : 2;
      } else if(m.status||m.volatileStatus==='leechseed'||['toxic','thunderwave','willowisp','spore','sleeppowder','yawn'].includes(slot.id)) {
        score=enemyStatus?0:28;
        if(memory.lastStatusId===slot.id&&state.turn-(memory.lastStatusTurn??-100)<=2)score*=0.15;
      } else if(m.sideCondition||m.weather||m.terrain||m.pseudoWeather) {
        const own=state.field?.sideEffects?.[0]??[];
        const opponent=state.field?.sideEffects?.[1]??[];
        const id=m.sideCondition;
        const target=['reflect','lightscreen','auroraveil','tailwind'].includes(id)?
          own:opponent;
        const layers=target.find(effect=>effect.id===id)?.layers??0;
        const maximum=id==='spikes'?3:id==='toxicspikes'?2:1;
        const already=(id&&target.some(effect=>effect.id===id)&&
          (layers===0||layers>=maximum))||
          (m.weather&&state.field?.weather===m.weather)||
          (m.terrain&&state.field?.terrain===m.terrain)||
          (m.pseudoWeather==='trickroom'&&state.field?.trickRoomTurns>0);
        score=already?-30:state.turn<9?17:4;
      } else if(slot.id==='revivalblessing') {
        score=self.team.some(p=>p.hp<=0)?55:0;
      } else if(['protect','kingsshield','spikyshield','silktrap'].includes(slot.id)) {
        score=hpRatio<0.4 ? 12 : 5;
      } else score=4;
      if(m.accuracy!==true && typeof m.accuracy==='number')score*=m.accuracy/100;
    } else {
      if(foe.hpPercent<=score*0.85)score+=25; // Prefer a likely knockout.
      if(!m.basePower)score=Math.max(score,2);
      if(m.priority>0&&foe.hpPercent<40)score+=7;
    }
    score+=(rng?rng.integer(1000):0)*0.0002;
    if(score>bestScore){bestScore=score;best={kind:'move',index};}
  }
  const alternatives=self.team.some((p,index)=>index!==self.active&&p.hp>0);
  if(self.canSwitch&&alternatives && memory.lastSwitchTurn!==state.turn &&
      (memory.lastSwitchTurn===undefined||state.turn-memory.lastSwitchTurn>2)) {
    const available=chooseSwitch(state,data,rng);
    const advantage=available
      ?switchScore(self.team[available.index],foe,data)-switchScore(active,foe,data):0;
    if(available && hpRatio>0.15 && incomingThreat(active,foe,data)>1.5 &&
       attackPeak<30 && advantage>23) {
      memory.lastSwitchTurn=state.turn;
      return available;
    }
  }
  const moveId=active.moves[best.index].id;
  if(moveId==='revivalblessing') {
    const targetIndex=self.team.findIndex((p,index)=>index!==self.active&&p.hp<=0);
    if(targetIndex>=0)best.targetIndex=targetIndex;
  }
  if(self.canMega)best.mega=true;
  if(HEAL.has(moveId)||data.moves[moveId]?.heal){memory.lastHealId=moveId;memory.lastHealTurn=state.turn;}
  if(data.moves[moveId]?.status||['toxic','thunderwave','willowisp','spore','sleeppowder','yawn'].includes(moveId)){
    memory.lastStatusId=moveId;memory.lastStatusTurn=state.turn;
  }
  return best;
}
