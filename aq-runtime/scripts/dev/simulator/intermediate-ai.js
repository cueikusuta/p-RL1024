// Intermediate-strength, public-information-only battle policy.
// All opponent estimates come from the visibly revealed species/types/stages.
// It never reads the hidden roster, move slots, ability, item, or selected set.
import { effectiveness, elementalDamage } from '../../../src/battle-mechanics.js';

const HEALS = new Set('recover roost softboiled slackoff synthesis moonlight morningsun wish milkdrink healorder shoreup lifedew rest strengthsap'.split(' '));
const SETUPS = new Set('cosmicpower swordsdance nastyplot dragondance quiverdance calmmind bulkup shellsmash agility irondefense amnesia coil workup shiftgear honeclaws'.split(' '));
const STATUS = new Set('toxic thunderwave willowisp spore sleeppowder yawn glare stunspore'.split(' '));
const HAZARDS = new Set('stealthrock spikes toxicspikes stickyweb'.split(' '));
const REMOVAL = new Set('rapidspin defog mortalspin tidyup courtchange'.split(' '));
const SCREENS = new Set('reflect lightscreen auroraveil tailwind'.split(' '));
const SHIELDS = new Set('protect kingsshield spikyshield silktrap banefulbunker'.split(' '));
const STAGE = n => n >= 0 ? (2 + Math.min(6,n))/2 : 2/(2 - Math.max(-6,n));
const pct = (hp,maxHP) => 100*hp/Math.max(1,maxHP);
const accuracy = move => move?.accuracy===true?1:Math.max(0,Math.min(1,Number(move?.accuracy??100)/100));
const typeEffect = (moveType,types,data) => (moveType&&types?.length)
  ?effectiveness(moveType,types,data.typeChart):1;
const hasEffect=(effects,id)=>effects?.some(x=>x.id===id);
// Mirror public type immunity rules in battle-status-moves.js and battle-status.js.
function publiclyImmune(id,move,actor,foe,data) {
  if(!foe?.types?.length||move.target==='self')return false;
  const types=foe.types;
  // Damaging moves also need the public type-immunity check. In particular,
  // Rapid Spin / Mortal Spin cannot remove hazards when their hit is immune.
  // Scrappy is our own known ability, so bypassing Ghost is not hidden data.
  if(move.category!=='Status'&&move.ignoreImmunity!==true) {
    const effectiveTypes=types.filter(type=>!(type==='Ghost'&&
      actor.abilityId==='scrappy'&&['Normal','Fighting'].includes(move.type)));
    if(typeEffect(move.type,effectiveTypes,data)===0)return true;
  }
  if(move.flags?.powder&&types.includes('Grass'))return true;
  if(move.ignoreImmunity===false&&typeEffect(move.type,types,data)===0)return true;
  if(id==='leechseed'&&types.includes('Grass'))return true;
  const status=move.status==='tox'?'psn':move.status;
  if(status)for(const type of types) {
    if(data.typeChart?.[type.toLowerCase()]?.[status]===3&&
       !(status==='psn'&&['Steel','Poison'].includes(type)&&
         actor.abilityId==='corrosion'))return true;
  }
  return false;
}
const living=team=>team.filter(p=>p.hp>0).length;
function seenEnemySpeeds(view,data,memory) {
  // Only observed opponent species; their unrevealed team/stats remain private.
  const id=view.opponent?.speciesId;
  if(id) {
    memory.publicSpeeds??={};
    memory.publicSpeeds[id]=foeSpeed(view.opponent,data)*
      STAGE(view.opponent.stages?.spe??0);
  }
  return Object.values(memory.publicSpeeds??{});
}
function fieldPreference(id,move,active,view,data,threat,speeds) {
  const hp=pct(active.hp,active.maxHP),allies=living(view.self.team);
  const enemies=Math.max(1,view.opponent.remaining??3);
  const own=view.field?.sideEffects?.[0]??[],other=view.field?.sideEffects?.[1]??[];
  if(HAZARDS.has(id)||HAZARDS.has(move.sideCondition)) {
    const hazard=HAZARDS.has(id)?id:move.sideCondition;
    const limit=hazard==='spikes'?3:hazard==='toxicspikes'?2:1;
    if(effectLayers(other,hazard)>=limit||enemies<=1||hp<=threat*1.08)return null;
    return 36+enemies*12-effectLayers(other,hazard)*12;
  }
  if(SCREENS.has(id)||SCREENS.has(move.sideCondition)) {
    const condition=SCREENS.has(id)?id:move.sideCondition;
    if(hasEffect(own,condition)||allies<2||threat<32||
      hp<=threat*(move.priority<0?1.2:0.85)||
      (condition==='auroraveil'&&!['snowscape','hail'].includes(view.field?.weather)))return null;
    return 35+Math.min(25,threat*0.38)+allies*8;
  }
  if(id==='trickroom') {
    if(view.field?.trickRoomTurns>0||allies<2||!speeds.length||
       hp<=threat*1.3)return null; // -7 priority: opponent usually acts first.
    const known=speeds.reduce((sum,x)=>sum+x,0)/speeds.length;
    const own=view.self.team.filter(x=>x.hp>0);
    const slower=own.filter(p=>mineSpeed(p)<known*0.94).length;
    if(slower<Math.floor(own.length/2)+1)return null;
    return 37+slower*12+speeds.length*3;
  }
  if(REMOVAL.has(id)) {
    const hazards=own.filter(x=>HAZARDS.has(x.id));
    return hazards.length?27+hazards.length*12+allies*8:null;
  }
  return null;
}
const effectLayers=(effects,id)=>effects?.find(x=>x.id===id)?.layers??0;
const foeSpeed=(foe,data) => Number(data.species?.[foe.speciesId]?.baseStats?.spe??85)+32;
const mineSpeed=p => (Number(p.stats?.spe)||100)*STAGE(p.boosts?.spe??0);
const futureThreat=(p,foe,data)=>{
  const species=data.species?.[foe.speciesId],types=foe.types??species?.types??[];
  const main=Math.max(Number(species?.baseStats?.atk??85),Number(species?.baseStats?.spa??85));
  const defenses=Math.min(Number(p.stats?.def)||100,Number(p.stats?.spd)||100);
  const matchup=Math.max(1,...types.map(t=>typeEffect(t,p.types,data)));
  // Conservative proxy: the opponent's actual moves and stats are secret.
  const raw=((22*75*main/Math.max(1,defenses))/50+2)*1.5*matchup;
  return Math.max(2,Math.min(100,100*raw/Math.max(1,p.maxHP)));
};
// Match the project's elementalDamage() rounding and Champions stats model.
// Opponent defense and HP must be estimated from PUBLIC base species data:
// unseen stat-point investment, nature, item and ability are never read.
function damageEstimate(p,foe,move,data,weather=null) {
  if(!move||move.category==='Status')return 0;
  let power=Math.floor(Number(move.basePower??0));
  if(power<=0&&move.ohko)power=150;
  if(power<=0)return 0;
  const special=move.category==='Special';
  const stats=data.species?.[foe.speciesId]?.baseStats??{};
  const defense=Math.max(1,Math.floor((Number(stats[special?'spd':'def'])||85)+32)*
    STAGE(foe.stages?.[special?'spd':'def']??0));
  let attack=Math.max(1,Math.floor((Number(p.stats?.[special?'spa':'atk'])||95)*
    STAGE(p.boosts?.[special?'spa':'atk']??0)));
  if(p.abilityId==='technician'&&power<=60)power=Math.floor(power*1.5);
  if((p.itemId==='choiceband'&&!special)||
     (p.itemId==='choicespecs'&&special))attack=Math.floor(attack*1.5);
  const effectiveness=typeEffect(move.type,foe.types??data.species?.[foe.speciesId]?.types,data);
  let weatherModifier=4096;
  if(weather==='sunnyday') {
    if(move.type==='Fire')weatherModifier=6144;
    if(move.type==='Water')weatherModifier=2048;
  } else if(weather==='raindance') {
    if(move.type==='Water')weatherModifier=6144;
    if(move.type==='Fire')weatherModifier=2048;
  }
  const percent=100*elementalDamage({
    attack,defense,power,roll:8,stab:p.types?.includes(move.type)??false,
    effectiveness,weatherModifier,
    burned:!special&&p.status==='brn'&&p.abilityId!=='guts',
    finalModifier:p.itemId==='lifeorb'?5325:4096,
  })/Math.max(1,(Number(stats.hp)||85)+85);
  return Math.max(0,Math.min(200,percent));
}
function movePressure(p,foe,data,weather=null) {
  return Math.max(0,...p.moves.map(slot=>damageEstimate(p,foe,data.moves?.[slot.id],data,weather)*
    accuracy(data.moves?.[slot.id])));
}
function switchValue(p,foe,data,hazards,trickRoom=false) {
  if(p.hp<=0)return -Infinity;
  const hp=pct(p.hp,p.maxHP),outgoing=movePressure(p,foe,data);
  const danger=futureThreat(p,foe,data);
  const speedGap=foeSpeed(foe,data)*STAGE(foe.stages?.spe??0)-mineSpeed(p);
  const roomBonus=trickRoom?Math.max(-20,Math.min(20,speedGap*0.20)):0;
  return 0.75*Math.min(outgoing,110)+hp*0.17-danger*0.68+roomBonus-
    (hazards?.length?8+hazards.length*3:0);
}
function bestSwitch(view,data) {
  const p=view.self.team,opponent=view.opponent;
  let best=null,bestScore=-Infinity;
  const ownHazards=view.field?.sideEffects?.[0]??[];
  const room=(view.field?.trickRoomTurns??0)>0;
  for(let i=0;i<p.length;i++) {
    if(i===view.self.active||p[i].hp<=0)continue;
    const score=switchValue(p[i],opponent,data,ownHazards,room);
    if(score>bestScore){bestScore=score;best=i;}
  }
  return best===null?null:{kind:'switch',index:best,score:bestScore};
}
function planScore(id,move,active,view,data,pressure,threat) {
  const foe=view.opponent,hp=pct(active.hp,active.maxHP),turn=view.turn;
  const ownSide=view.field?.sideEffects?.[0]??[];
  const enemySide=view.field?.sideEffects?.[1]??[];
  const faster=mineSpeed(active)>foeSpeed(foe,data)*STAGE(foe.stages?.spe??0);
  const trickRoom=(view.field?.trickRoomTurns??0)>0;
  const actsFirst=trickRoom?!faster:faster;
  let damage=damageEstimate(active,foe,move,data,view.field?.weather);
  if(move.category!=='Status' && damage>0) {
    const hit=accuracy(move),likelyKO=damage*0.86>=foe.hpPercent;
    const possibleKO=damage>=foe.hpPercent*0.9;
    let score=damage*hit;
    if(possibleKO)score+=hit*(likelyKO?43:22);
    if(Number(move.priority??0)>0 && !actsFirst)score+=Math.min(22,threat*0.45);
    if(Number(move.priority??0)<0 && hp<=threat)score-=35;
    if(move.recoil)score-=hp<35?18:5;
    if(move.drain)score+=(100-hp)*0.11;
    if(move.multihit)score+=3;
    if(id==='rapidspin'||id==='mortalspin')score+=ownSide.length*15;
    return score;
  }
  if(HEALS.has(id)||move.heal||move.flags?.heal) {
    if(hp>84)return -25;
    const healPotential=Math.min(100-hp,50);
    return healPotential*(threat<40?1.25:0.95) +
      (hp<threat&&actsFirst?14:0) - (hp<threat&&!actsFirst?30:0);
  }
  if(id==='revivalblessing')
    return view.self.team.some(x=>x.hp<=0)?67:-50;
  if(id==='batonpass') {
    const boost=active.boosts??{};
    const offensive=Math.max(boost.atk??0,boost.spa??0);
    const defensive=Math.max(0,boost.def??0)+Math.max(0,boost.spd??0);
    const usable=Math.max(0,offensive)+Math.max(0,boost.spe??0)*0.65+
      defensive*0.7;
    const receivers=view.self.team.some((p,i)=>i!==view.self.active&&
      p.hp>p.maxHP*0.35);
    // Defensive passers (e.g. Cosmic Power) need two stages before passing,
    // while one sharply boosted attacking stat can be useful immediately.
    const minimum=defensive>0&&offensive<1?2.5:1.5;
    return receivers&&usable>=minimum ? 26+usable*14+(hp<threat?11:0):-22;
  }
  // Field support is selected by fieldPreference() using public viability
  // rules, not by a fixed score competing with immediate attacks.
  if(id==='trickroom'||HAZARDS.has(id)||HAZARDS.has(move.sideCondition)||
     SCREENS.has(id)||SCREENS.has(move.sideCondition)||REMOVAL.has(id))
    return -35;
  if(STATUS.has(id)||move.status||move.volatileStatus==='leechseed') {
    if(foe.status && (STATUS.has(id)||move.status))return -30;
    if(foe.hpPercent<18)return 5;
    return 25+(hp>threat*1.2?8:0);
  }
  if(SETUPS.has(id)||move.boosts||move.self?.boosts) {
    const boosts=active.boosts??{};
    const stage=Math.max(boosts.atk??0,boosts.spa??0,boosts.spe??0,
      ((boosts.def??0)+(boosts.spd??0))/2);
    if(stage>=3||hp<threat*1.2||foe.hpPercent<Math.max(25,pressure))return -15;
    const defensivePassSetup=id==='cosmicpower'&&stage<2&&
      active.moves.some(slot=>slot.id==='batonpass');
    return 21+Math.min(17,pressure*0.16)-stage*9+
      (defensivePassSetup?20:0);
  }
  if(SHIELDS.has(id))return hp<=threat?12:1;
  if(id==='taunt'||id==='encore'||id==='haze'||id==='clearsmog')
    return foe.stages && Math.max(...Object.values(foe.stages))>=2?30:8;
  if(move.weather||move.terrain||move.pseudoWeather)
    return turn<12?16:3;
  if(id==='partingshot'||id==='uturn'||id==='voltswitch')
    return 15+(hp<threat?12:0);
  return 3;
}

export function chooseIntermediateAction(view,{data,rng,memory={}}={}) {
  const {self,opponent:foe}=view;
  if(self.confirmed||['wait','finished'].includes(view.requestType))return null;
  const active=self.team[self.active];
  const alternative=bestSwitch(view,data);
  if(view.requestType==='replacement'||view.requestType==='pivot'||active.hp<=0) {
    if(!self.needsSwitch&&active.hp>0)return null;
    if(!alternative)throw new Error('No living replacement available');
    // A Baton Pass pivot should prioritize a recipient that can use the
    // transferred boosts rather than discard the intended setup synergy.
    let selected=alternative.index;
    if(view.requestType==='pivot'&&memory.lastMoveId==='batonpass'&&
       view.turn-memory.lastMoveTurn<=1) {
      const buff=active.boosts??{};
      const enemy=view.opponent,ownHazards=view.field?.sideEffects?.[0]??[];
      let best=-Infinity;
      self.team.forEach((p,i)=>{
        if(i===self.active||p.hp<=0)return;
        const modified={...p,boosts:{...p.boosts,...buff}};
        const score=switchValue(modified,enemy,data,ownHazards,
           view.field?.trickRoomTurns>0);
        if(score>best){best=score;selected=i;}
      });
    }
    memory.lastSwitchTurn=view.turn;
    return {kind:'switch',index:selected};
  }
  if(view.requestType!=='turn')return null;
  const legal=active.moves.map((slot,index)=>({slot,index,move:data.moves?.[slot.id]}))
    .filter(({slot,move})=>!slot.disabled&&move);
  if(!legal.length)return {kind:'move',index:'struggle'};
  const available=legal.filter(({slot,move})=>
    !publiclyImmune(slot.id,move,active,foe,data));
  if(!available.length&&self.canSwitch&&alternative) {
    memory.lastSwitchTurn=view.turn;
    return {kind:'switch',index:alternative.index};
  }
  // If every move is ineffective and switching is impossible, still submit a
  // legal engine action, not an illegal Struggle.
  const choices=available.length?available:legal;
  const speeds=seenEnemySpeeds(view,data,memory);
  const threat=futureThreat(active,foe,data),hp=pct(active.hp,active.maxHP);
  const pressure=movePressure(active,foe,data,view.field?.weather);
  let best=null,bestScore=-Infinity,priorityField=null,priorityScore=-Infinity;
  let knockout=null,knockoutScore=-Infinity;
  for(const {slot,index,move} of choices) {
    let score=planScore(slot.id,move,active,view,data,pressure,threat)*
      (move.category==='Status'?accuracy(move):1);
    if(memory.lastMoveId===slot.id && view.turn-memory.lastMoveTurn<=1 &&
        (move.category==='Status'||SHIELDS.has(slot.id)))score-=12;
    // A seeded jitter resolves equal-score actions without nondeterminism.
    score+=(rng?rng.integer(1000):0)*0.000001;
    if(score>bestScore){bestScore=score;best={kind:'move',index};}
    const preference=fieldPreference(slot.id,move,active,view,data,threat,speeds);
    if(preference!==null&&preference>priorityScore) {
      priorityScore=preference;priorityField={kind:'move',index};
    }
    if(move.category!=='Status'&&accuracy(move)>=0.7) {
      const damage=damageEstimate(active,foe,move,data,view.field?.weather);
      if(damage*0.86>=foe.hpPercent&&damage*accuracy(move)>knockoutScore) {
        knockoutScore=damage*accuracy(move);knockout={kind:'move',index};
      }
    }
  }
  // Forced rules beat ordinary point estimates; an immediate KO beats setup.
  // A field move is useful only when its public viability test above passes.
  if(knockout)best=knockout;
  else if(priorityField)best=priorityField;
  if(!knockout&&!priorityField&&self.canSwitch&&alternative&&hp>10&&
      view.turn-(memory.lastSwitchTurn??-100)>2) {
    const current=switchValue(active,foe,data,view.field?.sideEffects?.[0]??[],
        view.field?.trickRoomTurns>0);
    if(alternative.score-current>22 && threat>25 &&
        (pressure<32 || hp<threat*1.8) && bestScore<55) {
      memory.lastSwitchTurn=view.turn;
      return {kind:'switch',index:alternative.index};
    }
  }
  if(!best)return {kind:'move',index:'struggle'};
  const selected=active.moves[best.index].id;
  if(selected==='revivalblessing') {
    const target=self.team.findIndex((p,i)=>i!==self.active&&p.hp<=0);
    if(target>=0)best.targetIndex=target;
  }
  if(self.canMega) {
    // Mega is once per battle. Avoid consuming it on a doomed, slower
    // attacker when a surviving teammate can use the slot later.
    const priority=Number(data.moves?.[selected]?.priority??0);
    const selfSpeed=mineSpeed(active),opponentSpeed=foeSpeed(foe,data)*
      STAGE(foe.stages?.spe??0);
    const faster=selfSpeed>=opponentSpeed;
    const actsFirst=priority>0 || (view.field?.trickRoomTurns>0?!faster:faster);
    const reserveMega=self.team.some((p,i)=>i!==self.active&&
      p.hp>0&&p.megaCandidate);
    const doomed=!actsFirst&&hp<threat*0.85;
    const weakAndWounded=reserveMega&&hp<40&&pressure<25;
    if(!doomed&&!weakAndWounded)best.mega=true;
  }
  memory.lastMoveId=selected;memory.lastMoveTurn=view.turn;
  return best;
}
