// Issue #737 P2: deterministic, publicly observed battle action ranking.
// No hidden opponent roster, move, item or ability is ever accepted.
import {effectiveness,elementalDamage} from '../battle-mechanics.js';

const stage=n=>n>=0?(2+Math.min(6,n))/2:2/(2-Math.max(-6,n));
const hpPercent=p=>100*p.hp/Math.max(1,p.maxHP);
const HEAL=new Set('recover roost slackoff rest synthesis moonlight morningsun softboiled milkdrink strengthsap'.split(' '));
const SETUP=new Set('swordsdance nastyplot dragondance quiverdance calmmind bulkup shellsmash agility irondefense amnesia coil workup shiftgear honeclaws cosmicpower'.split(' '));
const HAZARD=new Set('stealthrock spikes toxicspikes stickyweb'.split(' '));
const REMOVAL=new Set('rapidspin defog mortalspin tidyup courtchange'.split(' '));
const SCREEN=new Set('reflect lightscreen auroraveil'.split(' '));
const STATUS=new Set('toxic thunderwave willowisp spore sleeppowder glare yawn taunt encore'.split(' '));
// Taunt, Encore and stat debuffs remain useful even if the foe already has
// a primary status condition. Only another primary status attempt is blocked.
const PRIMARY_STATUS=new Set('toxic thunderwave willowisp spore sleeppowder glare yawn'.split(' '));
const accuracy=move=>move?.accuracy===true?1:Math.max(0,Math.min(1,Number(move?.accuracy??100)/100));
const speed=p=>(p.stats?.spe??100)*stage(p.boosts?.spe??0)*(p.itemId==='choicescarf'?1.5:1);
const opponentSpeed=(foe,data)=>((data.species?.[foe.speciesId]?.baseStats?.spe??85)+25)
  *stage(foe.stages?.spe??0);
const side=(view,seat)=>view.field?.sideEffects?.[seat]??[];
const has=(effects,id)=>effects.some(e=>e.id===id);
const layers=(effects,id)=>effects.find(e=>e.id===id)?.layers??0;

// Approximation using only public defender species/typing; not true set info.
// Nonstandard effects and unknown information return null, never fake 0 damage.
export function publicMoveDamage(actor,foe,move,data,field={}) {
  if(!move||move.category==='Status'||!(move.basePower>0)||
     move.overrideOffensivePokemon||move.ohko||
     move.referenceHandlers?.length||
     !Array.isArray(foe?.types)||foe.types.length===0)return null;
  const base=data.species?.[foe.speciesId]?.baseStats;
  if(!base)return null;
  const special=move.category==='Special',atkKey=special?'spa':'atk',defKey=special?'spd':'def';
  const types=actor.abilityId==='scrappy'&&['Normal','Fighting'].includes(move.type)
    ?foe.types.filter(t=>t!=='Ghost'):foe.types;
  const factor=types.length?effectiveness(move.type,types,data.typeChart):1;
  if(factor===0)return 0;
  let atk=Math.floor((actor.stats?.[atkKey]??95)*stage(actor.boosts?.[atkKey]??0));
  if(!special&&actor.itemId==='choiceband'||special&&actor.itemId==='choicespecs')
    atk=Math.floor(atk*1.5);
  const def=Math.max(1,Math.floor((base[defKey]+25)*stage(foe.stages?.[defKey]??0)));
  const rain=field?.weather==='raindance',sun=field?.weather==='sunnyday';
  const weather=(rain&&move.type==='Water'||sun&&move.type==='Fire')?6144:
    (rain&&move.type==='Fire'||sun&&move.type==='Water')?2048:4096;
  const damage=elementalDamage({
    attack:Math.max(1,atk),defense:def,power:Math.floor(move.basePower),roll:8,
    stab:actor.types?.includes(move.type)??false,effectiveness:factor,
    weatherModifier:weather,
    burned:!special&&actor.status==='brn'&&actor.abilityId!=='guts',
    finalModifier:actor.itemId==='lifeorb'?5325:4096,
  });
  return Math.min(200,100*damage/(base.hp+75));
}
function threat(p,foe,data) {
  const base=data.species?.[foe.speciesId]?.baseStats;
  if(!base||!Array.isArray(foe.types))return 30;
  const weak=Math.max(1,...foe.types.map(type=>
    effectiveness(type,p.types??['Normal'],data.typeChart)));
  const defense=Math.max(1,Math.min(p.stats?.def??100,p.stats?.spd??100));
  return Math.min(100,100*((22*75*(Math.max(base.atk,base.spa)+25)/defense)/50+2)
    *weak*1.5/Math.max(1,p.maxHP));
}
const offense=(p,foe,data,field)=>Math.max(0,...(p.moves??[])
  .map(slot=>publicMoveDamage(p,foe,data.moves[slot.id??slot],data,field)??0));
const grounded=p=>p.itemId==='ironball'||(
  !p.types?.includes('Flying')&&p.abilityId!=='levitate'&&p.itemId!=='airballoon');
// Only HP lost immediately on entry can rule out a voluntary switch.
// Toxic Spikes changes status and has no immediate entry-HP damage.
function entryHPDamage(p,effects,data) {
  if(p.abilityId==='magicguard')return 0;
  let loss=has(effects,'stealthrock')?12.5*effectiveness(
    'Rock',p.types??['Normal'],data.typeChart):0;
  if(grounded(p)) {
    const n=layers(effects,'spikes');
    loss+=n===0?0:n===1?12.5:n===2?100/6:25;
  }
  return loss;
}
function toxicSpikesRisk(p,effects) {
  if(!has(effects,'toxicspikes')||!grounded(p)||p.status||
    p.types?.some(t=>t==='Steel'||t==='Poison')||
    p.itemId==='lumberry'||p.abilityId==='immunity')return 0;
  // A future poison tick is a utility cost, never instant entry damage.
  return layers(effects,'toxicspikes')>=2?12:8;
}

function validatedForecast(matchup,moveId) {
  // A P1 assessment must have been produced from the authorized public belief.
  // If unavailable/inconsistent, do not pretend its estimate is reliable.
  if(matchup?.modelVersion!=='matchup-p1-v1'||
     !Array.isArray(matchup.moveEstimates))return null;
  const row=matchup.moveEstimates.find(entry=>entry.id===moveId);
  if(!row||!Number.isFinite(row.expectedPercent)||
     !Number.isFinite(row.minPercent)||
     !Number.isFinite(row.maxPercent)||
     !Number.isFinite(row.immuneCandidateWeight)||
     row.minPercent<0||row.minPercent>row.expectedPercent||
     row.expectedPercent>row.maxPercent||
     row.immuneCandidateWeight<0||row.immuneCandidateWeight>1)return null;
  return row;
}

function scoreMove(p,foe,id,move,view,data,memory,matchup) {
  const hp=hpPercent(p),incoming=threat(p,foe,data),
    own=side(view,0),enemy=side(view,1),enemies=foe.remaining??3;
  const forecast=validatedForecast(matchup,id);
  // When P1 provides a candidate-weighted estimate, use it rather than an
  // optimistic species-only calculation that cannot account for Levitate.
  const damage=forecast?.expectedPercent??
    publicMoveDamage(p,foe,move,data,view.field);
  const faster=speed(p)>opponentSpeed(foe,data);
  const first=(view.field?.trickRoomTurns??0)>0?!faster:faster;
  let category='utility',score=0;
  if(damage!==null) {
    category='attack';score=damage*accuracy(move);
    // A forecast is KO-secure only when EVERY catalog hypothesis survives
    // the accuracy threshold and none is immune. "Expected KO" is weaker.
    const guaranteedByForecast=forecast&&
      forecast.minPercent>=foe.hpPercent*1.12&&
      forecast.immuneCandidateWeight===0&&accuracy(move)>=0.95;
    if(guaranteedByForecast)score+=80;
    else if(damage>=foe.hpPercent*1.12&&accuracy(move)>=0.95)
      score+=forecast?12:35; // raw public type-only estimate is not certain
    else if(damage>=foe.hpPercent*0.9)score+=18*accuracy(move);
    if(move.priority>0&&!first)score+=12;
    if(move.recoil)score-=hp<35?18:5;
    if(move.drain)score+=(100-hp)*0.1;
    if(REMOVAL.has(id)&&own.length)score+=own.length*12;
  } else if(id==='revivalblessing') {
    category='teamRecovery';
    score=view.self.team.some((x,i)=>i!==view.self.active&&x.hp<=0)?52:-95;
  } else if(id==='healingwish') {
    category='teamRecovery';
    // Healing Wish faints its user. Without a living teammate needing
    // recovery it is a self-KO with no compensating benefit.
    const recipients=view.self.team.filter((ally,i)=>
      i!==view.self.active&&ally.hp>0);
    const benefit=Math.max(0,...recipients.map(ally=>
      100*(1-ally.hp/Math.max(1,ally.maxHP))+(ally.status?20:0)));
    score=recipients.length&&benefit>=20
      ?Math.min(40,benefit*0.7)-incoming*0.2:-95;
  } else if(HEAL.has(id)||move.heal) {
    category='recovery';
    score=hp>=90?-95:Math.min(50,100-hp)*1.2-incoming*0.45;
    if(!first&&incoming>=hp)score=-95;
  } else if(HAZARD.has(id)||HAZARD.has(move.sideCondition)) {
    category='hazard';
    const key=HAZARD.has(id)?id:move.sideCondition,max=key==='spikes'?3:key==='toxicspikes'?2:1;
    score=enemies<=1||layers(enemy,key)>=max?-95:
      27+(enemies-1)*8-incoming*0.4;
  } else if(SCREEN.has(id)||SCREEN.has(move.sideCondition)) {
    category='screen';
    const key=SCREEN.has(id)?id:move.sideCondition;
    score=has(own,key)||(key==='auroraveil'&&!['hail','snowscape'].includes(view.field?.weather))
      ?-95:29+Math.min(18,incoming*0.2);
  } else if(REMOVAL.has(id)) {
    category='hazardRemoval';score=own.some(x=>HAZARD.has(x.id))?35+own.length*8:-70;
  } else if(SETUP.has(id)||
    (move.target==='self'&&Object.values(move.boosts??{}).some(v=>v>0))||
    Object.values(move.self?.boosts??{}).some(v=>v>0)) {
    category='setup';
    const b=p.boosts??{},level=Math.max(b.atk??0,b.spa??0,b.spe??0,
      ((b.def??0)+(b.spd??0))/2);
    score=29+0.1*offense(p,foe,data,view.field)-level*13-incoming*0.3;
    if(hp<=incoming*1.05||level>=3)score=-90;
  } else if(STATUS.has(id)||move.status||
    (move.target!=='self'&&move.boosts)) {
    category='status';
    const attemptsPrimaryStatus=!!move.status||PRIMARY_STATUS.has(id);
    score=attemptsPrimaryStatus&&foe.status?-80:26-incoming*0.25;
    if(move.flags?.powder&&foe.types?.includes('Grass'))score=-95;
  } else if(id==='batonpass') {
    category='pivot';
    score=Object.values(p.boosts??{}).some(x=>x>0)
      &&view.self.team.some((x,i)=>i!==view.self.active&&x.hp>0)?30:-75;
  } else if(id==='protect'){category='protection';score=5;}
  else score=8;
  // Our custom rule explicitly loses the round if the last Pokemon
  // uses Explosion and both final Pokemon faint. With no living reserve,
  // Explosion always removes our final participant, so do not treat its
  // apparent one-turn damage as a valuable KO. Other self-destruct moves
  // have unresolved rules in SPEC.md; do not silently extend this policy.
  if(id==='explosion'&&view.self.team.filter(ally=>ally.hp>0).length===1)
    score=-95;
  if(category!=='attack'&&memory.lastMoveId===id
    &&view.turn-(memory.lastMoveTurn??-100)<=2)score-=25;
  if(category!=='attack')score*=accuracy(move);
  return {category,score};
}

// Every emitted action comes from a legal slot in the authoritative AI view.
export function rankStrategyActions(view,{
  data,memory={},matchup=null,switchMatchups=null,
}={}) {
  if(!view?.self||!view?.opponent||!data?.moves||!data?.typeChart)
    throw new TypeError('Expected authorized AI view and battle definitions');
  const self=view.self,foe=view.opponent,active=self.team?.[self.active];
  if(!active||self.confirmed||['wait','finished'].includes(view.requestType))return [];
  // A replacement phase is shared by both seats, but only a fainted active
  // member must act. Battle.submit explicitly rejects a healthy seat here.
  if(view.requestType==='replacement'&&active.hp>0&&!self.needsSwitch)
    return [];
  if(view.requestType==='pivot'&&active.hp>0&&!self.needsSwitch)
    return [];
  const forced=['replacement','pivot'].includes(view.requestType)||active.hp<=0;
  if(!forced&&view.requestType!=='turn')return [];
  const results=[],own=side(view,0);
  if(forced||self.canSwitch)for(let i=0;i<self.team.length;i++){
    const p=self.team[i];
    if(i===self.active||p.hp<=0)continue;
    const immediateDamage=entryHPDamage(p,own,data);
    const cost=immediateDamage+toxicSpikesRisk(p,own);
    // If switching is mandatory, a living reserve remains selectable even
    // when entry hazards may immediately faint it; the engine resolves that.
    // Poison from Toxic Spikes is a future risk, not a reason to hide a
    // legal low-HP voluntary switch.
    if(!forced&&immediateDamage>=hpPercent(p))continue;
    const recent=Number.isInteger(memory.lastSwitchTurn)
      &&view.turn-memory.lastSwitchTurn<=2?22:0;
    // Optional P1 evaluations are indexed by OWN reserve slot; no hidden
    // enemy team/set is accepted. Unknown projections add no fake safety.
    const projection=switchMatchups?.[i]?.switchIn;
    const safety=projection?.status==='evaluated'&&
      Number.isFinite(projection.score)
      ?Math.max(-1,Math.min(1,projection.score))*35:0;
    results.push({action:{kind:'switch',index:i},category:'switch',
      score:10+0.6*offense(p,foe,data,view.field)
        +0.2*hpPercent(p)-0.4*threat(p,foe,data)-cost-recent+safety});
  }
  if(!forced)for(let i=0;i<(active.moves??[]).length;i++){
    const slot=active.moves[i],move=data.moves[slot.id];
    if(slot.disabled||!move)continue;
    const action={kind:'move',index:i};
    if(slot.id==='revivalblessing'){
      const target=self.team.findIndex((p,j)=>j!==self.active&&p.hp<=0);
      if(target<0)continue;
      action.targetIndex=target;
    }
    if(self.canMega)action.mega=true;
    results.push({action,...scoreMove(active,foe,slot.id,move,view,data,memory,matchup)});
  }
  if(!forced&&!results.length)results.push({
    action:{kind:'move',index:'struggle'},category:'attack',score:0});
  results.sort((a,b)=>b.score-a.score||
    Number(a.action.kind==='switch')-Number(b.action.kind==='switch')||
    (a.action.index==='struggle'?999:a.action.index)
      -(b.action.index==='struggle'?999:b.action.index));
  return results;
}
export function chooseStrategyAction(view,{
  data,memory={},matchup=null,switchMatchups=null,
}={}){
  const choices=rankStrategyActions(view,{data,memory,matchup,switchMatchups});
  if(!choices.length)return null;
  const action=choices[0].action;
  if(action.kind==='switch')memory.lastSwitchTurn=view.turn;
  else {
    memory.lastMoveId=typeof action.index==='number'
      ?view.self.team[view.self.active].moves[action.index].id:'struggle';
    memory.lastMoveTurn=view.turn;
  }
  return action;
}
