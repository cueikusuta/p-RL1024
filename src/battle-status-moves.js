import { requireRule } from './draft.js';
import { effectiveMove, changeBoosts, healHP, bellyDrum, painSplit, weatherHeal, makeSubstitute, scaledStat, boostStage, whiteHerb, resetBoosts } from './battle-moves.js';
import { setStatus, clearStatus, rest, statusBerry } from './battle-status.js';
import { consumeHealingBerry, markUnburden, announceHeldItemEvent, isFormLockedHeldItem } from './battle-mechanics.js';
import { blocksStatusMove, isProtected, hitsAccuracy, protectMove } from './battle-hit.js';
import { grounded, semiInvulnerable, tryAbilityImmunity, moveEffectiveness, drainHealth } from './battle-effects.js';
import { setTerrain, setWeather, toggleTrickRoom, placeHazard, setSideCondition, useWish, useLeechSeed, useYawn, defogField, electricSeed, grassySeed, psychicSeed, updateMimicry } from './battle-field.js';
import { applyTaunt, applyEncore, applyDisable, sleepTalkMove, mentalHerb } from './battle-actions.js';
import { prepareProtean } from './battle-secondary.js';
import { snapshotConditions, notifyConditionChanges } from './battle-messages.js';

export const CONNECTED_STATUS_MOVES=new Set('agility auroraveil babydolleyes bellydrum bulkup calmmind charm coil cottonspore cottonguard defog destinybond disable dragondance eerieimpulse electricterrain encore endure featherdance glare growth haze healbell healingwish hypnosis irondefense kingsshield lightscreen leechseed minimize morningsun moonlight nastyplot painsplit noretreat octolock perishsong protect psychicterrain quiverdance recover recycle reflect rest revivalblessing roost shedtail shellsmash slackoff sleeppowder sleeptalk spikyshield spikes stealthrock stunspore stickyweb strengthsap substitute sunnyday swordsdance synthesis tailwind taunt thunderwave toxic toxicspikes poisonpowder willowisp wish yawn trickroom roar whirlwind batonpass partingshot'.split(' '));
CONNECTED_STATUS_MOVES.add('trick');
CONNECTED_STATUS_MOVES.add('curse');
CONNECTED_STATUS_MOVES.add('memento');
CONNECTED_STATUS_MOVES.add('chillyreception');
CONNECTED_STATUS_MOVES.add('soak');
CONNECTED_STATUS_MOVES.add('transform');
CONNECTED_STATUS_MOVES.add('tidyup');
CONNECTED_STATUS_MOVES.add('silktrap');
CONNECTED_STATUS_MOVES.add('clangoroussoul');
for(const id of ['cosmicpower','grassyterrain','milkdrink','acidarmor'])CONNECTED_STATUS_MOVES.add(id);
for(const id of ['raindance','sandstorm','snowscape','metalsound','screech'])CONNECTED_STATUS_MOVES.add(id);
export function exchangeHeldItems(actor,target,field,{ignoreAbility=false}={}) {
  if(actor.megaStoneId && actor.itemId===actor.megaStoneId || target.megaStoneId && target.itemId===target.megaStoneId || isFormLockedHeldItem(actor) || isFormLockedHeldItem(target))return false;
  if(!actor.hp || !target.hp || !actor.itemId && !target.itemId || !ignoreAbility && target.abilityId==='stickyhold')return false;
  const actorItem=actor.itemId??'',targetItem=target.itemId??'';
  [actor.itemId,target.itemId]=[targetItem,actorItem];
  announceHeldItemEvent(actor,{kind:'exchange',item:targetItem,other:target,otherItem:actorItem});
  if(actorItem&&!targetItem)markUnburden(actor);
  if(targetItem&&!actorItem)markUnburden(target);
  for(const pokemon of [target,actor]) {
    // The initial catalog excludes all restricted form items. Choice Scarf's
    // Start clears its prior lock; losing it also releases the lock.
    delete pokemon.volatiles.choicelock;
    whiteHerb(pokemon);electricSeed(pokemon,field);grassySeed(pokemon,field);psychicSeed(pokemon,field);
    mentalHerb(pokemon);statusBerry(pokemon);consumeHealingBerry(pokemon);
  }
  return true;
}
function checkStatusMovePreconditions({id,actor,target,teams,seat,reviveIndex}) {
  if(id==='noretreat' && actor.volatiles?.noretreat)return {success:false};
  if(id==='jawlock' && (actor.volatiles?.jawlock || target.volatiles?.jawlock))return {success:false};
  if(id==='octolock' && target.types.includes('Ghost'))return {success:false};
  if(id==='soak' && target.types.length===1 && target.types[0]==='Water')return {success:false};
  if(id==='revivalblessing' && (!Number.isInteger(reviveIndex) || !teams[seat][reviveIndex] || teams[seat][reviveIndex]===actor || teams[seat][reviveIndex].hp>0))return {success:false};
  return null;
}
function getStatusMoveTargeting(move,actor,target) {
  const self=move.target==='self';
  const recipient=self ? actor : target;
  return {self,recipient};
}
function checkStatusMoveTarget({id,definition,move,actor,target,field,seat,teams,active,chart,definitions,rng,turn,ignoreAbility,self,targetsFoe,reflects,priority,remainingActions,targetWillMove,overrideAction,bounced}) {
  if(targetsFoe && field.terrain==='psychicterrain' && priority>0.1 && grounded(target,{ignoreAbility}))return {success:false,blocked:true};
  if(targetsFoe && !ignoreAbility && target.abilityId==='aromaveil' &&
    ['attract','disable','encore','healblock','taunt','torment'].includes(id))return {success:false,blocked:true};
  if(!ignoreAbility && target.abilityId==='armortail' && priority>0.1 &&
    (targetsFoe && move.target!=='foeSide' ||
      move.target==='all' && ['perishsong','flowershield','rototiller'].includes(id)))return {success:false,blocked:true};
  if(targetsFoe && isProtected(move,actor,target,turn)) {
    if(target.volatiles?.silktrap?.turn===turn && move.flags?.contact && actor.hp)
      changeBoosts(actor,{spe:-1},{source:target,effect:'silktrap'});
    return {success:false,blocked:true,notFail:true};
  }
  if(!bounced && reflects && move.flags?.reflectable && target.abilityId==='magicbounce' && !ignoreAbility && !semiInvulnerable(target)) {
    const reflected=executeStatusMove({id,definition,actor:target,target:actor,field,seat:1-seat,teams,active,chart,definitions,rng,turn,
      remainingActions,targetWillMove:false,overrideAction,bounced:true,priority:move.priority??0});
    return {...reflected,bounced:true};
  }
  if(targetsFoe && tryAbilityImmunity(move,actor,target,{ignoreAbility}))return {success:false,immune:true};
  if(!bounced && targetsFoe && actor.abilityId==='prankster' && target.types.includes('Dark'))return {success:false,immune:true};
  if (!self && !['all','allyTeam','foeSide','allySide'].includes(move.target)) {
    if (move.flags?.powder && target.types.includes('Grass'))return {success:false,immune:true};
    if (move.ignoreImmunity===false && !moveEffectiveness(move,actor,target,chart))return {success:false,immune:true};
    if (!hitsAccuracy({...move,id},actor,target,rng,{weather:field.weather,ignoreAbility}))return {success:false,missed:true};
    if(id!=='defog' && blocksStatusMove(move,actor,target,turn))return {success:false,blocked:true};
  }
  return null;
}
function applyStatusMoveEffect({id,move,actor,target,recipient,self,field,seat,teams,active,chart,definitions,rng,turn,transform,reviveIndex,remainingActions,targetWillMove,overrideAction,ignoreAbility,effectiveWeather,statusOptions}) {
  const conditionBefore=new Map([actor,target].map(p=>[p,snapshotConditions(p)]));
  let success=false,calledMove=null,selfDamage=0,notFail=false;
  const hpChanges=[];
  // Reaching this point means the opponent-facing move passed Protect,
  // Substitute, ability immunity, type immunity, and accuracy checks.
  const landed=id==='memento' && !self;
  if (id==='transform')success=transform();
  else if (id==='partingshot') {
    success=Object.keys(changeBoosts(recipient,{atk:-1,spa:-1},{source:actor,ignoreAbility,effect:id})).length>0 || recipient.abilityId==='mirrorarmor';
  }
  else if (id==='tidyup') {
    let removed=false;
    for(const pokemon of active)if(pokemon.volatiles?.substitute){delete pokemon.volatiles.substitute;removed=true;}
    for(const side of field.sides)for(const condition of ['spikes','toxicspikes','stealthrock','stickyweb'])if(side[condition]){delete side[condition];removed=true;}
    const boosted=changeBoosts(actor,{atk:1,spe:1},{effect:id});
    success=Object.keys(boosted).length>0||removed;
  }
  else if (id==='clangoroussoul') {
    if(actor.maxHP!==1 && actor.hp>actor.maxHP*33/100) {
      const boosted=changeBoosts(actor,move.boosts,{source:actor,effect:id});
      success=Object.keys(boosted).length>0;
      if(success) {
        selfDamage=Math.max(1,Math.floor(actor.maxHP*33/100));
        actor.hp-=selfDamage;consumeHealingBerry(actor);
      }
    }
  }
  else if (move.boosts) {
    success=Object.keys(changeBoosts(recipient,move.boosts,{source:actor,ignoreAbility,effect:id})).length>0;
  }
  else if (move.status)success=setStatus(recipient,move.status,rng,statusOptions);
  else if (id==='rest')success=rest(actor,rng,statusOptions);
  else if (move.heal) {
    success=healHP(actor,Math.round(actor.maxHP*move.heal[0]/move.heal[1]))>0;
    if (success && id==='roost' && actor.types.includes('Flying')) {
      actor.volatiles.roost={types:[...actor.types]};
      actor.types=actor.types.filter(type=>type!=='Flying');if(!actor.types.length)actor.types=['Normal'];
    }
  }
  else if (id==='morningsun' || id==='moonlight' || id==='synthesis')success=weatherHeal(actor,actor.abilityId==='megasol'?'sunnyday':effectiveWeather)>0;
  else if (id==='bellydrum') {
    success=bellyDrum(actor,{effect:id});
    // The cost is paid before a possible healing berry. The net HP delta
    // cannot reliably identify the source of an actually paid cost.
    if(success)selfDamage=Math.floor(actor.maxHP/2);
  }
  else if (id==='painsplit')success=painSplit(actor,target,(pokemon,delta)=>{
    if(delta)hpChanges.push({target:pokemon,delta});
  });
  else if (id==='substitute') {
    success=makeSubstitute(actor);notFail=!success;
    if(success)selfDamage=Math.floor(actor.maxHP/4);
  }
  else if (['protect','endure','kingsshield','silktrap','spikyshield'].includes(id))success=protectMove(actor,id,turn,rng,{remainingActions});
  else if (id==='taunt')success=applyTaunt(target,{willMove:targetWillMove,causeMoveId:id});
  else if (id==='trick')success=exchangeHeldItems(actor,target,field,{ignoreAbility});
  else if(id==='recycle') {
    if(!actor.itemId && actor.lastItemId && actor.lastItemId!==actor.megaStoneId) {
      const item=actor.lastItemId;actor.itemId=item;actor.lastItemId='';success=true;
      announceHeldItemEvent(actor,{item,kind:'recycled'});
      // The pinned Showdown Recycle -> setItem path immediately runs the
      // restored item's Start event. Terrain Seeds must react here too.
      electricSeed(actor,field);grassySeed(actor,field);psychicSeed(actor,field);
    }
  }
  else if (id==='soak') {recipient.types=['Water'];success=true;}
  else if (id==='jawlock') {
    actor.volatiles.jawlock={sourceId:target.instanceId};target.volatiles.jawlock={sourceId:actor.instanceId};success=true;
  }
  else if (id==='octolock') {
    if(!target.volatiles.octolock){target.volatiles.octolock={sourceId:actor.instanceId};success=true;}
  }
  else if (id==='batonpass')success=teams[seat].some(member=>member!==actor && member.hp>0) && !actor.volatiles.commanded;
  else if (id==='shedtail') {
    if(teams[seat].some(member=>member!==actor && member.hp>0) && !actor.volatiles.commanded && actor.hp>Math.ceil(actor.maxHP/2) && !actor.volatiles.substitute) {
      const cost=Math.ceil(actor.maxHP/2);actor.hp-=cost;
      actor.volatiles.substitute={hp:Math.floor(actor.maxHP/4)};
      consumeHealingBerry(actor);selfDamage=cost;success=true;
    }
  }
  else if (id==='healingwish') {
    if(teams[seat].some(member=>member!==actor && member.hp>0)) {
      field.healingWishes[seat]={hp:actor.maxHP,turn};selfDamage=actor.hp;actor.hp=0;success=true;
    }
  }
  else if (id==='revivalblessing') {
    const revived=teams[seat][reviveIndex];revived.hp=Math.max(1,Math.floor(revived.maxHP/2));revived.status='';revived.statusState={};revived.volatiles={};resetBoosts(revived);success=true;
  }
  else if (id==='curse') {
    if(!actor.types.includes('Ghost'))success=Object.keys(changeBoosts(actor,{spe:-1,atk:1,def:1},{effect:id})).length>0;
    else if(!target.volatiles.curse) {
      selfDamage=Math.min(actor.hp,Math.max(1,Math.floor(actor.maxHP/2)));
      actor.hp-=selfDamage;consumeHealingBerry(actor);
      target.volatiles.curse=true;success=true;
    }
  }
  else if (id==='encore') {
    const result=applyEncore(target,definitions,{willMove:targetWillMove,causeMoveId:id});success=result.applied;
    if(result.override)overrideAction(result.override);
  }
  else if(id==='disable')success=applyDisable(target,{willMove:targetWillMove,activeMove:targetWillMove,causeMoveId:id});
  else if(id==='haze') {
    for(const pokemon of active)if(pokemon.hp)resetBoosts(pokemon,{notify:true});
    success=true;
  }
  else if(id==='perishsong') {
    for(const pokemon of active)if(pokemon.hp && !tryAbilityImmunity(move,actor,pokemon,{ignoreAbility}) && !(actor.abilityId==='prankster'&&pokemon!==actor&&pokemon.types.includes('Dark')) && !pokemon.volatiles.perishsong) {
      pokemon.volatiles.perishsong={duration:4};success=true;
    }
  }
  else if(id==='destinybond') {
    if(actor.volatiles.destinybond)delete actor.volatiles.destinybond;
    else {actor.volatiles.destinybond={turn};success=true;}
  }
  else if (id==='healbell') {
    for(const pokemon of teams[seat])if(pokemon===actor || !['soundproof','goodasgold'].includes(pokemon.abilityId))success=clearStatus(pokemon)||success;
  }
  else if (id==='leechseed')success=useLeechSeed(target,seat);
  else if (id==='yawn')success=useYawn(target,turn,field,chart,{ignoreAbility});
  else if (move.slotCondition==='Wish')success=useWish(field,seat,actor,turn);
  else if (move.weather) {
    const weatherId={SunnyDay:'sunnyday',RainDance:'raindance',Sandstorm:'sandstorm',sun:'sunnyday',rain:'raindance',sand:'sandstorm',snow:'snowscape'}[move.weather]??move.weather;
    success=setWeather(field,weatherId,actor);
    if(id==='chillyreception' && !success && !actor.volatiles?.commanded)
      success=teams[seat].some(member=>member!==actor && member.hp>0);
  }
  else if (move.terrain)success=setTerrain(field,move.terrain,actor,active);
  else if (id==='sunnyday')success=setWeather(field,'sunnyday',actor);
  else if (id==='electricterrain')success=setTerrain(field,'electricterrain',actor,active);
  else if (move.pseudoWeather==='trickroom')success=toggleTrickRoom(field);
  else if (move.sideCondition)success=['tailwind','reflect','lightscreen','auroraveil'].includes(move.sideCondition)
    ? setSideCondition(field,move.target==='foeSide'?1-seat:seat,move.sideCondition,actor) : placeHazard(field,1-seat,move.sideCondition);
  else if (id==='defog') {
    if (!target.volatiles.substitute || actor.abilityId==='infiltrator')success=Object.keys(changeBoosts(target,{evasion:-1},{source:actor,ignoreAbility,effect:id})).length>0;
    success=defogField(field,1-seat)||success;
    for(const pokemon of active)updateMimicry(pokemon,field.terrain);
  }
  else if (id==='strengthsap') {
    if(boostStage(target,'atk')>-6) {
      const amount=scaledStat(target.stats.atk,boostStage(target,'atk'));
      const changed=changeBoosts(target,{atk:-1},{source:actor,ignoreAbility,effect:id});
      success=Boolean(drainHealth(actor,target,amount,{ignoreAbility}) || Object.keys(changed).length);
    }
  }
  else if (id==='sleeptalk') {calledMove=sleepTalkMove(actor,definitions,rng);success=Boolean(calledMove);}
  if(id==='minimize' && !recipient.volatiles?.minimize) {
    recipient.volatiles??={};recipient.volatiles.minimize=true;success=true;
  }

  for(const [pokemon,before] of conditionBefore)notifyConditionChanges(pokemon,before,{causeMoveId:id});
  return {success,calledMove,selfDamage,notFail,landed,...(hpChanges.length?{hpChanges}:{})};
}
function finishStatusMove({id,actor,target,recipient,move,success,calledMove,selfDamage,hpChanges,notFail,landed,bounced,turn,actorAlreadyTrapped}) {
  if(success && id==='silktrap') {recipient.volatiles??={};recipient.volatiles.silktrap={turn};}
  if(success && id==='noretreat' && move.volatileStatus && !actorAlreadyTrapped) {
    recipient.volatiles??={};recipient.volatiles[move.volatileStatus]=true;
  }
  if(success){if(id!=='memento')whiteHerb(actor);whiteHerb(target);}
  if(id==='memento' && landed) {
    // Memento selfdestructs after a valid hit even when the target cannot
    // lower either stat or is already at the stage floor.
    success=true;
    selfDamage=actor.hp;
    actor.hp=0;
  }
  const switchMode=success && !bounced && id!=='revivalblessing' ? (id==='batonpass'?'copyvolatile':id==='shedtail'?'shedtail':move.selfSwitch===true||move.selfSwitch==='copyvolatile'?move.selfSwitch:null) : null;
  return {success,calledMove,selfDamage,...(hpChanges?.length?{hpChanges}:{}),...(notFail?{notFail:true}:{}),...(switchMode?{switchMode}:{}),...(id==='revivalblessing'&&success?{revival:true}:{}),...(id==='healingwish'&&success?{healingWish:true}:{}),...(id==='memento'?{landed}:{})};
}
export function executeStatusMove({id,definition,actor,target,field,seat,teams,active,chart,definitions,rng,turn,transform=()=>false,reviveIndex=null,remainingActions=true,targetWillMove=true,overrideAction=()=>{},actorAlreadyTrapped=false,bounced=false,priority=(definition.priority??0)+(actor.abilityId==='prankster'?1:0)}) {
  requireRule(CONNECTED_STATUS_MOVES.has(id),'Status move not connected: '+id);
  const effectiveWeather=actor.abilityId==='megasol'?'sunnyday':field.weatherSuppressed?'':field.weather;
  const move=effectiveMove(id,definition,actor,target,{weather:effectiveWeather,terrain:field.terrain});
  const ignoreAbility=actor.abilityId==='moldbreaker';
  if(id==='curse' && !actor.types.includes('Ghost'))move.target='self';
  const preconditionFailure=checkStatusMovePreconditions({id,actor,target,teams,seat,reviveIndex});
  if(preconditionFailure)return preconditionFailure;
  const {self,recipient}=getStatusMoveTargeting(move,actor,target);
  if (!actor.hp || !self && !recipient.hp && !['all','allyTeam','foeSide'].includes(move.target))return {success:false};
  const targetsFoe=!self && ['normal','adjacentFoe','foe','any','allAdjacentFoes','allAdjacent'].includes(move.target);
  const reflects=!self && (targetsFoe || move.target==='foeSide');
  prepareProtean(actor,move);
  const targetFailure=checkStatusMoveTarget({id,definition,move,actor,target,field,seat,teams,active,chart,definitions,rng,turn,ignoreAbility,self,targetsFoe,reflects,priority,remainingActions,targetWillMove,overrideAction,bounced});
  if(targetFailure)return targetFailure;
  const statusOptions={chart,ignoreAbility,sunny:effectiveWeather==='sunnyday'||actor.abilityId==='megasol',electricTerrain:field.terrain==='electricterrain',mistyTerrain:field.terrain==='mistyterrain',grounded:grounded(recipient,{ignoreAbility:ignoreAbility && recipient!==actor}),source:actor};
  const effectResult=applyStatusMoveEffect({id,move,actor,target,recipient,self,field,seat,teams,active,chart,definitions,rng,turn,transform,reviveIndex,remainingActions,targetWillMove,overrideAction,ignoreAbility,effectiveWeather,statusOptions});
  return finishStatusMove({id,actor,target,recipient,move,...effectResult,bounced,turn,actorAlreadyTrapped});
}
export function endRoost(pokemon) {
  if (!pokemon.volatiles?.roost)return;
  pokemon.types=[...pokemon.volatiles.roost.types];delete pokemon.volatiles.roost;
}
