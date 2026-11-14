import { requireRule } from './draft.js';
import { changeBoosts } from './battle-moves.js';
import { setStatus, addConfusion, addFlinch, thawFromHit } from './battle-status.js';
import { mentalHerb, applyHealBlock, applyThroatChop, applyDisable } from './battle-actions.js';
import { snapshotConditions, notifyConditionChanges } from './battle-messages.js';

function volatileEffect(id,target,rng,options) {
  const before=snapshotConditions(target);
  const result=applyVolatileEffect(id,target,rng,options);
  notifyConditionChanges(target,before,{causeMoveId:['partiallytrapped','saltcure','glaiverush','lockedmove'].includes(id)?options.moveId:null});return result;
}
function applyVolatileEffect(id,target,rng,options) {
  if (id==='confusion')return addConfusion(target,rng,options);
  if (id==='flinch') {
    const applied=!options.targetAlreadyActed && addFlinch(target,{ignoreAbility:options.ignoreAbility});
    return applied;
  }
  if (id==='glaiverush') { target.volatiles??={};target.volatiles.glaiverush=true;return true; }
  if (id==='mustrecharge') { target.volatiles??={};target.volatiles.recharge=true;return true; }
  if (id==='saltcure') { target.volatiles??={};target.volatiles.saltcure=true;return true; }
  if (id==='sparklingaria') { target.volatiles??={};target.volatiles.sparklingaria=true;return true; }
  if (id==='lockedmove') {
    target.volatiles??={};
    if(target.volatiles.lockedmove)return false;
    target.volatiles.lockedmove={move:options.moveId,duration:2+rng.integer(2)};return true;
  }
  if (id==='partiallytrapped') {
    if(target.volatiles?.partiallytrapped)return false;
    target.volatiles??={};target.volatiles.partiallytrapped={duration:5+rng.integer(2),sourceId:options.source?.instanceId};return true;
  }
  if (id==='healblock')return applyHealBlock(target);
  if (id==='throatchop')return applyThroatChop(target,{causeMoveId:options.moveId});
  requireRule(false,'Unsupported secondary volatile: '+id);
}
function hitEffect(effect,source,target,rng,options) {
  const results=[];
  if (target.hp) {
    if (effect.boosts)results.push({kind:'boost',target,changes:changeBoosts(target,effect.boosts,{source,ignoreAbility:options.ignoreAbility,effect:'secondary',causeMoveId:options.moveId})});
    if (effect.status)results.push({kind:'status',target,status:effect.status,applied:setStatus(target,effect.status,rng,{...options,source})});
    if (effect.volatileStatus)results.push({kind:'volatile',target,id:effect.volatileStatus,applied:volatileEffect(effect.volatileStatus,target,rng,options)});
  }
  if (effect.self && source.hp)results.push(...hitEffect(effect.self,source,source,rng,{...options,ignoreAbility:false}));
  return results;
}
export function applySecondaryEffects(move,actor,target,rng,options={}) {
  if ((move.hasSheerForce && !move.kingsRockAfterSheerForce) || !actor.hp)return [];
  const result=[];
  for(const effect of move.secondaries ?? (move.secondary ? [move.secondary] : [])) {
    const draw=rng.integer(100);
    if (effect.chance!==undefined && draw>=effect.chance)continue;
    result.push(...hitEffect(effect,actor,target,rng,{...options,moveId:move.id}));
    // Fixed-source callbacks that are intentionally not serialized in JSON.
    if (!move.hasSheerForce && move.id==='direclaw' && target.hp) {
      const status=['psn','par','slp'][rng.integer(3)];
      result.push({kind:'status',target,status,applied:setStatus(target,status,rng,{...options,source:actor})});
    }
    if (!move.hasSheerForce && move.id==='alluringvoice' && target.hp && target.statsRaisedThisTurn)result.push({kind:'volatile',target,id:'confusion',applied:addConfusion(target,rng,options)});
    if (!move.hasSheerForce && move.id==='throatchop' && target.hp)result.push({kind:'volatile',target,id:'throatchop',applied:applyThroatChop(target,{causeMoveId:move.id})});
  }
  // Spirit Shackle's fixed reference stores trapping as a nested secondary
  // callback, which is intentionally not serialized into runtime battle data.
  // Keep it distinct from partial trapping: it prevents switching but deals
  // no residual damage and has no duration.
  if(!move.hasSheerForce && move.id==='spiritshackle' && actor.hp && target.hp && !target.types.includes('Ghost') && !target.volatiles?.trapped) {
    const before=snapshotConditions(target);
    target.volatiles??={};target.volatiles.trapped={sourceId:actor.instanceId};
    notifyConditionChanges(target,before,{causeMoveId:move.id});
    result.push({kind:'volatile',target,id:'trapped',applied:true});
  }
  return result;
}
// Reference spreadMoveHit converts a Substitute target to null, then still
// evaluates secondaries. This preserves RNG and secondary.self while suppressing
// every effect aimed at the protected target.
export function applySubstituteSecondarySelfEffects(move,actor,rng,options={}) {
  if((move.hasSheerForce&&!move.kingsRockAfterSheerForce)||!actor.hp)return [];
  const result=[];
  for(const effect of move.secondaries ?? (move.secondary ? [move.secondary] : [])) {
    const draw=rng.integer(100);
    if(effect.chance!==undefined&&draw>=effect.chance)continue;
    if(effect.self&&actor.hp)result.push(...hitEffect(effect.self,actor,actor,rng,{...options,ignoreAbility:false,moveId:move.id}));
  }
  return result;
}
export function applyPrimarySelfEffects(move,actor,target,rng,options={}) {
  if (!move.self || !actor.hp)return [];
  // The reference consumes this draw even for unconditional self boosts.
  if (move.self.boosts) {
    const draw=rng.integer(100);
    if (move.self.chance!==undefined && draw>=move.self.chance)return [];
  }
  return hitEffect(move.self,actor,actor,rng,{...options,moveId:move.id});
}
export function applyPrimaryMoveEffects(move,actor,target,rng,options={}) {
  if(!move.volatileStatus||!target.hp)return [];
  return [{kind:'volatile',target,id:move.volatileStatus,applied:volatileEffect(move.volatileStatus,target,rng,{...options,source:actor,moveId:move.id})}];
}
export function applyPostMoveSelfBoost(move,actor) {
  return actor.hp && move.selfBoost?.boosts ? changeBoosts(actor,move.selfBoost.boosts,{effect:move.id}) : {};
}
export function makesContact(move,actor) {
  return Boolean(move.flags?.contact) && actor.itemId!=='protectivepads';
}
export function contactStatusAbility(move,actor,target,rng,{chart,weather='',terrain='',grounded=()=>true,substitute=false,phase='both',actorAbility=actor.abilityId,targetAbility=target.abilityId}={}) {
  if (substitute)return [];
  // These reactive contact abilities are not breakable; Mold Breaker does not suppress them.
  const options={chart,sunny:weather==='sunnyday',electricTerrain:terrain==='electricterrain',mistyTerrain:terrain==='mistyterrain',grounded:grounded(actor),source:target};
  const results=[];
  if (phase!=='source' && makesContact(move,actor)) {
    const status={static:'par',flamebody:'brn',poisonpoint:'psn'}[targetAbility];
    if (status && rng.integer(10)<3 && actor.hp)results.push({status,target:actor,source:target,causeAbilityId:targetAbility,applied:setStatus(actor,status,rng,options)});
    if(targetAbility==='cutecharm' && actor.abilityId!=='oblivious' && actor.gender && target.gender && actor.gender!=='N' && target.gender!=='N' && actor.gender!==target.gender && !actor.volatiles?.attract && rng.integer(10)<3 && actor.hp){
      // TryAddVolatile checks Aroma Veil after the originating ability's RNG.
      if(actor.abilityId!=='aromaveil'){actor.volatiles??={};actor.volatiles.attract={sourceId:target.instanceId};mentalHerb(actor);}
    }
    if (targetAbility==='effectspore' && !actor.types.includes('Grass')) {
      const draw=rng.integer(100),status=draw<11 ? 'slp' : draw<21 ? 'par' : draw<30 ? 'psn' : '';
      if (status && actor.hp)results.push({status,target:actor,source:target,causeAbilityId:'effectspore',applied:setStatus(actor,status,rng,options)});
    }
  }
  // SourceDamagingHit deliberately checks the receiver's contact protection,
  // rather than the attacking holder's Pads (pinned Poison Touch callback).
  if(phase!=='target' && actorAbility==='poisontouch' && target.abilityId!=='shielddust' && target.itemId!=='covertcloak' &&
    makesContact(move,target) && rng.integer(10)<3 && target.hp)
    results.push({status:'psn',target,source:actor,causeAbilityId:'poisontouch',applied:setStatus(target,'psn',rng,{...options,source:actor,grounded:grounded(target)})});
  if (phase!=='source' && targetAbility==='cursedbody' && !actor.volatiles?.disable && !move.isMax && !move.flags?.futuremove && move.id!=='struggle' && rng.integer(10)<3)
    results.push({id:'disable',applied:actor.abilityId!=='aromaveil' && applyDisable(actor,{willMove:false,activeMove:true,causeAbilityId:'cursedbody',causeSource:target})});
  return results;
}
export function prepareProtean(actor,move) {
  if (!['protean','libero'].includes(actor.abilityId) || actor.proteanUsed || move.flags?.futuremove || move.callsMove || move.type==='???' || !move.type || actor.types.join()===move.type)return false;
  actor.types=[move.type];actor.proteanUsed=true;return true;
}
export function startTurnFlags(pokemon) {
  pokemon.statsRaisedThisTurn=false;pokemon.statsLoweredThisTurn=false;
}
export function moxieAfterFaint(actor) {
  if(!actor.hp)return {};
  if(actor.abilityId==='moxie')return changeBoosts(actor,{atk:1},{effect:'moxie'});
  if(actor.abilityId==='asonespectrier')return changeBoosts(actor,{spa:1},{effect:'asonespectrier'});
  if(actor.abilityId==='asoneglastrier')return changeBoosts(actor,{atk:1},{effect:'chillingneigh'});
  return {};
}
export { thawFromHit };
