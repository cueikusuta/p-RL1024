import { requireRule } from './draft.js';
import { boostStage } from './battle-moves.js';
import { modify, applyMoveDamage, captureItemMessages } from './battle-mechanics.js';

export function moveAccuracy(move, actor, target, {weather='',ignoreAbility=false, subsequentHit=false, effectiveSpeeds=null} = {}) {
  let accuracy = move.accuracy;
  if(target.volatiles?.minimize && move.flags?.minimize)return true;
  if(target.volatiles?.glaiverush)return true;
  const effectiveWeather=actor.abilityId==='megasol'?'sunnyday':weather;
  if(move.id==='toxic'&&actor.types?.includes('Poison'))accuracy=true;
  if(move.id==='hurricane'||move.id==='thunder')accuracy=effectiveWeather==='raindance'?true:effectiveWeather==='sunnyday'?50:accuracy;
  if(move.id==='blizzard'&&['snowscape','hail'].includes(effectiveWeather))accuracy=true;
  if (accuracy === true || move.alwaysHit || actor.abilityId === 'noguard' || target.abilityId === 'noguard') return true;
  // OHKO accuracy bypasses ModifyAccuracy and both accuracy/evasion stages.
  if (move.ohko)return accuracy;
  requireRule(Number.isFinite(accuracy) && accuracy >= 0, 'Missing move accuracy');
  const atk = move.ignoreAccuracy || !ignoreAbility&&target.abilityId==='unaware' ? 0 : boostStage(actor,'accuracy');
  const def = move.ignoreEvasion || actor.abilityId==='keeneye' || actor.abilityId==='illuminate' || actor.abilityId==='unaware' ? 0 : boostStage(target,'evasion');
  // Pinned Showdown's ModifyAccuracy event chains all modifiers, then rounds
  // once. Applying modify() separately changes the threshold by one point for
  // combinations such as Compound Eyes + Bright Powder.
  // The pinned runEvent orders same-priority callbacks by the holder's
  // effective Speed. For example, both Wide Lens and Bright Powder have
  // ModifyAccuracy priority -2; their 12-bit chain is order-sensitive.
  const accuracyMods=[];
  if(!ignoreAbility&&target.abilityId==='tangledfeet'&&target.volatiles?.confusion)accuracyMods.push({factor:2048,priority:-1,holder:'target'});
  if(actor.abilityId==='compoundeyes')accuracyMods.push({factor:5325,priority:-1,holder:'actor'});
  if(actor.abilityId==='hustle'&&move.category==='Physical')accuracyMods.push({factor:3277,priority:-1,holder:'actor'});
  if(actor.itemId==='widelens')accuracyMods.push({factor:4505,priority:-2,holder:'actor'});
  if(target.itemId==='brightpowder')accuracyMods.push({factor:3686,priority:-2,holder:'target'});
  const actorSpeed=effectiveSpeeds?.actor ?? actor.stats?.spe;
  const targetSpeed=effectiveSpeeds?.target ?? target.stats?.spe;
  accuracyMods.sort((a,b)=>b.priority-a.priority || (
    Number.isFinite(actorSpeed)&&Number.isFinite(targetSpeed)
      ? (b.holder==='actor'?actorSpeed:targetSpeed)-(a.holder==='actor'?actorSpeed:targetSpeed)
      : 0
  ));
  // Multiaccuracy hits apply stages before ModifyAccuracy; the first accuracy
  // check does so afterward (sim/battle-actions.ts).
  if(subsequentHit) {
    const stage=n=>n>=0 ? (3+n)/3 : 3/(3-n);
    accuracy *= stage(atk)/stage(def);
  }
  // The pinned runEvent only finalModify's nonnegative integer relay values.
  // Multiaccuracy boost stages can yield fractions; those keep full precision
  // and bypass the chained modifier, even when callbacks have registered one.
  if(accuracyMods.length && (!subsequentHit || Number.isInteger(accuracy))) {
    const chained=accuracyMods.reduce((value,{factor})=>Math.floor((value*factor+2048)/4096),4096);
    accuracy=subsequentHit
      ? Math.floor((Math.floor(accuracy*chained)+2047)/4096)
      : modify(accuracy,chained);
  }
  if(!subsequentHit) {
    const stage=Math.max(-6,Math.min(6,atk-def));
    accuracy=Math.floor(stage>=0 ? accuracy*(3+stage)/3 : accuracy*3/(3-stage));
  }
  return accuracy;
}
export function hitsAccuracy(move,actor,target,rng,options={}) {
  const accuracy=moveAccuracy(move,actor,target,options);
  if (accuracy === true) return true;
  // Pinned Showdown hitStepMoveHitLoop calls randomChance(accuracy, 100),
  // whose PRNG draws integer(100) even when accuracy is fractional.
  return rng.integer(100)<accuracy;
}
export function hitCount(move,rng) {
  if (!move.multihit) return 1;
  if (Number.isInteger(move.multihit)) return move.multihit;
  const [min,max]=move.multihit;
  requireRule(Number.isInteger(min)&&Number.isInteger(max)&&min>0&&max>=min,'Invalid multihit');
  if (min === 2 && max === 5) return [2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,5,5,5][rng.integer(20)];
  return min + rng.integer(max-min+1);
}
export function movePrecondition(move,actor,target,{targetAction=null,targetAlreadyActed=false,allowFaintedSelfdestruct=false,allowFaintedActor=false,turn=0,terrain='',priority=move.priority??0,ignoreAbility=false}={}) {
  if(move.id==='aurawheel'&&actor.baseSpeciesId!=='morpeko')return false;
  if (['fakeout','firstimpression'].includes(move.id) && (actor.activeMoveActions ?? 0)>1) return false;
  // Sucker Punch's target-action requirement is only one failure condition.
  // Successful target-action checks must still pass priority blockers (e.g.
  // Psychic Terrain / Queenly Majesty) and the shared fainted-target guard.
  if (move.id === 'suckerpunch' && (targetAlreadyActed || target.volatiles?.recharge ||
    targetAction?.kind !== 'move' || !targetAction.move || targetAction.move.category === 'Status')) return false;
  // Passing a move-specific onTry check must still pass the shared faint guards.
  if (move.id === 'poltergeist' && !target.itemId) return false;
  if (move.id === 'doubleshock' && !actor.types.includes('Electric')) return false;
  if (move.id === 'endeavor' && actor.hp>=target.hp) return false;
  if (move.id === 'counter' || move.id === 'mirrorcoat') {
    const hit=actor.lastDamagedBy;
    if (!(hit?.turn===turn && hit.damage>=0 && hit.category===(move.id==='counter'?'Physical':'Special'))) return false;
  }
  if (move.ohko && (actor.level??50)<(target.level??50)) return false;
  if (move.id==='sheercold' && target.types.includes('Ice')) return false;
  if(move.id==='noretreat' && actor.volatiles?.noretreat)return false;
  if(move.id==='octolock' && target.types.includes('Ghost'))return false;
  if(!ignoreAbility&&['queenlymajesty','armortail'].includes(target.abilityId)&&priority>0.1&&move.target!=='foeSide'&&
    !(move.target==='all'&&['perishsong','flowershield','rototiller'].includes(move.id))&&(target!==actor||move.target==='all'))return false;
  // Terrain blocks priority against grounded targets, including Flying or
  // Levitate holders of Iron Ball. Keep this consistent with grounded() in
  // battle-effects; unbreakable airborne abilities remain airborne.
  const targetGrounded=target.itemId==='ironball' || (target.itemId!=='airballoon' &&
    !target.types?.includes('Flying') && (ignoreAbility || !['levitate','eelevate'].includes(target.abilityId)));
  if(terrain==='psychicterrain' && priority>0 && target!==actor && targetGrounded)return false;
  return (actor.hp>0 || allowFaintedActor || allowFaintedSelfdestruct && move.selfdestruct==='always') && target.hp>0;
}
export function protectMove(pokemon,id,turn,rng,{remainingActions=true}={}) {
  requireRule(['protect','endure','kingsshield','silktrap','spikyshield'].includes(id),'Invalid protecting move');
  pokemon.volatiles ??= {};
  const stall=pokemon.volatiles.stall;
  if (!remainingActions) return false;
  if (stall && stall.expires>=turn && rng.integer(stall.counter)!==0) {
    delete pokemon.volatiles.stall; return false;
  }
  pokemon.volatiles[id]={turn};
  pokemon.volatiles.stall={counter:stall && stall.expires>=turn ? Math.min(729,stall.counter*3) : 3,expires:turn+1};
  return true;
}
export function isProtected(move,actor,target,turn) {
  return actor!==target && Boolean(move.flags?.protect && ['protect','kingsshield','silktrap','spikyshield'].some(id=>
    (id!=='kingsshield'||move.category!=='Status') && target.volatiles?.[id]?.turn===turn));
}
export function bypassesSubstitute(move,actor,target) {
  return actor===target || Boolean(move.flags?.bypasssub || move.flags?.sound || move.infiltrates || actor.abilityId==='infiltrator');
}
export function blocksStatusMove(move,actor,target,turn) {
  return isProtected(move,actor,target,turn) || Boolean(target.volatiles?.substitute && !bypassesSubstitute(move,actor,target));
}

// Each hit is a complete event boundary: item/ability updates can modify HP or
// interrupt a move before its next hit. The owning engine supplies those hooks.
export function runDamageHits({move,actor,target,turn,rng,damageForHit,onHit=()=>{},onSubstitute=()=>{},onDisguiseDamage=()=>{},onTryHit=()=>{},onBeforeDamage=()=>{},accuracyOptions={},allowFaintedSelfdestruct=false,allowFaintedActor=false}) {
  requireRule(typeof damageForHit==='function','Damage calculator required');
  requireRule(!allowFaintedSelfdestruct || move.selfdestruct==='always','Fainted actor only allowed for always-selfdestruct moves');
  requireRule(!allowFaintedActor || Boolean(move.flags?.futuremove),'Fainted actor only allowed for delayed future moves');
  const protectedByProtect=move.breaksProtect ? false : isProtected(move,actor,target,turn);
  const piercesProtect=protectedByProtect&&actor.abilityId==='piercingdrill'&&Boolean(move.flags?.contact);
  const bypassesProtect=protectedByProtect&&actor.abilityId==='unseenfist'&&Boolean(move.flags?.contact);
  if (protectedByProtect&&!piercesProtect&&!bypassesProtect) return {hits:0,totalDamage:0,blocked:'protect'};
  if (!hitsAccuracy(move,actor,target,rng,accuracyOptions)) return {hits:0,totalDamage:0,missed:true};
  onTryHit();
  const count=hitCount(move,rng);
  let hits=0,totalDamage=0,substituteHits=0,substituteDamage=0;
  for (let hit=1;hit<=count;hit++) {
    if ((!actor.hp && !(allowFaintedSelfdestruct && hit===1) && !allowFaintedActor) || !target.hp || hit>1 && actor.status==='slp' && !move.sleepUsable) break;
    if (hit>1 && move.multiaccuracy && !hitsAccuracy(move,actor,target,rng,{...accuracyOptions,subsequentHit:true})) break;
    const sub=target.volatiles?.substitute;
    const hitsSubstitute=Boolean(sub && !bypassesSubstitute(move,actor,target));
    const disguiseBlocksHit=!hitsSubstitute && target.abilityId==='disguise' && !target.disguiseBusted && !accuracyOptions.ignoreAbility;
    // Disguise's onDamage runs after getDamage for every hit, so the damage
    // calculation (including its RNG draws) happens before the body hit is
    // replaced with zero. Update changes the form before a following hit.
    let damage=damageForHit(hit);
    if(disguiseBlocksHit)damage=0;
    if(piercesProtect)damage=modify(damage,1024);
    requireRule(Number.isInteger(damage)&&damage>=0,'Invalid hit damage');
      if(!hitsSubstitute&&!disguiseBlocksHit&&damage>0)onBeforeDamage({move,actor,target,damage,hit});
    let lost;
    if (hitsSubstitute) {
      lost=Math.min(sub.hp,damage); sub.hp-=lost;
      if (!sub.hp) delete target.volatiles.substitute;
      substituteHits++;
      substituteDamage+=lost;
    } else {
      const beforeHP=target.hp;
      const disguiseBreak=disguiseBlocksHit;
      let disguiseDamage=0,applyDisguiseDamage=null,flushItemMessages=()=>{};
      if(disguiseBreak) {
        target.disguiseBusted=true;
        disguiseDamage=Math.min(target.hp,Math.max(1,Math.floor(target.maxHP/8)));
        let applied=false;
        applyDisguiseDamage=()=>{
          if(applied)return;
          applied=true;
          target.hp-=disguiseDamage;
          onDisguiseDamage({target,damage:disguiseDamage,source:actor,direct:false,beforeHP,kind:'disguise'});
        };
        lost=0;
      } else {
        if (target.volatiles?.endure?.turn===turn && damage>=target.hp) damage=Math.max(0,target.hp-1);
        const captured=captureItemMessages(target,()=>applyMoveDamage(target,damage,{ignoreAbility:accuracyOptions.ignoreAbility}));
        lost=captured.value;flushItemMessages=captured.flush;
      }
      // Numeric zero (for example Endure at 1 HP) is still a body hit in the
      // reference history. Substitute hits take the other branch entirely.
      target.timesAttacked=(target.timesAttacked??0)+1;
      target.lastDamagedBy={turn,sourceId:actor.instanceId,damage:lost,...(move.category?{category:move.category}:{})};
      hits++;totalDamage+=lost;
      const event={hit,damage:lost,beforeHP,substitute:false,brokenSubstitute:false,flushItemMessages,...(disguiseBreak?{disguise:true,applyDisguiseDamage}:{})};
      const continueMove=onHit(event);
      flushItemMessages();
      if(disguiseBreak)applyDisguiseDamage();
      if(continueMove===false)break;
      continue;
    }
    hits++;totalDamage+=lost;
    const event={hit,damage:lost,substitute:hitsSubstitute,brokenSubstitute:hitsSubstitute&&!target.volatiles.substitute};
    const continueMove=hitsSubstitute ? onSubstitute(event) : onHit(event);
    if (continueMove===false) break;
  }
  return {hits,totalDamage,substituteHits,substituteDamage};
}
export function endTurnProtection(pokemon,turn) {
  for(const id of ['protect','endure','kingsshield','silktrap','spikyshield'])if(pokemon.volatiles?.[id]?.turn<=turn)delete pokemon.volatiles[id];
  if(pokemon.volatiles?.stall?.expires<=turn)delete pokemon.volatiles.stall;
}
