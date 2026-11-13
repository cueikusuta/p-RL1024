import { modify, effectiveness, elementalDamage, consumeHealingBerry, consumeHeldItem, afterBerryEaten } from './battle-mechanics.js';
import { damageStats, changeBoosts, healHP, effectiveMove, whiteHerb, fixedMoveDamage, resetBoosts } from './battle-moves.js';
import { bypassesSubstitute, isProtected, runDamageHits, movePrecondition } from './battle-hit.js';
import { requireRule } from './draft.js';
import { isFormLockedHeldItem } from './battle-mechanics.js';
import { grounded } from './battle-grounding.js';
import { notifyEffectEvent } from './battle-messages.js';
import { applySecondaryEffects, applySubstituteSecondarySelfEffects, applyPrimarySelfEffects, applyPrimaryMoveEffects, applyPostMoveSelfBoost, contactStatusAbility, makesContact, prepareProtean, moxieAfterFaint, thawFromHit } from './battle-secondary.js';

const TYPE_ITEMS={dracoplate:'Dragon',dreadplate:'Dark',skyplate:'Flying',meadowplate:'Grass',mindplate:'Psychic',ironplate:'Steel',pixieplate:'Fairy',zapplate:'Electric',insectplate:'Bug',earthplate:'Ground',fistplate:'Fighting',splashplate:'Water',flameplate:'Fire',spookyplate:'Ghost',stoneplate:'Rock',toxicplate:'Poison',icicleplate:'Ice',sharpbeak:'Flying',twistedspoon:'Psychic',softsand:'Ground',nevermeltice:'Ice',metalcoat:'Steel',blackbelt:'Fighting',blackglasses:'Dark',charcoal:'Fire',fairyfeather:'Fairy',hardstone:'Rock',magnet:'Electric',miracleseed:'Grass',mysticwater:'Water',poisonbarb:'Poison',silkscarf:'Normal',spelltag:'Ghost'};
const RESIST_BERRIES={passhoberry:'Water',roseliberry:'Fairy',wacanberry:'Electric',chartiberry:'Rock',chopleberry:'Fighting',cobaberry:'Flying',kasibberry:'Ghost',shucaberry:'Ground',yacheberry:'Ice',habanberry:'Dragon',occaberry:'Fire'};
export function chainModifiers(modifiers) {
  return modifiers.reduce((value,next)=>Math.floor((value*next+2048)/4096),4096);
}
export function preparedAbilityMove(move,actor) {
  const result=structuredClone(move);
  const converted={pixilate:'Fairy',aerilate:'Flying',refrigerate:'Ice',dragonize:'Dragon'}[actor.abilityId];
  if(converted && result.type==='Normal' && !['judgment','multiattack','naturalgift','revelationdance','technoblast','terrainpulse','weatherball'].includes(result.id)){result.type=converted;result.typeChangerBoosted=true;}
  if(actor.abilityId==='parentalbond' && result.category!=='Status' && !result.multihit && !result.flags?.charge && !result.flags?.noparentalbond && !result.flags?.futuremove){result.multihit=2;result.parentalBond=true;}
  if(actor.abilityId==='skilllink') {
    if(Array.isArray(result.multihit))result.multihit=result.multihit[1];
    if(result.multiaccuracy)delete result.multiaccuracy;
  }
  if (actor.abilityId==='infiltrator')result.infiltrates=true;
  if (actor.abilityId==='sheerforce' && (result.secondary || result.secondaries?.length)) {
    delete result.secondary;delete result.secondaries;delete result.self;result.hasSheerForce=true;
  }
  // Fixed Showdown: King's Rock onModifyMovePriority:-1 runs after Sheer Force.
  // The original move secondary is removed, but the Rock's 10% flinch survives.
  if(actor.itemId==='kingsrock' && result.category!=='Status') {
    const effects=result.secondaries ?? (result.secondary ? [result.secondary] : []);
    if(!effects.some(effect=>effect.volatileStatus==='flinch')) {
      result.secondaries=[...effects,{chance:10,volatileStatus:'flinch'}];
      if(result.hasSheerForce)result.kingsRockAfterSheerForce=true;
    }
  }
  return result;
}
export { grounded } from './battle-grounding.js';
export function semiInvulnerable(pokemon) {
  return ['fly','bounce','dive','dig','phantomforce','shadowforce','skydrop'].some(id=>pokemon.volatiles?.[id]);
}
export function moveEffectiveness(move,actor,target,chart) {
  let factor=1;
  for(const type of target.types) {
    if (type==='Ghost' && ['Normal','Fighting'].includes(move.type) && actor.abilityId==='scrappy')continue;
    if (type==='Flying' && move.type==='Ground' && target.itemId==='ironball')continue;
    factor*=effectiveness(move.type,[type],chart);
  }
  if (move.id==='freezedry' && target.types.includes('Water'))factor*=4;
  return factor;
}
function healOnAbilityImmunity(target,amount) {
  const healed=healHP(target,amount);
  // The confirmed recovery is caused by the defender's revealed ability, not the blocked move.
  if(healed>0)notifyEffectEvent(target,{kind:'heal',ability:target.abilityId});
}
export function tryAbilityImmunity(move,actor,target,{ignoreAbility=false,onAbsorbed=()=>{}}={}) {
  if (actor===target || ignoreAbility) return false;
  if(move.ohko && target.abilityId==='sturdy')return true;
  if(move.type==='Water' && target.abilityId==='dryskin') {healOnAbilityImmunity(target,Math.max(1,Math.floor(target.maxHP/4)));return true;}
  if(move.flags?.sound && target.abilityId==='soundproof')return true;
  if(move.flags?.bullet && target.abilityId==='bulletproof')return true;
  if(move.type==='Electric' && target.abilityId==='motordrive') {changeBoosts(target,{spe:1},{effect:'motordrive'});return true;}
  if(move.type==='Ground' && target.abilityId==='eartheater') {healOnAbilityImmunity(target,Math.max(1,Math.floor(target.maxHP/4)));return true;}
  if(move.type==='Ground' && target.abilityId==='eelevate')return true;
  if(target.abilityId==='oblivious' && ['attract','captivate','taunt'].includes(move.id))return true;
  if(move.type==='Grass' && target.abilityId==='sapsipper'){changeBoosts(target,{atk:1},{effect:'sapsipper'});return true;}
  if(move.category==='Status' && target.abilityId==='goodasgold')return true;
  if (move.type==='Ground' && target.abilityId==='levitate' && target.itemId!=='ironball')return true;
  if (move.type==='Fire' && target.abilityId==='flashfire') {
    target.volatiles ??= {};target.volatiles.flashfire=true;
    // The actual converted Fire move has triggered the immunity, not merely
    // a previous Flash Fire charge on this target.
    onAbsorbed('flashfire');return true;
  }
  if (move.type==='Water' && target.abilityId==='waterabsorb' || move.type==='Electric' && target.abilityId==='voltabsorb') {
    healOnAbilityImmunity(target,Math.max(1,Math.floor(target.maxHP/4)));return true;
  }
  if (move.type==='Electric' && target.abilityId==='lightningrod') {
    changeBoosts(target,{spa:1},{effect:'lightningrod'});return true;
  }
  return false;
}

export function battleDamage({move,actor,target,chart,crit=false,roll=0,weather='',terrain='',field=null,targetSeat=null,targetAlreadyActed=false,ignoreAbility=['moldbreaker','teravolt','turboblaze'].includes(actor.abilityId)||move.id==='sunsteelstrike'}) {
  const factor=moveEffectiveness(move,actor,target,chart);
  if (!factor || move.type==='Ground' && target.itemId==='airballoon')return 0;
  const stats=damageStats(move,actor,target,crit,ignoreAbility);
  const attackMods=[];
  if(actor.abilityId==='guts' && actor.status && move.category==='Physical')attackMods.push(6144);
  if(actor.abilityId==='firemane' && move.type==='Fire')attackMods.push(6144);
  if(actor.abilityId==='solarpower' && weather==='sunnyday' && move.category==='Special')attackMods.push(6144);
  if(!ignoreAbility && target.abilityId==='marvelscale' && target.status && move.category==='Physical')stats.defense=modify(stats.defense,6144);
  if(!ignoreAbility && target.abilityId==='grasspelt' && terrain==='grassyterrain' && move.category==='Physical')stats.defense=modify(stats.defense,6144);
  // getDamage selects the raw stat first, then ModifyAtk/SpA by move category.
  // Thus Huge/Pure Power also multiply Body Press's Defense-derived attack.
  if (move.category==='Physical' && ['hugepower','purepower'].includes(actor.abilityId))attackMods.push(8192);
  // Hustle returns a directly modified stat; other attack callbacks chain.
  if (move.category==='Physical' && actor.abilityId==='hustle')stats.attack=modify(stats.attack,6144);
  // Ruin abilities modify all other active Pokemon's stats, not the holder's.
  // In singles the opposite active holder is the only eligible source.
  if (move.category==='Special' && target.abilityId==='vesselofruin' && actor.abilityId!=='vesselofruin' && actor!==target)stats.attack=modify(stats.attack,3072);
  if (move.category==='Physical' && actor.abilityId==='swordofruin' && target.abilityId!=='swordofruin' && actor!==target)stats.defense=modify(stats.defense,3072);
  if(actor.abilityId==='orichalcumpulse'&&weather==='sunnyday'&&move.category==='Physical')attackMods.push(5461);
  if(actor.abilityId==='hadronengine'&&terrain==='electricterrain'&&move.category==='Special')attackMods.push(5461);
  const starterType={blaze:'Fire',torrent:'Water',overgrow:'Grass',swarm:'Bug'}[actor.abilityId];
  if (starterType===move.type && actor.hp<=actor.maxHP/3)attackMods.push(6144);
  if (actor.abilityId==='flashfire' && actor.volatiles?.flashfire && move.type==='Fire')attackMods.push(6144);
  if (actor.abilityId==='waterbubble' && move.type==='Water')attackMods.push(8192);
  if (!ignoreAbility && (
    target.abilityId==='thickfat' && ['Fire','Ice'].includes(move.type) ||
    ['heatproof','waterbubble'].includes(target.abilityId) && move.type==='Fire' ||
    target.abilityId==='purifyingsalt' && move.type==='Ghost'
  ))attackMods.push(2048);
  // ModifyAtk/SpA applies the combined event modifier once, before base damage.
  stats.attack=modify(stats.attack,chainModifiers(attackMods));
  const powerMods=[];
  // Recorded BasePower event priorities: Technician30, abilities21,
  // category items16, type items15, gem14, terrain6.
  if (actor.abilityId==='technician' && move.basePower<=60)powerMods.push(6144);
  if(['collisioncourse','electrodrift'].includes(move.id)&&factor>1)powerMods.push(5461);
  if(!ignoreAbility && target.abilityId==='dryskin' && move.type==='Fire')powerMods.push(5120);
  if(actor.abilityId==='strongjaw' && move.flags?.bite)powerMods.push(6144);
  if(actor.abilityId==='reckless' && (move.recoil || move.hasCrashDamage))powerMods.push(4915);
  if(actor.abilityId==='punkrock' && move.flags?.sound)powerMods.push(5325);
  if((actor.abilityId==='fairyaura'||target.abilityId==='fairyaura') && move.type==='Fairy')powerMods.push(5448);
  if (actor.abilityId==='toughclaws' && move.flags?.contact)powerMods.push(5325);
  if (actor.abilityId==='megalauncher' && move.flags?.pulse)powerMods.push(6144);
  if(actor.abilityId==='ironfist' && move.flags?.punch)powerMods.push(4915);
  if(actor.abilityId==='sharpness' && move.flags?.slicing)powerMods.push(6144);
  if(move.typeChangerBoosted)powerMods.push(4915);
  if (move.hasSheerForce)powerMods.push(5325);
  if (actor.abilityId==='analytic' && targetAlreadyActed)powerMods.push(5325);
  if (actor.abilityId==='sandforce' && weather==='sandstorm' && ['Ground','Rock','Steel'].includes(move.type))powerMods.push(5325);
  if (actor.abilityId==='supremeoverlord' && actor.supremeOverlordFallen)powerMods.push([4096,4506,4915,5325,5734,6144][Math.min(5,actor.supremeOverlordFallen)]);
  if (actor.itemId==='muscleband' && move.category==='Physical' || actor.itemId==='wiseglasses' && move.category==='Special')powerMods.push(4505);
  if (TYPE_ITEMS[actor.itemId]===move.type)powerMods.push(4915);
  if (actor.volatiles?.gem && move.type==='Normal')powerMods.push(5325);
  // Charge's condition uses BasePower priority 9, between items and terrain.
  if (actor.volatiles?.charge && move.type==='Electric')powerMods.push(8192);
  if (terrain==='electricterrain' && move.type==='Electric' && grounded(actor))powerMods.push(5325);
  if (terrain==='grassyterrain' && move.type==='Grass' && grounded(actor))powerMods.push(5325);
  if (terrain==='psychicterrain' && move.type==='Psychic' && grounded(actor))powerMods.push(5325);
  if (terrain==='grassyterrain' && grounded(target,{ignoreAbility}) && !semiInvulnerable(target) && ['bulldoze','earthquake','magnitude'].includes(move.id))powerMods.push(2048);
  if (move.id==='knockoff' && target.itemId && target.itemId!==target.megaStoneId && !isFormLockedHeldItem(target) && (ignoreAbility||target.abilityId!=='stickyhold'))powerMods.push(6144);
  if (move.id==='solarbeam' && ['raindance','sandstorm','snowscape'].includes(weather))powerMods.push(2048);
  const power=Math.max(1,modify(move.basePower,chainModifiers(powerMods)));
  const finalMods=[];
  if(target.volatiles?.minimize && move.flags?.minimize)finalMods.push(8192);
  if (!ignoreAbility && target.abilityId==='multiscale' && target.hp===target.maxHP)finalMods.push(2048);
  if (factor>1 && (target.abilityId==='prismarmor' || !ignoreAbility && ['solidrock','filter'].includes(target.abilityId)))finalMods.push(3072);
  if(!ignoreAbility && target.abilityId==='fluffy' && move.flags?.contact)finalMods.push(2048);
  if(!ignoreAbility && target.abilityId==='punkrock' && move.flags?.sound)finalMods.push(2048);
  if(!ignoreAbility && target.abilityId==='fluffy' && move.type==='Fire')finalMods.push(8192);
  if(terrain==='mistyterrain' && move.type==='Dragon' && grounded(target,{ignoreAbility}))finalMods.push(2048);
  if(crit && actor.abilityId==='sniper')finalMods.push(6144);
  if(target.volatiles?.glaiverush)finalMods.push(8192);
  if(!ignoreAbility && target.abilityId==='auraguard' && move.flags?.contact)finalMods.push(2048);
  const screens=field?.sides?.[targetSeat];
  if(move.id!=='brickbreak' && !crit && actor.abilityId!=='infiltrator' && screens && (
    screens.auroraveil || move.category==='Physical' && screens.reflect || move.category==='Special' && screens.lightscreen
  ))finalMods.push(2048);
  if (actor.itemId==='expertbelt' && factor>1)finalMods.push(4915);
  if (actor.itemId==='lifeorb')finalMods.push(5324);
  if (factor>1 && !target.volatiles?.unnerved && RESIST_BERRIES[target.itemId]===move.type && (!target.volatiles?.substitute || bypassesSubstitute(move,actor,target))) {
    const berry=consumeHeldItem(target,{kind:'activated'});afterBerryEaten(target,berry);finalMods.push(2048);
  }
  if(actor.abilityId==='megasol')weather='sunnyday';
  const weatherModifier=(weather==='sunnyday' && move.type==='Fire' || weather==='raindance' && move.type==='Water') ? 6144 :
    (weather==='sunnyday' && move.type==='Water' || weather==='raindance' && move.type==='Fire') ? 2048 : 4096;
  if (weather==='sandstorm' && target.types.includes('Rock') && move.category==='Special')stats.defense=modify(stats.defense,6144);
  if (weather==='snowscape' && target.types.includes('Ice') && move.category==='Physical')stats.defense=modify(stats.defense,6144);
  return elementalDamage({...stats,power,crit,roll,baseModifier:move.parentalBondHit ? 1024 : 4096,stab:actor.types.includes(move.type) ? (actor.abilityId==='adaptability' ? 8192 : 6144) : false,effectiveness:factor,weatherModifier,
    burned:actor.status==='brn' && actor.abilityId!=='guts' && move.category==='Physical',finalModifier:chainModifiers(finalMods)});
}
export function indirectDamage(pokemon,amount,{deferBerry=false}={}) {
  if (pokemon.hp<=0 || pokemon.abilityId==='magicguard')return 0;
  const lost=Math.min(pokemon.hp,Math.max(1,Math.floor(amount)));pokemon.hp-=lost;
  if(!deferBerry)consumeHealingBerry(pokemon);
  return lost;
}
export function contactDamage(move,actor,target,{substitute=false,ignoreAbility=false,onSource=()=>{}}={}) {
  if (!makesContact(move,actor) || substitute || actor.hp<=0)return 0;
  let lost=0;
  // Fixed Showdown DamagingHit order: Rough Skin (1), then Rocky Helmet (2).
  // A Sitrus Berry must not heal between these two damage callbacks.
  if (target.abilityId==='roughskin'){
    const amount=indirectDamage(actor,actor.maxHP/8,{deferBerry:true});
    lost+=amount;if(amount)onSource({kind:'ability',id:'roughskin',amount});
  }
  // Showdown onDamagingHitOrder:1  Ea contact attack that KOs the holder
  // triggers Aftermath before the order-2 Rocky Helmet callback.
  // The regular hit path excludes substitute-only damage from this callback.
  if (target.abilityId==='aftermath' && !target.hp && actor.abilityId!=='damp') {
    const amount=indirectDamage(actor,actor.maxHP/4,{deferBerry:true});
    lost+=amount;if(amount)onSource({kind:'ability',id:'aftermath',amount});
  }
  if (target.itemId==='rockyhelmet'){
    const amount=indirectDamage(actor,actor.maxHP/6,{deferBerry:true});
    lost+=amount;if(amount)onSource({kind:'item',id:'rockyhelmet',amount});
  }
  if(lost)consumeHealingBerry(actor);
  return lost;
}
export function drainHealth(actor,target,amount,{ignoreAbility=false}={}) {
  const base=Math.floor(amount);
  const boosted=actor.itemId==='bigroot' ? modify(base,5324) : base;
  if (!ignoreAbility && target.abilityId==='liquidooze')return -indirectDamage(actor,boosted);
  return healHP(actor,boosted);
}
export function afterAttackItems(move,actor,totalDamage,{forcedSwitch=false,hitOccurred=totalDamage>0}={}) {
  if (forcedSwitch || move.hasSheerForce || move.category==='Status' || !hitOccurred || actor.hp<=0)return 0;
  if (actor.itemId==='lifeorb')return -indirectDamage(actor,actor.maxHP/10);
  if (actor.itemId==='shellbell' && totalDamage>0)return healHP(actor,Math.max(1,Math.floor(totalDamage/8)));
  return 0;
}
export function attackRecoil(move,actor,totalDamage) {
  if (!move.recoil || totalDamage<=0 || actor.abilityId==='rockhead')return 0;
  return indirectDamage(actor,Math.round(totalDamage*move.recoil[0]/move.recoil[1]));
}

function failedMoveRecoil(id,actor,onDamage) {
  if(!actor.hp || !['supercellslam','steelbeam'].includes(id))return;
  const amount=id==='steelbeam'?Math.round(actor.maxHP/2):Math.floor(actor.maxHP/2);
  const damage=indirectDamage(actor,amount);
  if(damage)onDamage({target:actor,damage,kind:'crash',moveId:id});
}
export function activateNormalGem(move,actor,target) {
  if (move.category==='Status' || move.type!=='Normal' || actor===target || actor.itemId!=='normalgem')return false;
  consumeHeldItem(actor,{kind:'activated'});actor.volatiles ??= {};actor.volatiles.gem=true;return true;
}
export function popBalloon(target) {
  if (target.itemId!=='airballoon')return false;
  consumeHeldItem(target,{kind:'activated'});return true;
}

// One independent damaging-move execution. The owner handles victory, logging,
// switching and move-specific handlers. Require a handler for declared effects
// rather than silently dropping them while testing production move definitions.
export function resolveAttack({id,definition,actor,target,chart,turn,rng,weather='',terrain='',field=null,targetSeat=null,targetAlreadyActed=false,targetAction=null,effectiveSpeeds=null,priority=definition.priority??0,onSecondary,onSecondaryStatus=()=>{},onContactStatus=()=>{},onDamage=()=>{},onHeal=()=>{},onImpact=()=>{},onCrit=()=>{},onDamageMessagesComplete=()=>{},onHitComplete=()=>{},onAfterHitLoop=()=>{},onAfterRecoil=()=>{},onBlocked=()=>{},onTryHit=()=>{},onScreensBroken=()=>{},onBeforeDamage=()=>{},onAfterHit=()=>{},onDamagingHit=()=>{},onAfterDamagingHit=()=>{},onAfterMoveSecondary=()=>{},onPowerBoost=()=>{},automaticEffects=true,allowFaintedSelfdestruct=false,allowFaintedActor=false,sourceActive=!allowFaintedActor||actor.hp>0,deferAfterAttackItems=false,deferAfterMoveSecondary=false}) {  requireRule(definition.category!=='Status','Damaging move required');
  requireRule(!allowFaintedSelfdestruct || definition.selfdestruct==='always','Fainted actor only allowed for an always-selfdestruct move');
  requireRule(!allowFaintedSelfdestruct || actor.hp===0,'Selfdestruct user must faint before hit resolution');
  requireRule(!allowFaintedActor || Boolean(definition.flags?.futuremove),'Fainted actor only allowed for delayed future moves');
  requireRule(sourceActive || allowFaintedActor,'Inactive source only allowed for delayed future moves');
  // Inactive source effects are ignored without mutating the saved combatant.
  const effectsActor=sourceActive?actor:{...actor,abilityId:'noability',itemId:''};
  // Grounding observes suppression on the defending holder during this move,
  // while the attacker's own ability and airborne items/types remain active.
  const ignoreAbility=!allowFaintedActor && (['moldbreaker','teravolt','turboblaze'].includes(effectsActor.abilityId)||id==='sunsteelstrike');
  const activeGrounded=pokemon=>grounded(pokemon,{ignoreAbility:ignoreAbility && pokemon===target});
  if(effectsActor.abilityId==='megasol')weather='sunnyday';
  // futuremove stores an already prepared payload. It does not run ModifyMove
  // again (for example King's Rock), but active Infiltrator is read at impact.
  const move=allowFaintedActor
    ? {...structuredClone(definition),id,...(effectsActor.abilityId==='infiltrator'?{infiltrates:true}:{})}
    : preparedAbilityMove(effectiveMove(id,definition,effectsActor,target,{weather,terrain,gravity:Boolean(field?.gravityTurns),grounded:activeGrounded,turn,rng,effectiveSpeeds}),effectsActor);
  if(id==='knockoff')move.basePower=definition.basePower;
  requireRule(automaticEffects || !move.secondary && !move.secondaries?.length || typeof onSecondary==='function','Secondary effect handler required');
  // Delayed impacts execute saved moveData without another ModifyMove event.
  // Future Sight's stored payload does not contain Mold Breaker's ignoreAbility.
  const effectOptions={chart,ignoreAbility,sunny:weather==='sunnyday',electricTerrain:terrain==='electricterrain',mistyTerrain:terrain==='mistyterrain',grounded:activeGrounded(target),targetAlreadyActed};
  if(!movePrecondition(move,actor,target,{targetAction,targetAlreadyActed,allowFaintedSelfdestruct,allowFaintedActor,turn,terrain,priority,ignoreAbility}))return {hits:0,totalDamage:0,failed:true};
  if(id==='metalburst' && !fixedMoveDamage(id,actor,target,turn))return {hits:0,totalDamage:0,failed:true};
  if(!allowFaintedActor)prepareProtean(actor,move);
  // Champions Unseen Fist marks a contact hit through Protect for a separate
  // quarter-damage final modifier. Fixed-damage callbacks return before that
  // modifier in pinned getDamage, and moves that break Protect need no bypass.
  const unseenFistThroughProtect=effectsActor.abilityId==='unseenfist' && Boolean(move.flags?.contact) &&
    !move.breaksProtect && isProtected(move,actor,target,turn);
  const protectBypass=unseenFistThroughProtect || effectsActor.abilityId==='piercingdrill'&&Boolean(move.flags?.contact);
  const protectResolved=isProtected(move,actor,target,turn)&&!protectBypass;
  let confirmedAbsorbedAbility=null;
  if (!protectResolved && (tryAbilityImmunity(move,effectsActor,target,{
    ignoreAbility,onAbsorbed:id=>{confirmedAbsorbedAbility=id;},
  }) || !moveEffectiveness(move,effectsActor,target,chart) || move.type==='Ground' && target.itemId==='airballoon')) {
    if(id==='highjumpkick' && actor.hp) {const lost=indirectDamage(actor,actor.maxHP/2);if(lost)onDamage({target:actor,damage:lost,kind:'crash',moveId:id});}
    failedMoveRecoil(id,actor,onDamage);
    return {hits:0,totalDamage:0,immune:true,
      ...(confirmedAbsorbedAbility?{absorbedAbilityId:confirmedAbsorbedAbility}:{})};
  }
  let gemActivated=false;
  let impact='normal';
  let criticalHit=false;
  let bodyDamageForMove=0;
  const result=runDamageHits({move,actor:effectsActor,target,turn,rng,accuracyOptions:{weather,ignoreAbility,effectiveSpeeds},allowFaintedSelfdestruct,allowFaintedActor,
    onTryHit:()=>{
      // Pinned Brick Break TryHit clears only the defender's screens,
      // including through Substitute. Protection, miss and immunity exit before this hook.
      if (id==='brickbreak' && field?.sides?.[targetSeat]) {
        const side=field.sides[targetSeat], removed=[];
        for(const name of ['reflect','lightscreen','auroraveil'])
          if(side[name]){delete side[name];removed.push(name);}
        if(removed.length)onScreensBroken({ids:removed,source:actor,target});
      }
      onTryHit(move);
    },
    onBeforeDamage:event=>onBeforeDamage(move,event),
    onDisguiseDamage:event=>onDamage({...event,kind:'disguise',source:actor,direct:false}),
    damageForHit:hit=>{
      criticalHit=false;
      if (sourceActive && !gemActivated)gemActivated=activateNormalGem(move,actor,target);
      const fixed=fixedMoveDamage(id,actor,target,turn);
      // Fixed-damage callbacks bypass modifyDamage in the reference engine.
      // Callbacks such as Super Fang recalculate against the target's current HP
      // on each hit, so do not apply Parental Bond's ordinary-damage modifier.
      if(fixed!==null)return fixed;
      const hitMove=allowFaintedActor?{...move}:effectiveMove(id,move,effectsActor,target,{weather,terrain,grounded,hit,turn,rng,effectiveSpeeds});
      const factor=moveEffectiveness(hitMove,effectsActor,target,chart);
      impact=factor>1 ? 'super' : factor<1 ? 'weak' : 'normal';
      // Variable power is already prepared once, except Triple Axel's hit index.
      hitMove.basePower=id==='tripleaxel' ? 20*hit : move.basePower;
      const speciesId=String(actor.speciesId??'');
      const leekBonus=effectsActor.itemId==='leek' && (speciesId.startsWith('farfetchd') || speciesId==='sirfetchd') ? 2 : 0;
      const merciless=effectsActor.abilityId==='merciless' && ['psn','tox'].includes(target.status);
      const ratio=Math.max(0,Math.min(4,merciless ? 5 : (move.critRatio ?? 1)+(effectsActor.itemId==='scopelens' ? 1 : 0)+(effectsActor.abilityId==='superluck'?1:0)+leekBonus));
      const denominator=[0,24,8,2,1][ratio];
      const critImmune=target.abilityId==='shellarmor'&&!ignoreAbility;
      const forcedCritical=move.willCrit;
      const criticalCandidate=forcedCritical===undefined ? ratio>0 && rng.integer(denominator)===0 : Boolean(forcedCritical);
      const crit=criticalCandidate && !critImmune;
      criticalHit=crit;
      // Showdown getDamage evaluates the critical chance BEFORE the BasePower
      // event. Fickle Beam rolls one 30% doubling chance after that draw.
      if(id==='ficklebeam' && rng.integer(10)<3) {hitMove.basePower*=2;onPowerBoost(id);}
      if(move.parentalBond && hit===2)hitMove.parentalBondHit=true;
      const damage=battleDamage({move:hitMove,actor:effectsActor,target,chart,crit,roll:rng.integer(16),weather,terrain,field,targetSeat,targetAlreadyActed,ignoreAbility});
      // Pinned sim/battle-actions.ts#getDamage: bypassProtect scales normal
      // final damage by 0.25, then enforces the minimum of one HP.
      return unseenFistThroughProtect ? Math.max(1,modify(damage,1024)) : damage;
    },
    onHit:event=>{
      bodyDamageForMove+=event.damage;
      onImpact(impact,event);
      onDamage({target,damage:event.damage,kind:'move',source:effectsActor,moveId:id,direct:event.damage>0,beforeHP:event.beforeHP});
      if(criticalHit && event.damage>0 && !event.disguise)onCrit({hit:event.hit,target,damage:event.damage});
      onDamageMessagesComplete(event);
      event.flushItemMessages?.();
      popBalloon(target);
      thawFromHit(target,move);
      if (event.damage) {
        if (sourceActive && move.drain) {
          const amount=drainHealth(actor,target,Math.round(event.damage*move.drain[0]/move.drain[1]),{ignoreAbility});
          if (amount>0)onHeal({target:actor,amount,moveId:id});
          else if (amount<0)onDamage({target:actor,damage:-amount,kind:'liquidooze',source:target});
        }
      }
      if (automaticEffects)applyPrimarySelfEffects(move,actor,target,rng,effectOptions);
      if (automaticEffects)applyPrimaryMoveEffects(move,actor,target,rng,effectOptions);
      if(id==='clearsmog' && target.hp)resetBoosts(target,{notify:true});
      onAfterHit(move,event);
      if (actor.hp) {
        if (onSecondary)onSecondary(move,event);
        else if (automaticEffects) {
          // Retain the applied secondary source instead of guessing from the
          // final status (which could also come from a contact ability).
          for(const effect of applySecondaryEffects(move,actor,target,rng,effectOptions)) {
            if(effect.kind==='status' && effect.applied)
              onSecondaryStatus({target:effect.target,status:effect.status});
          }
        }
      }
      // Snapshot registered abilities before any DamagingHit handler mutates them.
      const actorAbility=actor.abilityId,targetAbility=target.abilityId;
      // comparePriority treats omitted Order as 2**32: explicit order1 damage
      // and order2 Helmet precede ordinary ability callbacks, not vice versa.
      if(sourceActive && target.abilityId==='innardsout' && !target.hp && event.damage && actor.hp){const lost=indirectDamage(actor,bodyDamageForMove);if(lost)onDamage({target:actor,damage:lost,kind:'innardsout',source:target});}
      const causes=[];
      const lost=sourceActive?contactDamage(move,actor,target,{ignoreAbility,onSource:cause=>causes.push(cause)}):0;
      if (lost)onDamage({target:actor,damage:lost,kind:'contact',source:target,causes});
      // Same-order handlers use cached action speed. Poison Touch wins an equal
      // speed tie via subOrder6 versus other abilities'7 (resolvePriority).
      const sourceFirst=actorAbility==='poisontouch' && (effectiveSpeeds?.actor??actor.stats?.spe??0)>=(effectiveSpeeds?.target??target.stats?.spe??0);
      const contactOptions={chart,weather,terrain,grounded,actorAbility,targetAbility};
      const reportContactStatuses=results=>{for(const result of results)if(result.status && result.applied && result.target?.status===result.status)onContactStatus(result);};
      if(automaticEffects && sourceActive && sourceFirst)reportContactStatuses(contactStatusAbility(move,actor,target,rng,{...contactOptions,phase:'source'}));
      if(!ignoreAbility && targetAbility==='thermalexchange' && move.type==='Fire' && target.hp)changeBoosts(target,{atk:1},{effect:'thermalexchange'});
      onDamagingHit(move,event);
      if(automaticEffects && sourceActive)reportContactStatuses(contactStatusAbility(move,actor,target,rng,{...contactOptions,phase:'target'}));
      if(automaticEffects && sourceActive && !sourceFirst)reportContactStatuses(contactStatusAbility(move,actor,target,rng,{...contactOptions,phase:'source'}));
      onAfterDamagingHit(move,event);
      if(automaticEffects && sourceActive && event.damage>0 && !target.hp) {
        moxieAfterFaint(actor);
        if(actor.hp>0 && actor.abilityId==='eelevate') {
          const stats=['atk','def','spa','spd','spe'];
          let best=stats[0];for(const stat of stats.slice(1))if(actor.stats[stat]>actor.stats[best])best=stat;
          changeBoosts(actor,{[best]:1},{effect:'eelevate'});
        }
      }
      if(id==='knockoff' && target.itemId!==target.megaStoneId && !isFormLockedHeldItem(target) && (!target.hp||ignoreAbility||target.abilityId!=='stickyhold'))consumeHeldItem(target,{kind:'removed'});
      onHitComplete(event);
    },
    onSubstitute:event=>{
      onImpact(impact,event);
      // Substitute.onTryPrimaryHit in fixed Showdown applies recoil immediately
      // for each absorbed hit; this damage is not in move.totalDamage, whose
      // separate end-of-move recoil covers only damage to the actual Pokemon.
      if (event.damage>0) {
        const recoil=attackRecoil(move,actor,event.damage);
        if(recoil)onDamage({target:actor,damage:recoil,kind:'recoil',moveId:id});
      }
      if (sourceActive && move.drain) {
        const amount=drainHealth(actor,target,Math.ceil(event.damage*move.drain[0]/move.drain[1]),{ignoreAbility});
        if(amount>0)onHeal({target:actor,amount,moveId:id});
      }
      popBalloon(target);
      if (automaticEffects)applyPrimarySelfEffects(move,actor,target,rng,effectOptions);
      if (automaticEffects)applySubstituteSecondarySelfEffects(move,actor,rng,effectOptions);
      onAfterHit(move,event);
      onAfterDamagingHit(move,event);
      onHitComplete(event);
    }});
  // A confirmed Protect block is public before any shield contact penalty.
  if(result.blocked==='protect')onBlocked({target,moveId:id});
  if(result.hits)onAfterHitLoop(move,result);
  if(id==='highjumpkick' && actor.hp && (result.missed || result.blocked==='protect')) {
    const lost=indirectDamage(actor,actor.maxHP/2);if(lost)onDamage({target:actor,damage:lost,kind:'crash',moveId:id});
  }
  if(actor.hp && (result.missed || result.blocked==='protect'))failedMoveRecoil(id,actor,onDamage);
  if(id==='steelbeam' && result.hits && actor.hp) {
    const damage=indirectDamage(actor,Math.round(actor.maxHP/2));
    if(damage)onDamage({target:actor,damage,kind:'recoil',moveId:id});
  }
  if(result.blocked==='protect' && move.flags?.contact && actor.hp) {
    if(target.volatiles?.kingsshield?.turn===turn)changeBoosts(actor,{atk:-1},{source:target,effect:'kingsshield'});
    if(target.volatiles?.silktrap?.turn===turn)changeBoosts(actor,{spe:-1},{source:target,effect:'silktrap'});
    if(target.volatiles?.spikyshield?.turn===turn) {const lost=indirectDamage(actor,actor.maxHP/8);if(lost)onDamage({target:actor,damage:lost,kind:'spikyshield'});}
  }
  // The pinned Champions hitStepMoveHitLoop excludes Substitute from move.totalDamage.
  // Recoil and Shell Bell use body damage; Life Orb still triggers on a successful hit.
  const bodyDamage=result.totalDamage-(result.substituteDamage??0);
  const recoil=attackRecoil(move,actor,bodyDamage);
  if (recoil)onDamage({target:actor,damage:recoil,kind:'recoil',moveId:id});
  if(result.hits)onAfterRecoil(move,result);
  const afterMove=deferAfterMoveSecondary ? null : onAfterMoveSecondary(move,result);
  const item=deferAfterAttackItems ? 0 : afterAttackItems(move,actor,bodyDamage,{forcedSwitch:Boolean(afterMove?.actorForcedSwitch),hitOccurred:result.hits>0});
  if (item<0)onDamage({target:actor,damage:-item,kind:'item',itemId:'lifeorb'});
  if (automaticEffects && result.hits) {
    applyPostMoveSelfBoost(move,actor);
    if(sourceActive)whiteHerb(actor);whiteHerb(target);
  }
  if(id==='doubleshock' && result.hits && actor.hp)actor.types=actor.types.map(type=>type==='Electric'?'???':type);
  if(result.hits && actor.volatiles?.charge && move.type==='Electric')delete actor.volatiles.charge;
  if (actor.volatiles)delete actor.volatiles.gem;
  return result;
}
