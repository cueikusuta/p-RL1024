import { isSingleMoveDittoSet } from './catalog-rules.js';
import { requireRule } from './draft.js';
import { calculateChampionsStats } from './champions-stats.js';
import { genderForSpeciesInstance } from './pokemon-gender.js';
import { movePP } from './battle-mechanics.js';
import { CONNECTED_STATUS_MOVES } from './battle-status-moves.js';
import { DYNAMIC_BATTLE_FORMS_BY_ABILITY } from './battle-forms.js';

// Explicit integration gate: every admitted catalog effect has an engine handler.
export const CONNECTED_ABILITIES=new Set(['imposter','analytic','aftermath','blaze','chlorophyll','drizzle','drought','earlybird','effectspore','flamebody','flashfire','hugepower','innerfocus','insomnia','levitate','lightningrod','limber','liquidooze','moxie','multiscale','naturalcure','noguard','overgrow','protean','purepower','rockhead','roughskin','sandforce','sandrush','sandstream','scrappy','sheerforce','solidrock','static','sturdy','swarm','technician','torrent','voltabsorb','waterabsorb']);
for(const id of ['intimidate','speedboost','supremeoverlord','goodasgold','clearbody','hypercutter','competitive','defiant','moldbreaker','infiltrator','cursedbody','trace'])CONNECTED_ABILITIES.add(id);
for(const id of ['berserk','eelevate','fairyaura','flowerveil','guarddog','healer','illuminate','libero','moody','oblivious','quickdraw','shellarmor','steadfast'])CONNECTED_ABILITIES.add(id);
// Flower Veil and Healer only act on adjacent allies; this one-active-per-side
// battle format has no ally target, so those reference callbacks are inert.
for(const id of 'bigpecks bulletproof cheekpouch cloudnine contrary corrosion dryskin eartheater electromorphosis firemane fluffy furcoat gooey grassysurge harvest heatproof hungerswitch icebody immunity justified leafguard magician motordrive mummy pickpocket poisontouch psychicsurge punkrock purifyingsalt reckless sandspit seedsower sniper solarpower spicyspray stamina stickyhold strongjaw tangledfeet toxicdebris unseenfist vitalspirit wanderingspirit waterbubble weakarmor'.split(' '))CONNECTED_ABILITIES.add(id);
for(const id of ['thickfat','toughclaws','filter','megalauncher','mistysurge','asonespectrier','vesselofruin','swordofruin'])CONNECTED_ABILITIES.add(id);
for(const id of 'multitype orichalcumpulse hadronengine intrepidsword asoneglastrier teravolt turboblaze prismarmor'.split(' '))CONNECTED_ABILITIES.add(id);
export const CONNECTED_ITEMS=new Set(['airballoon','bigroot','blackbelt','blackglasses','charcoal','chartiberry','chestoberry','choicescarf','chopleberry','cobaberry','damprock','electricseed','expertbelt','fairyfeather','focussash','hardstone','heatrock','ironball','kasibberry','leftovers','lifeorb','lumberry','magnet','mentalherb','miracleseed','muscleband','mysticwater','normalgem','poisonbarb','rockyhelmet','scopelens','shellbell','shucaberry','silkscarf','sitrusberry','spelltag','whiteherb','widelens','wiseglasses','yacheberry']);
for(const id of 'dracoplate dreadplate skyplate meadowplate mindplate ironplate pixieplate zapplate insectplate earthplate fistplate splashplate flameplate spookyplate stoneplate toxicplate icicleplate rustedsword'.split(' '))CONNECTED_ITEMS.add(id);
const CONNECTED_SPECIAL_ATTACKS=new Set('eruption fakeout firstimpression freezedry grassknot heavyslam hex hurricane knockoff lowkick metalburst poltergeist ragefist ragingbull rapidspin risingvoltage stompingtantrum suckerpunch superfang tripleaxel weatherball'.split(' '));
const CONNECTED_NESTED_MOVE_HANDLERS=new Map(['alluringvoice','direclaw','spiritshackle','throatchop'].map(id=>[id,['secondary.onHit']]));
for(const id of ['redcard','ejectbutton'])CONNECTED_ITEMS.add(id);
for(const id of 'keeneye magicguard prankster superluck snowwarning emergencyexit regenerator mirrorarmor ironfist magicbounce unaware disguise illusion galewings pixilate thermalexchange electricsurge innardsout parentalbond aerilate megasol stalwart refrigerate piercingdrill adaptability shadowtag dragonize guts skilllink swiftswim sharpness auraguard cutecharm damp soundproof synchronize merciless owntempo poisonpoint pressure marvelscale sapsipper whitesmoke'.split(' '))CONNECTED_ABILITIES.add(id);
for(const id of 'sharpbeak lightclay twistedspoon softsand nevermeltice shedshell passhoberry roseliberry wacanberry metalcoat'.split(' '))CONNECTED_ITEMS.add(id);
for(const id of 'brightpowder habanberry kingsrock leek lightball occaberry psychicseed quickclaw'.split(' '))CONNECTED_ITEMS.add(id);
CONNECTED_ITEMS.add('grassyseed');
for(const id of 'compoundeyes forecast frisk grasspelt hustle mimicry queenlymajesty screencleaner stancechange surgesurfer zerotohero unburden unnerve armortail aromaveil slushrush sweetveil'.split(' '))CONNECTED_ABILITIES.add(id);
for(const id of 'blizzard thunder venoshock acrobatics brickbreak clearsmog sludgewave wavecrash shadowclaw airslash razorshell powergem trailblaze boomburst overdrive thunderpunch'.split(' '))CONNECTED_SPECIAL_ATTACKS.add(id);
for(const id of 'alluringvoice appleacid avalanche barbbarrage beakblast bite ceaselessedge clangingscales counter direclaw doubleshock electroshot endeavor expandingforce fissure futuresight gravapple grassyglide guillotine gyroball hammerarm heatcrash highhorsepower highjumpkick horndrill hornleech hyperbeam icespinner infernalparade jawlock lastrespects meteorassault mirrorcoat mistyexplosion mountaingale mortalspin outrage psyshieldbash psychicfangs ragingfury sheercold shellsidearm sparklingaria spiritshackle stoneaxe storedpower thief throatchop tropkick watershuriken'.split(' '))CONNECTED_SPECIAL_ATTACKS.add(id);
for(const id of 'steelbeam supercellslam aurawheel ficklebeam twinbeam ruination'.split(' '))CONNECTED_SPECIAL_ATTACKS.add(id);
export function connectedProductionMove(definition,id) {
  // Phazing status moves have their own authoritative switch resolver.
  if(id==='roar' || id==='whirlwind')return definition?.category==='Status' && definition.forceSwitch===true;
  if(definition?.category==='Status')return CONNECTED_STATUS_MOVES.has(id);
  // Self-Destruct and Explosion share the fixed-reference always-faint
  // declaration and the same Battle resolver; keep this metadata-driven.
  if(definition?.selfdestruct==='always')return true;
  const handlers=definition?.referenceHandlers??[];
  const nestedHandlers=handlers.filter(handler=>handler.includes('.'));
  if(nestedHandlers.length && JSON.stringify(handlers)!==JSON.stringify(CONNECTED_NESTED_MOVE_HANDLERS.get(id)))return false;
  const explicitlyHandled=CONNECTED_SPECIAL_ATTACKS.has(id);
  if(definition?.volatileStatus && definition.volatileStatus!=='partiallytrapped' && !explicitlyHandled)return false;
  if(definition?.self?.volatileStatus && definition.self.volatileStatus!=='glaiverush' && !explicitlyHandled)return false;
  if(explicitlyHandled || ['solarbeam','explosion','uturn','voltswitch','flipturn','dragontail'].includes(id))return true;
  return definition && definition.category!=='Status' && !definition.referenceHandlers?.length && !definition.selfSwitch && !definition.forceSwitch && !definition.selfdestruct && definition.basePower>0;
}
export function productionCombatant(set,data) {
  requireRule(data?.referenceCommit==='b8f23c99629132611ba59e99480cb148ea8c5d3d','Missing fixed battle data');
  const species=data.species[set.speciesId],nature=data.natures[set.nature];
  requireRule(species && nature,'Unknown species or nature');
  requireRule(data.abilities?.[set.abilityId],'Missing battle ability data: '+set.abilityId);
  requireRule(CONNECTED_ABILITIES.has(set.abilityId),'Battle ability not connected: '+set.abilityId);
  const stone=data.megaStones?.[set.itemId];
  requireRule(!set.mega || stone && stone.speciesId===set.speciesId && stone.targetSpeciesId===set.mega.speciesId && stone.abilityId===set.mega.abilityId,'Invalid mega stone/form');
  requireRule(!stone || set.mega,'Missing mega form');
  requireRule(!set.mega || CONNECTED_ABILITIES.has(set.mega.abilityId),'Battle mega ability not connected: '+set.mega?.abilityId);
  for(const abilityId of [set.abilityId,set.mega?.abilityId].filter(Boolean))for(const formId of DYNAMIC_BATTLE_FORMS_BY_ABILITY[abilityId]??[])
    requireRule(Boolean(data.battleForms?.[formId]??data.species?.[formId]),'Missing battle form data for '+abilityId+': '+formId);
  requireRule(!set.itemId || data.items?.[set.itemId],'Missing battle item data: '+set.itemId);
  requireRule(!set.itemId || CONNECTED_ITEMS.has(set.itemId) || stone,'Battle item not connected: '+set.itemId);
  requireRule(isSingleMoveDittoSet(set) || set.moves?.length===4 && new Set(set.moves).size===4,'Invalid move set');
  const moves=set.moves.map(id=>{
    const m=data.moves[id];requireRule(connectedProductionMove(m,id),'Battle move not connected: '+id);
    const pp=movePP(m.pp,m.noPPBoosts);
    return {id,name:m.name,pp,maxPP:pp};
  });
  const stats=calculateChampionsStats(species.baseStats,set.statPoints,nature);
  const publicSpeciesName=set.publicSpeciesName ?? (set.mega ? species.name : set.speciesName ?? set.speciesNameJa ?? set.speciesId);
  return {instanceId:set.instanceId,speciesId:set.speciesId,baseSpeciesId:set.speciesId,speciesName:publicSpeciesName,
    gender:genderForSpeciesInstance(species,set.instanceId ?? set.setId),
    level:50,stats,hp:stats.hp,maxHP:stats.hp,moves,types:[...species.types],originalTypes:[...species.types],weightkg:species.weightkg,
    nature:set.nature,itemId:set.itemId,abilityId:set.abilityId,originalAbilityId:set.abilityId,cost:set.cost,
    ...(set.megaPreview?{megaPreview:structuredClone(set.megaPreview)}:{}),
    baseSpeciesName:publicSpeciesName,statPoints:structuredClone(set.statPoints),...(set.mega ? {mega:structuredClone(set.mega),megaStoneId:set.itemId,megaEvolved:false} : {}),
    boosts:{atk:0,def:0,spa:0,spd:0,spe:0,accuracy:0,evasion:0},atkStage:0,defStage:0,
    status:'',statusState:{},volatiles:{},entered:false,activeTurns:0,activeMoveActions:0,timesAttacked:0};
}
