// The boost subsystem's explicit, already-confirmed public causes.
// Single source of truth for mechanics notification, Japanese log rendering,
// and public-event projection. These are not permission to infer an ability
// from a private set or to reveal a move which was never executed.
export const confirmedBoostMoveEffects=new Set([
  'stickyweb','octolock','swordsdance','dragondance','bulkup','agility',
  'calmmind','coil','cottonguard','cosmicpower','growth','irondefense',
  'minimize','nastyplot','noretreat','quiverdance','shellsmash','acidarmor',
  'tidyup','clangoroussoul','curse','bellydrum','charm','eerieimpulse',
  'screech','metalsound','featherdance','babydolleyes','partingshot','defog',
  'strengthsap','memento','electroshot','silktrap','kingsshield','scaleshot',
  'clangingscales',
]);
export const confirmedBoostSecondaryMoves=new Set([
  'appleacid','breakingswipe','bulldoze','electroweb','gravapple','hammerarm',
  'razorshell','tropkick','icywind','rocktomb','snarl','lunge',
  'mysticalfire','spiritbreak','aquastep','aurawheel','flamecharge','psyshieldbash',
  'rapidspin','torchsong','trailblaze','armorcannon','closecombat','dracometeor',
  'headlongrush','icehammer','leafstorm','makeitrain','overheat','superpower',
  'luminacrash','bugbuzz','crunch','earthpower','moonblast','shadowball',
  'flashcannon','focusblast','psychic',
]);
export const visibleBoostMoveCauses=new Set([
  ...confirmedBoostMoveEffects,...confirmedBoostSecondaryMoves,
]);
export const confirmedBoostItemEffects=new Set([
  'electricseed','psychicseed','grassyseed',
]);
export const confirmedBoostAbilityEffects=new Set([
  'speedboost','moxie','moody','stamina','weakarmor','justified',
  'thermalexchange','guarddog','steadfast','gooey','berserk','motordrive',
  'sapsipper','lightningrod','eelevate',
]);
export const visibleBoostAbilityCauses=new Set([
  'speedboost','moxie','moody','stamina','weakarmor','justified',
  'thermalexchange','defiant','competitive','guarddog','mirrorarmor','clearbody',
  'steadfast','gooey','berserk','motordrive','sapsipper','lightningrod',
  'eelevate',
]);
export const visibleBoostItemCauses=new Set([
  'electricseed','psychicseed','grassyseed',
]);
