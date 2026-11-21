import test from 'node:test';
import assert from 'node:assert/strict';
import {projectRnadObservation} from '../src/ml/legal-action-adapter.js';
import {playableSets} from '../src/production-runtime.js';
import {encodeRnadThreePhaseDecisionV5} from '../src/ml/rnad-phase-features-v5.js';
import {encodeRnadThreePhaseDecisionV6,RNAD_V6_STATE_FEATURES,
  RNAD_V6_ACTION_FEATURES}
  from '../src/ml/rnad-phase-features-v6.js';

const pokemon=(i)=>({instanceId:'own-'+i,speciesId:'s'+i,
  types:i===0?['Fire']:['Water'],hp:100-i*30,maxHP:100,
  status:i===1?'brn':null,stats:{hp:100,atk:110,def:100,spa:105,spd:95,spe:90+i*10},
  boosts:{atk:0,def:0,spa:0,spd:0,spe:0},cost:10,
  moves:[{id:'flamethrower',pp:11,maxPP:15},'protect'],
  itemId:'leftovers',abilityId:'blaze'});
function obs(secret='one'){
  return projectRnadObservation({
    requestType:'turn',turn:6,phase:'battle',
    self:{active:0,team:[0,1,2].map(pokemon),canMega:true,canSwitch:true},
    opponent:{speciesId:'opponent-1',types:['Grass'],hpPercent:70,
      status:'par',privateTeam:[{moves:[secret],itemId:secret}],
      publicObservation:{usedMoves:['surf'],observedItemIds:[],observedAbilityIds:[]}},
    field:{weather:'rain',terrain:'electric',trickRoomTurns:2,
      sideEffects:[[{'id':'stealthrock','layers':1}],[{'id':'reflect','duration':3}]]}
  },{seat:0});
}
const actions=[{kind:'move',index:0},{kind:'switch',index:1}];
test('v6 encodes large meaningful public Battle tensors and preserves v5 prefix',()=>{
  const o=obs();const x=encodeRnadThreePhaseDecisionV6(o,actions);
  const base=encodeRnadThreePhaseDecisionV5(o,actions);
  assert.equal(x.state.length,RNAD_V6_STATE_FEATURES);
  assert.deepEqual(x.state.slice(0,96),base.state);
  assert.equal(x.actions.length,2);
  assert.ok(x.actions.every(row=>row.length===RNAD_V6_ACTION_FEATURES&&
    row.every(Number.isFinite)));
  assert.deepEqual(x.actions.map(row=>row.slice(0,48)),base.actions);
  assert.notDeepEqual(x.actions[0].slice(48),x.actions[1].slice(48));
  assert.ok(x.state.slice(96).some(v=>v!==0));
});
test('v6 ignores opponent private sets and unaccepted hostile history fields',()=>{
  const a=obs('secret-A'),b=obs('secret-B');
  assert.deepEqual(encodeRnadThreePhaseDecisionV6(a,actions),
                   encodeRnadThreePhaseDecisionV6(b,actions));
  // The authoritative seat projection must strip arbitrary injected memory.
  assert.equal(Object.hasOwn(a,'publicOpponentHistory'),false);
});
test('v6 changes only Battle-visible fields when public information changes',()=>{
  const a=obs(),b=structuredClone(a);
  b.opponent.hpPercent=15;
  b.field.weather='sun';
  const va=encodeRnadThreePhaseDecisionV6(a,actions);
  const vb=encodeRnadThreePhaseDecisionV6(b,actions);
  assert.notDeepEqual(va.state,vb.state);
  const c=structuredClone(a);
  c.publicOpponentHistory=[{speciesId:'revealed-previous',
    types:['Water'],hpPercent:50,status:'brn',lastSeen:2,
    usedMoves:['surf'],observedAbilityIds:[],observedItemIds:[]}];
  assert.notDeepEqual(va.state,encodeRnadThreePhaseDecisionV6(c,actions).state);
});
test('v6 damage-band features react to only public species, HP and own power',()=>{
  const a=obs();
  a.opponent.speciesId=playableSets.find(p=>p.speciesId)?.speciesId;
  a.opponent.types=['Grass'];
  a.opponent.hpPercent=80;
  const x=encodeRnadThreePhaseDecisionV6(a,actions);
  // index 93/94: broad lower/upper own move damage ratio proxies.
  assert.ok(x.actions[0][93]>0);
  assert.ok(x.actions[0][94]>=x.actions[0][93]);
  const b=structuredClone(a);
  b.opponent.hpPercent=25;
  const y=encodeRnadThreePhaseDecisionV6(b,actions);
  assert.ok(y.actions[0][93]>=x.actions[0][93]);
  const c=structuredClone(a);
  c.opponent.hiddenTeam=[{stats:{def:999,spd:999}}];
  assert.deepEqual(encodeRnadThreePhaseDecisionV6(c,actions),x);
});
test('v6 rejects oversized history and non-Battle input',()=>{
  const a=obs();
  a.publicOpponentHistory=Array.from({length:4},()=>({speciesId:'x',types:[]}));
  assert.throws(()=>encodeRnadThreePhaseDecisionV6(a,actions),TypeError);
  const x=obs();x.phase='selection';
  assert.throws(()=>encodeRnadThreePhaseDecisionV6(x,actions),TypeError);
});
