import test from 'node:test';
import assert from 'node:assert/strict';
import {playableSets} from '../src/production-runtime.js';
import {projectRnadObservation,rnadLegalActions}
  from '../src/ml/legal-action-adapter.js';
import {encodeRnadThreePhaseDecisionV4}
  from '../src/ml/rnad-phase-features-v4.js';
import {encodeRnadThreePhaseDecisionV5,publicSpeciesProfiles,
  RNAD_THREE_PHASE_FEATURE_SCHEMA_V5}
  from '../src/ml/rnad-phase-features-v5.js';
import data from '../data/catalog/battle-data.json' with {type:'json'};

const distinct=new Map();
for(const p of playableSets)if(!p.mega&&p.speciesNameJa&&
    data.species?.[p.speciesId]?.baseStats&&!distinct.has(p.speciesNameJa))
  distinct.set(p.speciesNameJa,p);
const all=[...distinct.values()];
const types=p=>[...data.species[p.speciesId].types].sort().join('/');
const grouped=new Map();
for(const p of all){const k=types(p);if(!grouped.has(k))grouped.set(k,[]);
  grouped.get(k).push(p);}
const own=all.slice(0,6).map((p,i)=>({...p,instanceId:'own-'+i,
  stats:{...data.species[p.speciesId].baseStats},
  types:data.species[p.speciesId].types,
  baseStats:data.species[p.speciesId].baseStats, hp:100,maxHP:100}));
const legal=rnadLegalActions({phase:'selection',self:{team:own},
  opponent:{species:[]}});
function observation(names,secret='first'){
  return projectRnadObservation({phase:'selection',round:1,
    score:[0,0],self:{team:structuredClone(own),credits:10},
    opponent:{species:names,privateTeam:[{moves:[secret]}]}},{seat:0});
}
test('v5 distinguishes public same-type species alias suppressed by v4',()=>{
  let compared=0;
  for(const choices of grouped.values()){
    if(choices.length<2)continue;
    const [x,y]=choices;
    const others=all.filter(p=>p.speciesNameJa!==x.speciesNameJa&&
      p.speciesNameJa!==y.speciesNameJa).slice(0,5).map(p=>p.speciesNameJa);
    const a=observation([x.speciesNameJa,...others]);
    const b=observation([y.speciesNameJa,...others]);
    if(JSON.stringify(encodeRnadThreePhaseDecisionV4(a,legal))!==
       JSON.stringify(encodeRnadThreePhaseDecisionV4(b,legal)))continue;
    const va=encodeRnadThreePhaseDecisionV5(a,legal),
      vb=encodeRnadThreePhaseDecisionV5(b,legal);
    assert.equal(va.schema,RNAD_THREE_PHASE_FEATURE_SCHEMA_V5);
    assert.equal(va.state.length,96);
    assert.equal(va.actions.length,60);
    assert.ok(va.actions.every(x=>x.length===48&&x.every(Number.isFinite)));
    if(JSON.stringify(va)!==JSON.stringify(vb))compared++;
    if(compared>0)break;
  }
  assert.ok(compared>0,'v5 must distinguish at least one real public v4 alias');
});
test('v5 public six-species profiles ignore secret sets, and are order-independent',()=>{
  const names=all.slice(0,6).map(p=>p.speciesNameJa);
  const a=observation(names,'hidden-attack'),b=observation(names,'other-secret');
  assert.deepEqual(encodeRnadThreePhaseDecisionV5(a,legal),
    encodeRnadThreePhaseDecisionV5(b,legal));
  assert.deepEqual(publicSpeciesProfiles(a),publicSpeciesProfiles(
    observation([...names].reverse())));
});
test('v5 first 48 state/32 action coordinates preserve v4, lead matters',()=>{
  const a=observation(all.slice(0,6).map(p=>p.speciesNameJa));
  const v4=encodeRnadThreePhaseDecisionV4(a,legal);
  const v5=encodeRnadThreePhaseDecisionV5(a,legal);
  assert.deepEqual(v5.state.slice(0,48),v4.state);
  assert.deepEqual(v5.actions.map(x=>x.slice(0,32)),v4.actions);
  const first=legal.findIndex(a=>a[0]===own[0].instanceId);
  const second=legal.findIndex(a=>a[0]===own[1].instanceId&&
    a.slice(1).join('|')===legal[first].slice(1).join('|'));
  if(second>=0)assert.notDeepEqual(v5.actions[first],v5.actions[second]);
});
test('v5 unknown public species and private-only information cannot fabricate profiles',()=>{
  assert.throws(()=>encodeRnadThreePhaseDecisionV5(
    observation(['unregistered',...all.slice(1,6).map(p=>p.speciesNameJa)]),
    legal),TypeError);
});
