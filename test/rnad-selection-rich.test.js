import test from 'node:test';
import assert from 'node:assert/strict';
import {playableSets} from '../src/production-runtime.js';
import {projectRnadObservation,rnadLegalActions}
  from '../src/ml/legal-action-adapter.js';
import {encodeRichSelectionActor,createSelectionRichMatch,selectionRichVocab}
  from '../scripts/rnad/export-selection-rich.mjs';
import data from '../data/catalog/battle-data.json' with {type:'json'};

const unique=new Map();
for(const p of playableSets)if(!p.mega&&p.speciesNameJa&&
    data.species?.[p.speciesId]?.baseStats&&!unique.has(p.speciesNameJa))
  unique.set(p.speciesNameJa,p);
const pool=[...unique.values()];
const team=pool.slice(0,6).map((p,i)=>({
  ...structuredClone(p),instanceId:'own-'+i,
  stats:{...data.species[p.speciesId].baseStats},
  baseStats:{...data.species[p.speciesId].baseStats},
  types:[...data.species[p.speciesId].types],hp:100,maxHP:100
}));
function build(secret='hidden-A'){
  const observation=projectRnadObservation({
    phase:'selection',round:1,score:[0,0],
    self:{team,credits:8},
    opponent:{species:pool.slice(8,14).map(p=>p.speciesNameJa),
      privateTeam:[{moves:[secret],itemId:secret,abilityId:secret}]}
  },{seat:0});
  return {observation,legal:rnadLegalActions(observation)};
}
test('rich tensors carry exact legal canonical 60 and fixed public/own dimensions',()=>{
  const {observation,legal}=build();
  const a=encodeRichSelectionActor(observation,legal);
  assert.equal(a.own_cats.length,6);
  assert.equal(a.foe_cats.length,6);
  assert.ok(a.own_cats.every(x=>x.length===9));
  assert.ok(a.own_numeric.every(x=>x.length===16));
  assert.ok(a.foe_cats.every(x=>x.length===3));
  assert.ok(a.foe_numeric.every(x=>x.length===6));
  assert.equal(a.legal.length,60);
  assert.equal(a.vocab_sha256,selectionRichVocab.sha256);
  assert.match(a.vocab_sha256,/^[0-9a-f]{64}$/);
  const tampered=legal.slice();
  [tampered[0],tampered[1]]=[tampered[1],tampered[0]];
  assert.throws(()=>encodeRichSelectionActor(observation,tampered),/Noncanonical/);
});
test('nonpublic opponent data never changes rich actor tensors',()=>{
  const x=build('red'),y=build('blue');
  assert.deepEqual(encodeRichSelectionActor(x.observation,x.legal),
                   encodeRichSelectionActor(y.observation,y.legal));
  const changed=structuredClone(y.observation);
  changed.self.team[0].stats.spe+=50;
  assert.notDeepEqual(encodeRichSelectionActor(x.observation,x.legal),
                      encodeRichSelectionActor(changed,y.legal));
});
test('a real complete accepted teacher Match gives separate core/lead training targets',()=>{
  const match=createSelectionRichMatch(2474001);
  assert.equal(match.completed,true);
  assert.equal(match.examples.length,2*match.rounds);
  assert.match(match.teacher_model_sha256,/^[a-f0-9]{64}$/);
  for(const e of match.examples){
    assert.equal(e.actor.legal.length,60);
    assert.equal(e.targets.core.length,20);
    assert.equal(e.targets.lead.length,6);
    assert.equal(e.targets.utilities.length,60);
    const utility=e.targets.utilities;
    assert.ok(Math.abs(Math.max(...utility)-utility[e.choice])<1e-5);
    assert.ok(e.state.length===96&&e.actions.length===60);
  }
});
