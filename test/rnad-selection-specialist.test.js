import test from 'node:test';
import assert from 'node:assert/strict';
import {legalPureRlSelections} from '../src/ai/pure-rl-phases.js';
import {rnadLegalActions,projectRnadObservation} from '../src/ml/legal-action-adapter.js';
import {encodeRnadThreePhaseDecisionV5} from '../src/ml/rnad-phase-features-v5.js';
import {playableSets} from '../src/production-runtime.js';
import data from '../data/catalog/battle-data.json' with {type:'json'};

// The Python selection specialist ties a trio score across all three lead
// choices. This is a strict equality contract, not a guessed action ordering.
function canonicalTripleIDs() {
  const out=[];
  for(let i=0;i<6;i++)for(let j=0;j<6;j++)for(let k=j+1;k<6;k++)
    if(i!==j&&i!==k)out.push([i,j,k]);
  return out;
}
const unique=new Map();
for(const p of playableSets)if(!p.mega&&data.species?.[p.speciesId]?.baseStats
  &&p.speciesNameJa&&!unique.has(p.speciesNameJa))unique.set(p.speciesNameJa,p);
const pool=[...unique.values()];
const mine=pool.slice(0,6).map((p,i)=>({
  ...structuredClone(p),
  instanceId:`own-${i}`,
  stats:{...data.species[p.speciesId].baseStats},
  baseStats:{...data.species[p.speciesId].baseStats},
  types:[...data.species[p.speciesId].types],
  hp:100,maxHP:100
}));
const enemies=pool.slice(8,14).map(p=>p.speciesNameJa);
function obs({team=mine,names=enemies,secret='unknown'}={}) {
  return projectRnadObservation({
    phase:'selection',round:1,score:[0,0],
    self:{team,credits:8},
    opponent:{species:names,privateTeam:[{moves:[secret],itemId:secret}]}
  },{seat:0});
}
test('legitimate selection enumerates exactly six leads × ten unordered reserve duos',()=>{
  const o=obs(),legal=rnadLegalActions(o);
  assert.equal(legal.length,60);
  const ids=canonicalTripleIDs();
  assert.deepEqual(legal,ids.map(row=>row.map(i=>mine[i].instanceId)));
  const trioKeys=legal.map(row=>[...row].sort().join('|'));
  const counts=new Map();
  for(const k of trioKeys)counts.set(k,(counts.get(k)??0)+1);
  assert.equal(counts.size,20);
  assert.ok([...counts.values()].every(n=>n===3));
  const encoding=encodeRnadThreePhaseDecisionV5(o,legal);
  assert.equal(encoding.actions.length,60);
  assert.ok(encoding.actions.every(x=>x.length===48));
  assert.ok(encoding.actions.every(x=>x.every(Number.isFinite)));
});

test('selection can use public opponent identity but ignores all private opponent sets',()=>{
  const a=obs({secret:'private-a'}),b=obs({secret:'private-b'});
  const legal=rnadLegalActions(a);
  assert.deepEqual(encodeRnadThreePhaseDecisionV5(a,legal),
    encodeRnadThreePhaseDecisionV5(b,legal));
  const choices=pool.slice(15,21).map(p=>p.speciesNameJa);
  const changed=encodeRnadThreePhaseDecisionV5(obs({names:choices}),legal);
  assert.notDeepEqual(changed,
    encodeRnadThreePhaseDecisionV5(a,legal),
    'Different public six-species preview should affect v5 state/action');
});

test('selection input reacts to known own speed, available item, and lead choice',()=>{
  const base=obs(),legal=rnadLegalActions(base);
  const before=encodeRnadThreePhaseDecisionV5(base,legal);
  const team=structuredClone(mine);
  team[0].stats.spe+=75;
  const after=encodeRnadThreePhaseDecisionV5(obs({team}),legal);
  assert.notDeepEqual(after,before,
    'Own known lead speed should change v5 action input');
  const trio=canonicalTripleIDs().findIndex(t=>t[0]===0&&t[1]===1&&t[2]===2);
  const other=canonicalTripleIDs().findIndex(t=>t[0]===1&&t[1]===0&&t[2]===2);
  assert.ok(trio>=0&&other>=0);
  assert.notDeepEqual(before.actions[trio],before.actions[other],
    'Changing the lead of the same trio must affect features');
  const items=[...new Set(playableSets.map(p=>p.itemId).filter(Boolean))].slice(0,40);
  const varied=items.some(item=>{
    const altered=structuredClone(mine);
    altered[0].itemId=item;
    return JSON.stringify(encodeRnadThreePhaseDecisionV5(obs({team:altered}),legal))!==
      JSON.stringify(before);
  });
  assert.ok(varied,'At least one real known own item change should affect features');
});
