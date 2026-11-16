import test from 'node:test';
import assert from 'node:assert/strict';
import {playableSets} from '../src/production-runtime.js';
import {projectRnadObservation,rnadLegalActions}
  from '../src/ml/legal-action-adapter.js';
import {encodeRnadThreePhaseDecisionV4}
  from '../src/ml/rnad-phase-features-v4.js';
import {rankStrategySelections}
  from '../src/ai/team-assessment.js';
import data from '../data/catalog/battle-data.json' with {type:'json'};

const distinct=new Map();
for(const p of playableSets)if(!p.mega&&p.speciesNameJa&&
    data.species?.[p.speciesId]&&!distinct.has(p.speciesNameJa))
  distinct.set(p.speciesNameJa,p);
const all=[...distinct.values()];
const types=p=>[...data.species[p.speciesId].types].sort().join('/');
const grouped=new Map();
for(const p of all){const k=types(p);if(!grouped.has(k))grouped.set(k,[]);grouped.get(k).push(p);}
const roster=all.slice(0,6).map((p,i)=>({...p,instanceId:'fixture-'+i}));
function observation(foe){
  const others=all.filter(x=>x.speciesNameJa!==foe.speciesNameJa)
    .slice(0,5).map(x=>x.speciesNameJa);
  return projectRnadObservation({phase:'selection',round:1,
    score:[0,0],budget:10,grant:10,
    self:{credits:10,team:structuredClone(roster)},
    opponent:{species:[foe.speciesNameJa,...others]}},{seat:0});
}
test('public opponent species identity is compressed to types in current v4; audit teacher sensitivity',
  {timeout:60000},()=>{
    assert.equal(roster.length,6);
    const legal=rnadLegalActions({phase:'selection',
      self:{team:roster},opponent:{species:[]}});
    assert.equal(legal.length,60);
    const pairs=[...grouped.values()].filter(g=>g.length>=2)
      .flatMap(g=>g.slice(0,4).slice(1).map(x=>[g[0],x]))
      .slice(0,10);
    assert.ok(pairs.length>0,'Need two public species with identical types');
    let teacherChanges=0,featureAliases=0;
    for(const [x,y] of pairs){
      const a=observation(x),b=observation(y);
      // Hold other five visible foes and our six sets fixed. The current
      // encoder must not leak private set data or magically reconstruct ID.
      if(JSON.stringify(a.opponent.species.slice(1))!==
         JSON.stringify(b.opponent.species.slice(1)))continue;
      const va=encodeRnadThreePhaseDecisionV4(a,legal);
      const vb=encodeRnadThreePhaseDecisionV4(b,legal);
      if(JSON.stringify(va)!==JSON.stringify(vb))continue;
      featureAliases++;
      const ra=rankStrategySelections(a,{catalog:playableSets,data});
      const rb=rankStrategySelections(b,{catalog:playableSets,data});
      if(ra.length===rb.length&&ra.some((row,i)=>
         Math.abs(row.score-rb[i].score)>1e-6||
         JSON.stringify(row.selection)!==JSON.stringify(rb[i].selection)))
        teacherChanges++;
      if(teacherChanges)break;
    }
    assert.ok(featureAliases>0,'Expected public same-type species to alias in v4');
    // An alias is observable; a nonzero teacher difference proves the
    // public AQ ranking uses information lost by the actor.
    assert.ok(teacherChanges>0,
      'Need confirmed teacher score/ranking sensitivity to public species identity');
  });
