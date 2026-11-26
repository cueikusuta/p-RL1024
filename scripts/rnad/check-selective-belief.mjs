import assert from 'node:assert/strict';
import {playableSets} from '../../aq-runtime/src/production-runtime.js';
import data from '../../aq-runtime/data/catalog/battle-data.json' with {type:'json'};
import {worldsFor,createSelectiveReranker} from './selective-belief-search.mjs';

let tested=false;
for(const set of playableSets){
  const peers=playableSets.filter(s=>s.speciesId===set.speciesId);
  if(peers.length<3||peers.length>4)continue;
  const move=set.moves.find(m=>peers.filter(s=>s.moves.includes(m)).length===2);
  if(!move)continue;
  const species=data.species[set.speciesId];
  const raw={opponent:{speciesId:set.speciesId,types:species.types,hpPercent:90,status:'',stages:{}}};
  const pub={revealed:[{active:true,speciesId:set.speciesId,usedMoves:[move]}]};
  const prior=worldsFor(raw,pub,4,false), history=worldsFor(raw,pub,4,true);
  assert(prior.length>=3 && history.length===2 && history.every(x=>x.moves.includes(move)));
  const corruptHidden={...raw,opponent:{...raw.opponent,setId:'private',abilityId:'different',
      itemId:'fake',moves:['private move'],maxHP:0,bench:[{private:true}]}};
  assert.deepEqual(worldsFor(corruptHidden,pub,4,true),history);
  assert.deepEqual(worldsFor(raw,{revealed:[{...pub.revealed[0],speciesId:'different'}]},4,true),[]);
  assert.deepEqual(worldsFor(raw,{revealed:[{...pub.revealed[0],usedMoves:['not-a-catalog-move']}]},4,true),[]);
  tested=true;break;
}
assert(tested,'Expected real catalog variants to exercise selective reveal filtering');
const {rerank}=createSelectiveReranker({mode:'History',model:{}});
for(const observation of [{phase:'draft'},{phase:'selection'},{phase:'battle',requestType:'replacement'}])
  assert.equal(rerank({observation,index:1}),1);
console.log('Verified actual catalog reveal filtering, hidden perturbation invariance, and nonbattle/forced fallback.');
