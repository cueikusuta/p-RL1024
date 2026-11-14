import test from 'node:test';
import assert from 'node:assert/strict';
import {models,getModel} from '../models/registry.mjs';
import * as linear from '../models/linear.mjs';
import * as deep from '../models/deep.mjs';
import * as v2e from '../models/v2e.mjs';
import * as aq from '../models/aq-base.mjs';
import * as deep001 from '../models/deep001.mjs';
import aqData from '../aq-runtime/data/catalog/battle-data.json' with {type:'json'};
import {legalPureRlActions} from '../src/ai/pure-rl-linear.js';

const view=()=>({requestType:'turn',turn:2,field:{weather:null,terrain:null,trickRoomTurns:0},self:{active:0,canSwitch:true,canMega:false,confirmed:false,needsSwitch:false,team:[{hp:50,maxHP:100,status:'',speciesId:'pikachu',types:['Electric'],moves:[{id:'tackle',disabled:false},{id:'recover',disabled:true}]},{hp:80,maxHP:100,status:'',speciesId:'blastoise',types:['Water'],moves:[]}]},opponent:{hpPercent:64,status:'',remaining:2,speciesId:'snorlax',types:['Normal']}});

test('registry IDs and runtime contract are unambiguous',()=>{
  assert.equal(new Set(models.map(m=>m.id)).size,models.length);
  assert.deepEqual(models.map(m=>m.id).sort(),['AQ-Base-001','RL-Deep','RL-Deep-001','RL-Linear','V2E']);
  assert.equal(getModel('RL-Linear').featureVersion,linear.featureVersion);
  assert.equal(getModel('RL-Deep').featureVersion,deep.featureVersion);
  assert.equal(getModel('V2E').featureVersion,v2e.featureVersion);
  assert.equal(getModel('AQ-Base-001').featureVersion,aq.featureVersion);
  assert.equal(getModel('RL-Deep-001').featureVersion,deep001.featureVersion);
  assert.equal(getModel('RL-Deep-001').trainingEntry,null);
  assert.equal(getModel('V2E').status,'frozen');
  assert.equal(getModel('V2E').trainingEntry,null);
  assert.throws(()=>getModel('AQ-Base'),/Unknown model/);
});
test('RL-Deep-001 uses verified frozen weights and public, legal features',()=>{
  const model=deep001.createModel(),a=view(),b=structuredClone(a),legal=legalPureRlActions(a).slice(1);
  b.opponent.hiddenMoves=['secret'];b.opponent.hiddenItem='secret';b.opponent.hiddenAbility='secret';b.opponent.hiddenSelection=['secret'];b.futureWinner=0;
  const rows=deep001.rankActions(model,a,legal,{seat:0});
  assert.ok(rows.length);assert.ok(rows.every(r=>legal.some(x=>JSON.stringify(x)===JSON.stringify(r.action))));
  assert.deepEqual(rows,deep001.rankActions(model,b,legal,{seat:0}));
  assert.deepEqual(rows,deep001.rankActions(deep001.loadModel(deep001.saveModel(model)),a,legal,{seat:0}));
  const changed=structuredClone(model);changed.w[0]+=1e-6;assert.throws(()=>deep001.loadModel(JSON.stringify(changed)),/SHA/);
});
test('AQ-Base-001 digest, legal mask, public input and save/load',()=>{
  const model=aq.createModel(),a=view(),b=structuredClone(a),legal=legalPureRlActions(a).slice(1);
  b.opponent.hiddenMoves=['secret'];b.opponent.hiddenItem='secret';b.opponent.hiddenAbility='secret';b.opponent.hiddenSelection=['secret'];b.futureWinner=0;
  const options={baselineScores:legal.map((_,i)=>100-i),data:aqData};
  const rows=aq.rankActions(model,a,legal,options);
  assert.ok(rows.length);assert.ok(rows.every(r=>legal.some(x=>JSON.stringify(x)===JSON.stringify(r.action))));
  assert.deepEqual(rows,aq.rankActions(model,b,legal,options));
  assert.deepEqual(rows,aq.rankActions(aq.loadModel(aq.saveModel(model)),a,legal,options));
  const tampered=structuredClone(model);tampered.weights[0]+=1;assert.throws(()=>aq.loadModel(JSON.stringify(tampered)),/SHA/);
});
for(const [id,adapter] of [['RL-Linear',linear],['RL-Deep',deep],['V2E',v2e]]) {
test(`${id} masks actions and ignores injected opponent secrets`,()=>{
  const model=adapter.createModel(),a=view(),b=structuredClone(a);
  b.opponent.hiddenMoves=['secret'];b.opponent.hiddenItem='secret';b.opponent.hiddenAbility='secret';b.opponent.hiddenType='secret';b.opponent.hiddenSelection=['secret'];b.futureWinner=0;
  const legal=legalPureRlActions(a).slice(1);
  const ranks=adapter.rankActions(model,a,legal);
  assert.ok(ranks.length);
  assert.ok(ranks.every(r=>legal.some(x=>JSON.stringify(x)===JSON.stringify(r.action))));
  assert.deepEqual(adapter.rankActions(model,a,legal),adapter.rankActions(model,b,legal));
});
test(`${id} deterministic inference survives model save/load`,()=>{
  const model=adapter.createModel();
  const original=adapter.rankActions(model,view());
  assert.deepEqual(adapter.rankActions(model,view()),original);
  assert.deepEqual(adapter.rankActions(adapter.loadModel(adapter.saveModel(model)),view()),original);
});
}
test('RL-Deep training updates only after accepted terminal trajectory',()=>{
  const model=deep.createModel(), before=deep.saveModel(model);
  const session=deep.createSession(model,{train:true,rng:{integer:()=>0}});
  const action=session.select(view());assert.ok(legalPureRlActions(view()).some(a=>JSON.stringify(a)===JSON.stringify(action)));
  assert.equal(deep.saveModel(model),before);session.commit();session.finish(1);
  assert.notEqual(deep.saveModel(model),before);assert.equal(session.stats().updates,1);
});
