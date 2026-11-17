import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {
  loadVerifiedASelection,checkASelectionParity,
  scoreFrozenASelection,selectFrozenAAction
} from '../scripts/rnad/selection-a-public-infer.mjs';

const dims={species:16,names:16,abilities:16,items:16,moves:16,types:16};
function actor(){
  const legal=[];
  for(let i=0;i<6;i++)for(let j=0;j<6;j++)for(let k=j+1;k<6;k++)
    if(i!==j&&i!==k)legal.push([i,j,k]);
  assert.equal(legal.length,60);
  return {schema:'rnad-i5-selection-public-actor-tensors-v1',
    vocab_sha256:'b'.repeat(64),legal,
    own_cats:Array.from({length:6},()=>[1,2,3,4,5,6,7,8,9]),
    foe_cats:Array.from({length:6},()=>[1,8,9]),
    own_numeric:Array.from({length:6},()=>Array(16).fill(.5)),
    foe_numeric:Array.from({length:6},()=>Array(6).fill(.5))};
}
function model(){
  const shapes={
    'embed.species.weight':[16,12],
    'embed.names.weight':[16,12],
    'embed.abilities.weight':[16,12],
    'embed.items.weight':[16,12],
    'embed.moves.weight':[16,12],
    'embed.types.weight':[16,8],
    'embed.own_proj.0.weight':[64,72],
    'embed.own_proj.0.bias':[64],
    'embed.foe_proj.0.weight':[64,26],
    'embed.foe_proj.0.bias':[64],
    'pair.weight':[64,64],
    'pair.bias':[64],
    'opponent.weight':[64,64],
    'opponent.bias':[64],
    'core.0.weight':[96,256],
    'core.0.bias':[96],
    'core.2.weight':[1,96],
    'core.2.bias':[1],
    'lead.0.weight':[96,192],
    'lead.0.bias':[96],
    'lead.2.weight':[1,96],
    'lead.2.bias':[1],
    'trio_index':[60],
    'lead_index':[60],
    'trio_rows':[20,3],
  };
  const tensors=Object.fromEntries(Object.entries(shapes).map(
    ([name,shape])=>[name,{shape,values:Array(shape.reduce((x,y)=>x*y,1)).fill(0)}]));
  const legal=actor().legal;
  const keys=[...new Set(legal.map(x=>[...x].sort((a,b)=>a-b).join(',')))].sort(
    (a,b)=>{const x=a.split(',').map(Number),y=b.split(',').map(Number);
           return x[0]-y[0]||x[1]-y[1]||x[2]-y[2];});
  tensors.trio_index.values=legal.map(x=>
    keys.indexOf([...x].sort((a,b)=>a-b).join(',')));
  tensors.lead_index.values=legal.map(x=>x[0]);
  tensors.trio_rows.values=keys.flatMap(k=>k.split(',').map(Number));
  tensors['core.2.bias'].values=[.7];
  tensors['lead.2.bias'].values=[.2];
  return {schema:'rnad-i5-A-public-actor-js-weights-v1',
    source_model_filename:'selection-A-2048-e40-s90253.pt',
    source_model_sha256:'a'.repeat(64),
    source_sha:'a7224b05eeef49650794b24546576119dffed951',
    engine_sha:'a'.repeat(40),
    teacher:'rnad-aq-public-strategy-selection-v1',
    public_actor_vocabulary:{
      schema:'rnad-i5-selection-rich-vocabulary-v1',
      sha256:'b'.repeat(64),sizes:dims},
    labels_as_inference_inputs:false,champion_promoted:false,
    tensors,parity_fixture:{actor:actor(),logits:Array(60).fill(.9)}};
}
function verify(obj){
  const raw=JSON.stringify(obj);
  const digest=createHash('sha256').update(raw).digest('hex');
  return loadVerifiedASelection(raw,digest);
}
test('public-only A scoring: verified immutable 60 candidates and parity',()=>{
  const p=verify(model());
  assert.equal(Object.isFrozen(p),true);
  assert.equal(Object.isFrozen(p.tensors['core.0.weight'].values),true);
  assert.equal(scoreFrozenASelection(p,actor()).length,60);
  assert.equal(selectFrozenAAction(p,actor()).index,0);
  assert.ok(checkASelectionParity(p)<1e-5);
});
test('opponent private baggage cannot affect A public actor logits',()=>{
  const p=verify(model()),a=actor();
  const before=scoreFrozenASelection(p,a);
  a.opponent_hidden_sets=[{itemId:'revealed-by-bug',moves:['private']}];
  assert.deepEqual(scoreFrozenASelection(p,a),before);
});
test('tampered, missing SHA, vocabulary mismatch and illegal action rejected',()=>{
  const obj=model(),raw=JSON.stringify(obj),sha=
    createHash('sha256').update(raw).digest('hex');
  assert.throws(()=>loadVerifiedASelection(raw,sha.replace(/./,'0')),
    /SHA256 mismatch/);
  assert.throws(()=>loadVerifiedASelection(raw),'TypeError');
  assert.throws(()=>scoreFrozenASelection(verify(obj),
    {...actor(),vocab_sha256:'c'.repeat(64)}),/canonical legal public/);
  const invalid=actor();invalid.legal.pop();
  assert.throws(()=>scoreFrozenASelection(verify(obj),invalid),
    /canonical legal public/);
  const corrupted=model();corrupted.tensors['core.0.weight'].values[0]=NaN;
  assert.throws(()=>verify(corrupted),/Bad A tensor/);
});
