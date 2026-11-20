// Regression against the pre-allocation scalar GRU / policy affine order.
// Synthetic tensors are NOT evidence of AI strength or end-to-end throughput.
import test from 'node:test';
import assert from 'node:assert/strict';
import {PersistentRNADInference,RNAD_MODEL_SCHEMA} from '../scripts/rnad/infer.mjs';
import {RNAD_FEATURE_SCHEMA} from '../src/ml/rnad-features.js';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA_V5}
  from '../src/ml/rnad-phase-features-v5.js';

const H=128;
const sigmoid=x=>1/(1+Math.exp(-x));
const relu=x=>x>0?x:0;
const tensorShape=(states,actions)=>({
  'state_proj.weight':[H,states],'state_proj.bias':[H],
  'gru.weight_ih':[3*H,H],'gru.weight_hh':[3*H,H],
  'gru.bias_ih':[3*H],'gru.bias_hh':[3*H],
  'policy_hidden.weight':[H,H+actions],'policy_hidden.bias':[H],
  'policy_out.weight':[1,H],'policy_out.bias':[1],
  'value_hidden.weight':[H,H],'value_hidden.bias':[H],
  'value_out.weight':[1,H],'value_out.bias':[1],
});
function fixture(states,actions,schema){
  const weights={};
  for(const [name,shape] of Object.entries(tensorShape(states,actions))){
    const n=shape.reduce((a,b)=>a*b,1);
    weights[name]={shape,values:Array.from({length:n},(_,i)=>
      (((i*17+name.length*13)%173)-86)/20000)};
  }
  return {schema:RNAD_MODEL_SCHEMA,hidden:H,feature_schema:schema,
    state_features:states,action_features:actions,weights};
}
// Scalar reference kept intentionally independent of production scratch code.
function originalLinear(params,prefix,x,out){
  const w=params[prefix+'.weight'],b=params[prefix+'.bias'];
  const result=new Array(out);
  for(let i=0;i<out;i++){
    let sum=b[i],offset=i*x.length;
    for(let j=0;j<x.length;j++)sum+=w[offset+j]*x[j];
    result[i]=sum;
  }
  return result;
}
function originalInfer(net,state,actions,seat,advance){
  const previous=net.memory[seat],params=net.params;
  const input=originalLinear(params,'state_proj',state,H).map(relu);
  const wI=params['gru.weight_ih'],wH=params['gru.weight_hh'];
  const bI=params['gru.bias_ih'],bH=params['gru.bias_hh'];
  const gi=new Array(3*H),gh=new Array(3*H);
  for(let i=0;i<3*H;i++){
    let a=bI[i],b=bH[i],offset=i*H;
    for(let j=0;j<H;j++){a+=wI[offset+j]*input[j];b+=wH[offset+j]*previous[j];}
    gi[i]=a;gh[i]=b;
  }
  const memory=new Array(H);
  for(let i=0;i<H;i++){
    const reset=sigmoid(gi[i]+gh[i]);
    const update=sigmoid(gi[H+i]+gh[H+i]);
    const candidate=Math.tanh(gi[2*H+i]+reset*gh[2*H+i]);
    memory[i]=(1-update)*candidate+update*previous[i];
  }
  const logits=actions.map(a=>{
    const hidden=originalLinear(params,'policy_hidden',[...memory,...a],H).map(relu);
    return originalLinear(params,'policy_out',hidden,1)[0];
  });
  const value=Math.tanh(originalLinear(params,'value_out',
    originalLinear(params,'value_hidden',memory,H).map(relu),1)[0]);
  if(advance)net.memory[seat]=memory;
  return {logits,value,memory};
}

for(const {states,actions,schema} of [
  {states:32,actions:16,schema:RNAD_FEATURE_SCHEMA},
  {states:96,actions:48,schema:RNAD_THREE_PHASE_FEATURE_SCHEMA_V5}
]){
  test('persistent GRU scratch exactly matches scalar candidate order '+schema,()=>{
    const payload=fixture(states,actions,schema);
    const optimized=new PersistentRNADInference(payload);
    const reference=new PersistentRNADInference(payload);
    let prior=null;
    for(let step=0;step<8;step++){
      const seat=step%2,advance=step%3!==0;
      const state=Array.from({length:states},(_,i)=>((i*3+step)%19-9)/9);
      const candidates=Array.from({length:[1,3,60,8][step%4]},(_,k)=>
        Array.from({length:actions},(_,j)=>((k*5+j+step)%23-11)/13));
      const actual=optimized.infer(state,candidates,seat,{advance});
      const expected=originalInfer(reference,state,candidates,seat,advance);
      assert.deepStrictEqual(actual,expected);
      assert.deepStrictEqual(optimized.memory,reference.memory);
      if(prior)assert.notStrictEqual(actual.memory,prior,
        'Caller-visible memory must never alias scratch or a later decision');
      prior=actual.memory;
    }
    optimized.reset();reference.reset();
    assert.deepStrictEqual(optimized.memory,reference.memory);
    const state=new Array(states).fill(0);
    const choices=[new Array(actions).fill(0)];
    assert.deepStrictEqual(optimized.infer(state,choices,0),
      originalInfer(reference,state,choices,0,true));
    assert.throws(()=>optimized.infer(state,[[NaN]],0),/Invalid RNAD infer inputs/);
  });
}

for(const {states,actions,schema} of [
  {states:32,actions:16,schema:RNAD_FEATURE_SCHEMA},
  {states:96,actions:48,schema:RNAD_THREE_PHASE_FEATURE_SCHEMA_V5}
]){
  test('opt-in shared policy prefix matches original scalar bits '+schema,()=>{
    const payload=fixture(states,actions,schema);
    const reference=new PersistentRNADInference(payload);
    const previous=process.env.RNAD_POLICY_HIDDEN_PREFIX_REUSE;
    let accelerated;
    try{
      process.env.RNAD_POLICY_HIDDEN_PREFIX_REUSE='1';
      accelerated=new PersistentRNADInference(payload);
    }finally{
      if(previous===undefined)delete process.env.RNAD_POLICY_HIDDEN_PREFIX_REUSE;
      else process.env.RNAD_POLICY_HIDDEN_PREFIX_REUSE=previous;
    }
    for(let step=0;step<12;step++){
      const seat=step%2,advance=step%4!==0;
      const state=Array.from({length:states},(_,i)=>((i*11+step)%37-18)/19);
      const candidates=Array.from({length:[1,3,60,8][step%4]},(_,k)=>
        Array.from({length:actions},(_,j)=>((k*13+j*7+step)%41-20)/23));
      const expected=originalInfer(reference,state,candidates,seat,advance);
      const actual=accelerated.infer(state,candidates,seat,{advance});
      assert.deepStrictEqual(actual,expected);
      assert.deepStrictEqual(accelerated.memory,reference.memory);
    }
    accelerated.reset();reference.reset();
    assert.deepStrictEqual(accelerated.memory,reference.memory);
  });
}
test('invalid prefix optimization flag fails closed',()=>{
  const previous=process.env.RNAD_POLICY_HIDDEN_PREFIX_REUSE;
  try{
    process.env.RNAD_POLICY_HIDDEN_PREFIX_REUSE='maybe';
    assert.throws(()=>new PersistentRNADInference(
      fixture(96,48,RNAD_THREE_PHASE_FEATURE_SCHEMA_V5)),
      /RNAD_POLICY_HIDDEN_PREFIX_REUSE must be 0 or 1/);
  }finally{
    if(previous===undefined)delete process.env.RNAD_POLICY_HIDDEN_PREFIX_REUSE;
    else process.env.RNAD_POLICY_HIDDEN_PREFIX_REUSE=previous;
  }
});
