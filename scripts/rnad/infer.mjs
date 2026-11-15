// RNAD-001 persistent synchronous JS CPU inference of the GRU128
// PyTorch-exported weights. This lives inside Node throughout a Match;
// no Python subprocess per action and no modification to original AI paths.
import {readFileSync} from 'node:fs';
import {encodeRnadDecision} from '../../src/ml/rnad-features.js';
export const RNAD_MODEL_SCHEMA='rnad-small-gru-v1';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const sigmoid=x=>1/(1+Math.exp(-x));
const relu=x=>x>0?x:0;
function weight(params,name,shape){
  const x=params[name];
  if(!x||!Array.isArray(x.shape)||JSON.stringify(x.shape)!==JSON.stringify(shape)||
    !Array.isArray(x.values)||x.values.length!==shape.reduce((a,b)=>a*b,1)||
    x.values.some(y=>!finite(y)))
    throw new TypeError('Malformed RNAD weight tensor: '+name);
  return x.values;
}
export function softmaxRnad(xs){
  if(!Array.isArray(xs)||xs.length<1||xs.some(x=>!finite(x)))
    throw new TypeError('Invalid RNAD logits');
  const high=Math.max(...xs);
  const weights=xs.map(x=>Math.exp(x-high));
  const z=weights.reduce((a,b)=>a+b,0);
  return weights.map(x=>x/z);
}
export class PersistentRNADInference{
  constructor(serialized){
    const payload=typeof serialized==='string'?JSON.parse(readFileSync(serialized,'utf8')):serialized;
    if(!payload||payload.schema!==RNAD_MODEL_SCHEMA||
      payload.hidden!==128||payload.state_features!==32||
      payload.action_features!==16||!payload.weights)
      throw new TypeError('Wrong RNAD model/schema dimensions');
    const p=payload.weights,H=128;
    this.params={};
    for(const [name,shape] of Object.entries({
      'state_proj.weight':[H,32],'state_proj.bias':[H],
      'gru.weight_ih':[3*H,H],'gru.weight_hh':[3*H,H],
      'gru.bias_ih':[3*H],'gru.bias_hh':[3*H],
      'policy_hidden.weight':[H,H+16],'policy_hidden.bias':[H],
      'policy_out.weight':[1,H],'policy_out.bias':[1],
      'value_hidden.weight':[H,H],'value_hidden.bias':[H],
      'value_out.weight':[1,H],'value_out.bias':[1]}))
      this.params[name]=weight(p,name,shape);
    this.memory=[new Array(H).fill(0),new Array(H).fill(0)];
  }
  reset(){this.memory=[new Array(128).fill(0),new Array(128).fill(0)];}
  linear(prefix,x,out){
    const w=this.params[prefix+'.weight'],b=this.params[prefix+'.bias'];
    const result=new Array(out);
    for(let i=0;i<out;i++){
      let sum=b[i],offset=i*x.length;
      for(let j=0;j<x.length;j++)sum+=w[offset+j]*x[j];
      result[i]=sum;
    }
    return result;
  }
  infer(state,actions,seat,{advance=true}={}){
    if(![0,1].includes(seat)||!Array.isArray(state)||state.length!==32||
      state.some(x=>!finite(x))||!Array.isArray(actions)||actions.length<1||
      actions.some(a=>!Array.isArray(a)||a.length!==16||a.some(x=>!finite(x))))
      throw new TypeError('Invalid RNAD infer inputs');
    const H=128,previous=this.memory[seat];
    const input=this.linear('state_proj',state,H).map(relu);
    // Explicit GRU gates below, no generic affine helper.
    // GRU has two weight matrices; unlike normal Linear it uses
    // separate input/hidden affine transforms and biases.
    // The generic linear() above is not used for GRU.
    const wI=this.params['gru.weight_ih'],wH=this.params['gru.weight_hh'];
    const bI=this.params['gru.bias_ih'],bH=this.params['gru.bias_hh'];
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
      const hidden=this.linear('policy_hidden',[...memory,...a],H).map(relu);
      return this.linear('policy_out',hidden,1)[0];
    });
    const value=Math.tanh(this.linear('value_out',
      this.linear('value_hidden',memory,H).map(relu),1)[0]);
    if(!finite(value)||logits.some(x=>!finite(x))||
      memory.some(x=>!finite(x)))
      throw new Error('Nonfinite RNAD network result');
    if(advance)this.memory[seat]=memory;
    return {logits,value,memory};
  }
  inferProjected(observation,legalActions,seat,opts){
    const f=encodeRnadDecision(observation,legalActions);
    return this.infer(f.state,f.actions,seat,opts);
  }
}
export class RNADMixedPolicy {
  constructor({model,magnet,epsilon=0.02}){
    if(!(model instanceof PersistentRNADInference)||
      !(magnet instanceof PersistentRNADInference)||
      !finite(epsilon)||epsilon<0||epsilon>.5)
      throw new TypeError('Invalid RNAD behavior/magnet');
    this.model=model;this.magnet=magnet;this.epsilon=epsilon;
  }
  reset(){this.model.reset();this.magnet.reset();}
  choose({observation,legalActions,seat,rng}){
    if(!rng||typeof rng.integer!=='function')
      throw new TypeError('RNAD policy requires deterministic seat RNG');
    const current=this.model.inferProjected(observation,legalActions,seat);
    const anchor=this.magnet.inferProjected(observation,legalActions,seat);
    const pi=softmaxRnad(current.logits),reference=softmaxRnad(anchor.logits);
    const probs=pi.map(x=>(1-this.epsilon)*x+this.epsilon/pi.length);
    const sample=rng.integer(0x1000000)/0x1000000;
    let cursor=0,idx=probs.length-1;
    for(let i=0;i<probs.length;i++){cursor+=probs[i];if(sample<cursor){idx=i;break;}}
    return {action:legalActions[idx],
      behaviorLogProbability:Math.log(probs[idx]),
      referenceLogProbability:Math.log(reference[idx]),
      predictedValue:current.value};
  }
}
