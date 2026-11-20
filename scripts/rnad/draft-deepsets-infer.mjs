// Optional R-NaD draft-only research inference; accepts ONLY seat-projected
// self.carry/self.fresh and authoritative six-Pokemon legal actions.
// This is not an opponent-info oracle, R-NaD gradient update, or Champion.
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {analyzeSetCapabilities,CAPABILITY_IDS} from '../../src/ai/set-capabilities.js';
import battleData from '../../data/catalog/battle-data.json' with {type:'json'};

export const DRAFT_DEEPSETS_ACTOR_SCHEMA='rnad-draft-deepsets-inference-v1';
export const DRAFT_DEEPSETS_FEATURE_WIDTH=44;
const TYPES=['normal','fire','water','electric','grass','ice','fighting',
  'poison','ground','flying','psychic','bug','rock','ghost','dragon',
  'dark','steel','fairy'];
const STATS=['hp','atk','def','spa','spd','spe'];
const FEATURES=TYPES.length+CAPABILITY_IDS.length+STATS.length+3;
const expected={
  'phi.0.weight':[48,FEATURES],'phi.0.bias':[48],
  'phi.2.weight':[48,48],'phi.2.bias':[48],
  'rho.0.weight':[64,148],'rho.0.bias':[64],
  'rho.2.weight':[32,64],'rho.2.bias':[32],
  'rho.4.weight':[1,32],'rho.4.bias':[1]
};
if(FEATURES!==DRAFT_DEEPSETS_FEATURE_WIDTH||
   CAPABILITY_IDS.join('|')!==[
     'attack','setup','recovery','teamRecovery','hazard',
     'hazardRemoval','phazing','yawn','pivot','screen',
     'batonPass','trickRoom','tailwind','statusControl',
     'weatherStart','terrainStart','priorityAttack'].join('|'))
  throw new Error('Draft feature contract no longer matches frozen teacher');

function parseModel(path,expectedSha){
  if(typeof path!=='string'||!path)throw new TypeError('Missing draft model path');
  const buf=readFileSync(path);
  const sha=createHash('sha256').update(buf).digest('hex');
  if(expectedSha!==undefined&&(typeof expectedSha!=='string'||
    expectedSha!==sha))
    throw new Error('Untrusted frozen Deep Sets model SHA256');
  const raw=JSON.parse(buf.toString('utf8'));
  if(raw?.schema!=='rnad-draft-deepsets-weights-v1'||
    raw.architecture!=='deepsets-soft'||raw.feature_width!==FEATURES||
    !/^[a-f0-9]{40}$/.test(raw.source_sha??'')||
    !raw.weights||Object.keys(raw.weights).length!==Object.keys(expected).length)
    throw new TypeError('Unsupported/invalid versioned Deep Sets model');
  const layers={};
  for(const [key,shape] of Object.entries(expected)){
    const tensor=raw.weights[key];
    const size=shape.reduce((a,b)=>a*b,1);
    if(!tensor||JSON.stringify(tensor.shape)!==JSON.stringify(shape)||
      !Array.isArray(tensor.values)||tensor.values.length!==size||
      tensor.values.some(x=>typeof x!=='number'||!Number.isFinite(x)))
      throw new TypeError('Invalid Deep Sets tensor '+key);
    layers[key]=Float64Array.from(tensor.values);
  }
  return {layers,sha,source:raw.source_sha};
}
const cap=x=>Math.min(1,x);
export function encodeOwnDraftPool(observation,legalActions){
  if(observation?.phase!=='draft'||!observation.self||
    !Array.isArray(legalActions)||!legalActions.length||
    !Number.isSafeInteger(observation.self.credits)||
    observation.self.credits<0)
    throw new TypeError('Expected seat-projected authority draft');
  const carry=observation.self.carry??[];
  const fresh=observation.self.fresh??[];
  if(!Array.isArray(carry)||!Array.isArray(fresh))
    throw new TypeError('Malformed own draft offers');
  const pool=[...carry,...fresh],ids=pool.map(p=>p.instanceId);
  if(pool.length<6||pool.length>100||new Set(ids).size!==ids.length||
    ids.some(id=>id===undefined||id===null))
    throw new TypeError('Invalid distinct self-only draft pool');
  const isCarry=new Set(carry.map(p=>p.instanceId));
  const byId=new Map(ids.map((id,i)=>[id,i]));
  const prices=pool.map(p=>isCarry.has(p.instanceId)?0:p.cost);
  if(prices.some(p=>!Number.isSafeInteger(p)||p<0))
    throw new TypeError('Invalid self draft credit price');
  const features=pool.map((p,i)=>{
    const stats=p.baseStats??p.stats;
    if(!stats||STATS.some(k=>!Number.isFinite(stats[k])||stats[k]<0)||
      !Array.isArray(p.types)||p.types.some(t=>typeof t!=='string'))
      throw new TypeError('Missing own-set stats/types');
    const types=new Set(p.types.map(t=>t.toLowerCase()));
    if([...types].some(t=>!TYPES.includes(t)))
      throw new TypeError('Unrecognized own-set type');
    const roles=new Set(analyzeSetCapabilities(p,battleData).tags);
    const f=[
      ...STATS.map(k=>cap(stats[k]/600)),
      ...TYPES.map(t=>Number(types.has(t))),
      ...CAPABILITY_IDS.map(t=>Number(roles.has(t))),
      Number(Boolean(p.mega)),cap(prices[i]/100),Number(isCarry.has(p.instanceId))
    ];
    if(f.length!==FEATURES||f.some(x=>!Number.isFinite(x)))
      throw new Error('Invalid draft model features');
    return f;
  });
  const codes=legalActions.map(action=>{
    if(!Array.isArray(action)||action.length!==6||
      new Set(action).size!==6||action.some(id=>!byId.has(id)))
      throw new TypeError('Malformed authoritative six-Pokemon draft action');
    const code=action.map(id=>byId.get(id));
    if(code.reduce((n,i)=>n+prices[i],0)>observation.self.credits)
      throw new TypeError('Nonlegal over-budget draft action');
    return code;
  });
  return {features,codes,prices,credits:observation.self.credits,
    carries:pool.map(p=>Number(isCarry.has(p.instanceId))),
    megas:pool.map(p=>Number(Boolean(p.mega)))};
}
function linear(values,weight,bias,outputs){
  const inputs=values.length, out=new Float64Array(outputs);
  for(let j=0;j<outputs;j++){
    let sum=bias[j],offset=j*inputs;
    for(let k=0;k<inputs;k++)sum+=values[k]*weight[offset+k];
    out[j]=sum;
  }
  return out;
}
function linearInto(values,weight,bias,out){
  // The frozen rho widths are 148/64/32 (all multiples of four).
  // Unroll the innermost dot product without re-associating additions:
  // preserve EXACT left-to-right IEEE-754 order, including bias addition.
  // This is a CPU hot path called once per legal six-Pokemon candidate.
  const inputs=values.length;
  if((inputs&3)===0){
    for(let j=0;j<out.length;j++){
      let sum=bias[j],offset=j*inputs;
      for(let k=0;k<inputs;k+=4){
        sum+=values[k]*weight[offset+k];
        sum+=values[k+1]*weight[offset+k+1];
        sum+=values[k+2]*weight[offset+k+2];
        sum+=values[k+3]*weight[offset+k+3];
      }
      out[j]=sum;
    }
  }else{
    // Retain the original scalar behavior for any future odd-width caller.
    for(let j=0;j<out.length;j++){
      let sum=bias[j],offset=j*inputs;
      for(let k=0;k<inputs;k++)sum+=values[k]*weight[offset+k];
      out[j]=sum;
    }
  }
  return out;
}
function relu(v){
  for(let i=0;i<v.length;i++)v[i]=Math.max(0,v[i]);
  return v;
}
function mlp(x,layers,name,widths){
  let values=x;
  for(let n=0;n<widths.length;n++){
    const key=name+'.'+(n*2);
    values=linear(values,layers[key+'.weight'],layers[key+'.bias'],widths[n]);
    if(n<widths.length-1)relu(values);
  }
  return values;
}
// Score-rank blend preserves the legal support and tempers a noisy
// supervised actor using the frozen v3 policy. Never a policy gradient update.
function centeredUnitVariance(scores){
  if(!Array.isArray(scores)||scores.length===0||
     scores.some(x=>!Number.isFinite(x)))
    throw new TypeError('Missing finite draft model scores');
  const mean=scores.reduce((a,b)=>a+b,0)/scores.length;
  const sd=Math.sqrt(scores.reduce((a,x)=>a+(x-mean)**2,0)/scores.length);
  if(!Number.isFinite(sd))throw new TypeError('Invalid draft score variance');
  return scores.map(x=>(x-mean)/Math.max(sd,1e-6));
}
export class BlendedDeepSetsDraftPolicy {
  constructor(deep,{deepShare=.5}={}){
    if(!(deep instanceof DeepSetsDraftPolicy)||
       ![.25,.5,.75].includes(deepShare))
      throw new TypeError('Invalid bounded frozen Deep Sets/v3 blend');
    this.deep=deep;this.deepShare=deepShare;
    this.modelSha256=deep.modelSha256;
    this.trainingSourceSha=deep.trainingSourceSha;
    this.schema=DRAFT_DEEPSETS_ACTOR_SCHEMA;
  }
  choose(observation,legalActions,baselineLogits){
    const x=this.deep.scoreEncoded(
      encodeOwnDraftPool(observation,legalActions));
    if(!Array.isArray(baselineLogits)||baselineLogits.length!==x.length)
      throw new TypeError('Missing frozen seat-v3 reference logits');
    const p=centeredUnitVariance(x),b=centeredUnitVariance(baselineLogits);
    const blended=p.map((s,i)=>this.deepShare*s+(1-this.deepShare)*b[i]);
    let index=0;
    for(let i=1;i<blended.length;i++)
      if(blended[i]>blended[index])index=i;
    const max=blended[index];
    const denom=blended.reduce((sum,v)=>sum+Math.exp(v-max),0);
    return {index,logProbability:blended[index]-max-Math.log(denom),
      score:blended[index],candidateCount:blended.length};
  }
}
// Experimental credit-utilization prior motivated by closed-loop logs.
// Uses ONLY self-visible candidate prices; keeps exactly the authority-legal
// support. Credit bias is a hypothesis, not a proven winning policy.
export class CreditAwareDeepSetsDraftPolicy {
  constructor(deep,{costWeight=.25}={}){
    if(!(deep instanceof DeepSetsDraftPolicy)||
       ![.25,.5].includes(costWeight))
      throw new TypeError('Invalid bounded draft credit calibration');
    this.deep=deep;this.costWeight=costWeight;
    this.modelSha256=deep.modelSha256;
    this.trainingSourceSha=deep.trainingSourceSha;
    this.schema=DRAFT_DEEPSETS_ACTOR_SCHEMA;
  }
  choose(observation,legalActions){
    const encoding=encodeOwnDraftPool(observation,legalActions);
    const deepScores=centeredUnitVariance(this.deep.scoreEncoded(encoding));
    const costs=encoding.codes.map(a=>
      a.reduce((s,i)=>s+encoding.prices[i],0));
    const relativeCosts=centeredUnitVariance(costs);
    const scores=deepScores.map((d,i)=>d+this.costWeight*relativeCosts[i]);
    let index=0;
    for(let i=1;i<scores.length;i++)
      if(scores[i]>scores[index])index=i;
    const max=scores[index];
    const sum=scores.reduce((s,x)=>s+Math.exp(x-max),0);
    return {index,logProbability:scores[index]-max-Math.log(sum),
      score:scores[index],candidateCount:scores.length};
  }
}
export class DeepSetsDraftPolicy{
  constructor(path,{sha256}={}){
    const m=parseModel(path,sha256);
    this.modelSha256=m.sha;
    this.trainingSourceSha=m.source;
    this.layers=m.layers;
    this.schema=DRAFT_DEEPSETS_ACTOR_SCHEMA;
  }
  // Pure input contract permits a Python/PyTorch parity check on fixed examples.
  scoreEncoded(encoding){
    if(!encoding||!Array.isArray(encoding.features)||
      !Array.isArray(encoding.codes)||!encoding.codes.length)
      throw new TypeError('Missing draft feature pool');
    const {features,codes,prices,carries,megas,credits}=encoding;
    if(!Number.isSafeInteger(credits)||!Array.isArray(prices)||
      features.length!==prices.length||features.length!==carries.length||
      features.length!==megas.length)
      throw new TypeError('Invalid encoded draft offer');
    const hidden=features.map(f=>{
      if(f.length!==FEATURES||f.some(x=>!Number.isFinite(x)))
        throw new TypeError('Malformed encoded own-set features');
      return relu(mlp(f,this.layers,'phi',[48,48]));
    });
    // A legal pool may contain thousands of six-Pokemon combinations.
    // Reuse rho aggregation/hidden buffers across candidates. No buffers
    // escape to the caller; logits and public authority checks are unchanged.
    const result=[];
    const avg=new Float64Array(48),second=new Float64Array(48);
    const high=new Float64Array(48),x=new Float64Array(148);
    const h0=new Float64Array(64),h2=new Float64Array(32);
    const scalar=new Float64Array(1);
    const rho0W=this.layers['rho.0.weight'],rho0B=this.layers['rho.0.bias'];
    const rho2W=this.layers['rho.2.weight'],rho2B=this.layers['rho.2.bias'];
    const rho4W=this.layers['rho.4.weight'],rho4B=this.layers['rho.4.bias'];
    for(const code of codes){
      if(!Array.isArray(code)||code.length!==6)
        throw new TypeError('Malformed draft candidate indices');
      // Fixed-length duplicate detection avoids a Set allocation for
      // every candidate while still rejecting identical numeric indices.
      for(let at=0;at<6;at++){
        const i=code[at];
        if(!Number.isSafeInteger(i)||i<0||i>=hidden.length||
          code.indexOf(i)!==at)
          throw new TypeError('Malformed draft candidate indices');
      }
      avg.fill(0);second.fill(0);high.fill(-Infinity);
      let spent=0,carry=0,mega=0;
      for(const i of code){
        const v=hidden[i];spent+=prices[i];carry+=carries[i];mega+=megas[i];
        for(let k=0;k<48;k++){
          avg[k]+=v[k]/6;
          second[k]+=v[k]*v[k]/6;
          high[k]=Math.max(high[k],v[k]);
        }
      }
      if(spent>credits)throw new Error('Over-budget legal draft pick');
      x.set(avg,0);x.set(second,48);x.set(high,96);
      x[144]=spent/100;x[145]=(credits-spent)/100;
      x[146]=carry/6;x[147]=mega/6;
      relu(linearInto(x,rho0W,rho0B,h0));
      relu(linearInto(h0,rho2W,rho2B,h2));
      const score=linearInto(h2,rho4W,rho4B,scalar)[0];
      if(!Number.isFinite(score))throw new Error('Nonfinite model draft score');
      result.push(score);
    }
    return result;
  }
  choose(observation,legalActions){
    const encoded=encodeOwnDraftPool(observation,legalActions);
    const logits=this.scoreEncoded(encoded);
    let index=0;
    for(let i=1;i<logits.length;i++)if(logits[i]>logits[index])index=i;
    const max=logits[index];
    const scale=logits.reduce((a,v)=>a+Math.exp(v-max),0);
    return {index,logProbability:logits[index]-max-Math.log(scale),
      score:logits[index],candidateCount:logits.length};
  }
}
