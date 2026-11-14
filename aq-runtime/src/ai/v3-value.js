// I877 D: bounded deterministic CPU Value baselines (logistic + tiny MLP).
// All inference inputs are the versioned numeric public observation ONLY.
import {V3_FEATURE_NAMES,V3_FEATURE_SCHEMA,validateV3Features}
  from './v3-observation.js';
export const V3_VALUE_MODEL_SCHEMA='i877-logistic-value-v1';
export const V3_MLP_MODEL_SCHEMA='i877-mlp-value-v1';
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):
  Math.exp(x)/(1+Math.exp(x));
const validWeights=(xs,n)=>Array.isArray(xs)&&xs.length===n&&
  xs.every(w=>finite(w)&&Math.abs(w)<=100);
function validateSamples(samples,epochs,learningRate,l2) {
  if(!Array.isArray(samples)||samples.length<2||samples.length>1000000||
     !Number.isInteger(epochs)||epochs<1||epochs>256||
     !finite(learningRate)||learningRate<=0||learningRate>1||
     !finite(l2)||l2<0||l2>0.5)
    throw new TypeError('Invalid V3 training data/configuration');
  for(const sample of samples){
    validateV3Features(sample?.features);
    if(sample.label!==0&&sample.label!==1)
      throw new TypeError('V3 labels must be completed Match outcomes');
  }
}
export function validateV3ValueModel(model) {
  if(!model||model.featureSchema!==V3_FEATURE_SCHEMA)
    throw new TypeError('V3 Value model/schema is invalid');
  if(model.schema===V3_VALUE_MODEL_SCHEMA&&
     validWeights(model.weights,V3_FEATURE_NAMES.length)&&
     finite(model.bias)&&Math.abs(model.bias)<=100)return model;
  if(model.schema===V3_MLP_MODEL_SCHEMA&&
     Array.isArray(model.hiddenWeights)&&
     model.hiddenWeights.length>=2&&model.hiddenWeights.length<=16&&
     model.hiddenWeights.every(w=>validWeights(w,V3_FEATURE_NAMES.length))&&
     validWeights(model.hiddenBias,model.hiddenWeights.length)&&
     validWeights(model.outWeights,model.hiddenWeights.length)&&
     finite(model.outBias)&&Math.abs(model.outBias)<=100)return model;
  throw new TypeError('V3 Value model/schema is invalid');
}
export function predictV3Value(model,features) {
  validateV3ValueModel(model);
  validateV3Features(features);
  let z;
  if(model.schema===V3_VALUE_MODEL_SCHEMA){
    z=model.bias;
    for(let i=0;i<features.length;i++)z+=features[i]*model.weights[i];
  }else{
    const h=model.hiddenWeights.map((row,k)=>Math.tanh(
      model.hiddenBias[k]+row.reduce((sum,w,i)=>sum+w*features[i],0)));
    z=model.outBias+h.reduce((sum,v,k)=>sum+v*model.outWeights[k],0);
  }
  if(!finite(z))throw new TypeError('V3 Value inference overflow');
  return Math.max(1e-7,Math.min(1-1e-7,sigmoid(z)));
}
export function fitV3ValueModel(samples,{epochs=40,learningRate=0.15,
  l2=0.002}={}) {
  validateSamples(samples,epochs,learningRate,l2);
  const weights=Array(V3_FEATURE_NAMES.length).fill(0);
  let bias=0;
  for(let epoch=0;epoch<epochs;epoch++) {
    const gradient=weights.map(()=>0);
    let biasGradient=0;
    for(const {features,label} of samples) {
      const p=sigmoid(bias+weights.reduce((sum,w,i)=>
        sum+w*features[i],0));
      const error=p-label;
      biasGradient+=error;
      for(let i=0;i<weights.length;i++)
        gradient[i]+=error*features[i];
    }
    bias-=learningRate*biasGradient/samples.length;
    for(let i=0;i<weights.length;i++)
      weights[i]-=learningRate*(gradient[i]/samples.length+l2*weights[i]);
  }
  return Object.freeze({schema:V3_VALUE_MODEL_SCHEMA,
    featureSchema:V3_FEATURE_SCHEMA,weights:Object.freeze(weights),bias});
}
// Full-batch backprop avoids nondeterministic shuffle and external ML runtime.
// This network is deliberately tiny (2..16 tanh units) and bounded.
export function fitV3MLPValueModel(samples,{epochs=80,learningRate=0.12,
  l2=0.002,hiddenUnits=8}={}) {
  validateSamples(samples,epochs,learningRate,l2);
  if(!Number.isInteger(hiddenUnits)||hiddenUnits<2||hiddenUnits>16)
    throw new TypeError('Invalid V3 MLP hidden size');
  const n=V3_FEATURE_NAMES.length;
  const hiddenWeights=Array.from({length:hiddenUnits},(_,k)=>
    Array.from({length:n},(_,i)=>0.12*Math.sin((k+1)*(i+1))));
  const hiddenBias=Array(hiddenUnits).fill(0);
  const outWeights=Array.from({length:hiddenUnits},(_,k)=>
    0.08*Math.cos(k+1));
  let outBias=0;
  for(let epoch=0;epoch<epochs;epoch++) {
    const gW=hiddenWeights.map(w=>w.map(()=>0));
    const gB=Array(hiddenUnits).fill(0);
    const gO=Array(hiddenUnits).fill(0);
    let gOutBias=0;
    for(const {features,label} of samples) {
      const h=hiddenWeights.map((row,k)=>Math.tanh(
        hiddenBias[k]+row.reduce((sum,w,i)=>sum+w*features[i],0)));
      const prediction=sigmoid(outBias+
        h.reduce((sum,v,k)=>sum+v*outWeights[k],0));
      const error=prediction-label;
      gOutBias+=error;
      for(let k=0;k<hiddenUnits;k++){
        gO[k]+=error*h[k];
        const gradient=error*outWeights[k]*(1-h[k]*h[k]);
        gB[k]+=gradient;
        for(let i=0;i<n;i++)gW[k][i]+=gradient*features[i];
      }
    }
    const count=samples.length;
    outBias-=learningRate*gOutBias/count;
    for(let k=0;k<hiddenUnits;k++){
      outWeights[k]-=learningRate*(gO[k]/count+l2*outWeights[k]);
      hiddenBias[k]-=learningRate*gB[k]/count;
      for(let i=0;i<n;i++)
        hiddenWeights[k][i]-=learningRate*(gW[k][i]/count+
          l2*hiddenWeights[k][i]);
    }
  }
  return Object.freeze({schema:V3_MLP_MODEL_SCHEMA,
    featureSchema:V3_FEATURE_SCHEMA,
    hiddenWeights:Object.freeze(hiddenWeights.map(w=>Object.freeze(w))),
    hiddenBias:Object.freeze(hiddenBias),
    outWeights:Object.freeze(outWeights),outBias});
}
export function scoreV3ValueModel(model,samples) {
  validateV3ValueModel(model);
  if(!Array.isArray(samples)||samples.length===0)
    throw new TypeError('Expected held-out V3 observations');
  let brier=0,logLoss=0;
  for(const row of samples) {
    if(row.label!==0&&row.label!==1)throw new TypeError('Invalid V3 label');
    const p=predictV3Value(model,row.features);
    brier+=(p-row.label)**2;
    logLoss-=row.label*Math.log(p)+(1-row.label)*Math.log(1-p);
  }
  return {examples:samples.length,brier:brier/samples.length,
    logLoss:logLoss/samples.length};
}
