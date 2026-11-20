// Exact arithmetic parity with the pre-optimization Deep Sets candidate scorer.
// This is a nonzero-weight regression fixture, not a model strength estimate.
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DeepSetsDraftPolicy,DRAFT_DEEPSETS_FEATURE_WIDTH}
  from '../scripts/rnad/draft-deepsets-infer.mjs';

const shapes={
  'phi.0.weight':[48,44],'phi.0.bias':[48],
  'phi.2.weight':[48,48],'phi.2.bias':[48],
  'rho.0.weight':[64,148],'rho.0.bias':[64],
  'rho.2.weight':[32,64],'rho.2.bias':[32],
  'rho.4.weight':[1,32],'rho.4.bias':[1],
};
function fixture(){
  const dir=mkdtempSync(join(tmpdir(),'rnad-draft-fastpath-'));
  const file=join(dir,'draft.json');
  const payload={
    schema:'rnad-draft-deepsets-weights-v1',
    architecture:'deepsets-soft',feature_width:DRAFT_DEEPSETS_FEATURE_WIDTH,
    source_sha:'a'.repeat(40),
    weights:Object.fromEntries(Object.entries(shapes).map(([key,shape])=>{
      const n=shape.reduce((a,b)=>a*b,1);
      return [key,{shape,values:Array.from({length:n},(_,i)=>
        ((i*173+key.length*31)%503-251)/4000)}];
    }))
  };
  writeFileSync(file,JSON.stringify(payload));
  return {dir,file};
}
function oldLinear(values,weight,bias,outputs){
  const inputs=values.length,out=new Float64Array(outputs);
  for(let j=0;j<outputs;j++){
    let sum=bias[j],offset=j*inputs;
    for(let k=0;k<inputs;k++)sum+=values[k]*weight[offset+k];
    out[j]=sum;
  }
  return out;
}
function oldRelu(v){
  for(let i=0;i<v.length;i++)v[i]=Math.max(0,v[i]);
  return v;
}
function oldMlp(x,layers,name,widths){
  let values=x;
  for(let n=0;n<widths.length;n++){
    const key=name+'.'+(n*2);
    values=oldLinear(values,layers[key+'.weight'],layers[key+'.bias'],widths[n]);
    if(n<widths.length-1)oldRelu(values);
  }
  return values;
}
function oldScore(encoding,layers){
  const {features,codes,prices,carries,megas,credits}=encoding;
  const hidden=features.map(f=>oldRelu(oldMlp(f,layers,'phi',[48,48])));
  const result=[];
  for(const code of codes){
    if(!Array.isArray(code)||code.length!==6||
       new Set(code).size!==6||code.some(i=>!Number.isSafeInteger(i)||
         i<0||i>=hidden.length))throw new TypeError('Malformed draft candidate indices');
    const avg=new Float64Array(48),second=new Float64Array(48);
    const high=Float64Array.from({length:48},()=>-Infinity);
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
    const x=[...avg,...second,...high,spent/100,
      (credits-spent)/100,carry/6,mega/6];
    const score=oldMlp(x,layers,'rho',[64,32,1])[0];
    if(!Number.isFinite(score))throw new Error('Nonfinite model draft score');
    result.push(score);
  }
  return result;
}
function encoding(n,codes){
  return {
    features:Array.from({length:n},(_,i)=>
      Array.from({length:44},(_,j)=>((i*29+j*11)%41-20)/27)),
    codes,credits:100,
    prices:Array.from({length:n},(_,i)=>i%5),
    carries:Array.from({length:n},(_,i)=>+(i%3===0)),
    megas:Array.from({length:n},(_,i)=>+(i%4===0))
  };
}
function combinations(n){
  const all=[];
  for(let a=0;a<n-5;a++)for(let b=a+1;b<n-4;b++)
    for(let c=b+1;c<n-3;c++)for(let d=c+1;d<n-2;d++)
      for(let e=d+1;e<n-1;e++)for(let f=e+1;f<n;f++)
        all.push([a,b,c,d,e,f]);
  return all;
}
test('nonzero frozen Deep Sets scores are bitwise-identical to original scalar rho',()=>{
  const {dir,file}=fixture();
  try{
    const model=new DeepSetsDraftPolicy(file);
    const subsets=combinations(11);
    const cases=[
      encoding(11,subsets),
      encoding(11,subsets.slice(0,40).map(row=>[...row].reverse())),
      encoding(11,[subsets.at(-1)]),
      encoding(11,subsets.slice(16,100))
    ];
    for(const input of cases){
      const expected=oldScore(input,model.layers);
      const actual=model.scoreEncoded(input);
      assert.deepStrictEqual(actual,expected);
      assert.equal(actual.length,input.codes.length);
    }
    const distinct=encoding(11,[subsets[0]]);
    distinct.codes=[[0,0,1,2,3,4]];
    assert.throws(()=>model.scoreEncoded(distinct),/Malformed draft candidate indices/);
    distinct.codes=[[0,1,2,3,4,11]];
    assert.throws(()=>model.scoreEncoded(distinct),/Malformed draft candidate indices/);
    distinct.codes=[[0,1,2,3,4,5]];
    distinct.credits=0;
    assert.throws(()=>model.scoreEncoded(distinct),/Over-budget legal draft pick/);
  }finally{rmSync(dir,{recursive:true,force:true});}
});

test('rho fast path retains IEEE-754 edge behavior and rejects nonfinite results',()=>{
  const {dir,file}=fixture();
  try{
    const input=encoding(7,[[0,1,2,3,4,5],[1,2,3,4,5,6]]);
    const variants=[
      layers=>{ layers['rho.4.weight'].fill(-0);layers['rho.4.bias'][0]=-0; },
      layers=>{ layers['rho.0.weight'].fill(Number.MIN_VALUE);
        layers['rho.0.bias'].fill(-Number.MIN_VALUE); },
      layers=>{ layers['rho.2.weight'].fill(-Number.MIN_VALUE);
        layers['rho.2.bias'].fill(Number.MIN_VALUE); },
      layers=>{ layers['rho.4.weight'].fill(Number.MAX_VALUE);
        layers['rho.4.bias'][0]=Number.MAX_VALUE; }
    ];
    for(const mutate of variants){
      const model=new DeepSetsDraftPolicy(file);
      mutate(model.layers);
      const outcome=fn=>{
        try{return {scores:fn()};}
        catch(error){return {error:error.message};}
      };
      const reference=outcome(()=>oldScore(input,model.layers));
      const actual=outcome(()=>model.scoreEncoded(input));
      assert.equal(actual.error,reference.error);
      if(reference.scores){
        assert.equal(actual.scores.length,reference.scores.length);
        for(let i=0;i<actual.scores.length;i++){
          // Object.is distinguishes +0 from -0, unlike numeric equality.
          assert.ok(Object.is(actual.scores[i],reference.scores[i]),
            'IEEE-754 score bit pattern drift');
        }
      }
    }
    for(const nonfinite of [NaN,Infinity,-Infinity]){
      const model=new DeepSetsDraftPolicy(file);
      model.layers['rho.0.weight'][0]=nonfinite;
      const reference=()=>oldScore(input,model.layers);
      const optimized=()=>model.scoreEncoded(input);
      const referenceOutcome=(()=>{try{return reference();}catch(e){return e.message;}})();
      const optimizedOutcome=(()=>{try{return optimized();}catch(e){return e.message;}})();
      assert.deepStrictEqual(optimizedOutcome,referenceOutcome);
    }
    for(const invalid of [NaN,Infinity,-Infinity]){
      const model=new DeepSetsDraftPolicy(file);
      const bad=encoding(7,[[0,1,2,3,4,5]]);
      bad.features[0][0]=invalid;
      assert.throws(()=>model.scoreEncoded(bad),/Malformed encoded own-set features/);
    }
  }finally{rmSync(dir,{recursive:true,force:true});}
});

test('opt-in rho four-output blocking preserves scalar bits for all legal picks',()=>{
  const {dir,file}=fixture();
  const before=process.env.RNAD_DRAFT_RHO_ROW_BLOCK4;
  try{
    process.env.RNAD_DRAFT_RHO_ROW_BLOCK4='1';
    const model=new DeepSetsDraftPolicy(file);
    const subsets=combinations(11);
    for(const input of [
      encoding(11,subsets),
      encoding(11,subsets.slice(0,70).map(v=>[...v].reverse())),
      encoding(11,[subsets.at(-1)])
    ]){
      const expected=oldScore(input,model.layers);
      const actual=model.scoreEncoded(input);
      assert.equal(actual.length,expected.length);
      for(let i=0;i<actual.length;i++)
        assert.ok(Object.is(actual[i],expected[i]),
          'Rho blocked sum changed IEEE-754 scalar output');
    }
    const nearZero=encoding(7,[[0,1,2,3,4,5],[1,2,3,4,5,6]]);
    model.layers['rho.0.weight'].fill(Number.MIN_VALUE);
    model.layers['rho.0.bias'].fill(-Number.MIN_VALUE);
    assert.deepStrictEqual(model.scoreEncoded(nearZero),
      oldScore(nearZero,model.layers));
  }finally{
    if(before===undefined)delete process.env.RNAD_DRAFT_RHO_ROW_BLOCK4;
    else process.env.RNAD_DRAFT_RHO_ROW_BLOCK4=before;
    rmSync(dir,{recursive:true,force:true});
  }
});
test('invalid draft rho row blocking flag fails closed',()=>{
  const {dir,file}=fixture();
  const before=process.env.RNAD_DRAFT_RHO_ROW_BLOCK4;
  try{
    process.env.RNAD_DRAFT_RHO_ROW_BLOCK4='yes';
    assert.throws(()=>new DeepSetsDraftPolicy(file),
      /RNAD_DRAFT_RHO_ROW_BLOCK4 must be 0 or 1/);
  }finally{
    if(before===undefined)delete process.env.RNAD_DRAFT_RHO_ROW_BLOCK4;
    else process.env.RNAD_DRAFT_RHO_ROW_BLOCK4=before;
    rmSync(dir,{recursive:true,force:true});
  }
});
