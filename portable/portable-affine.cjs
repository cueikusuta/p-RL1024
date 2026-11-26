'use strict';
const fs=require('node:fs'),path=require('node:path');
const buffer=fs.readFileSync(path.join(__dirname,'affine.wasm'));
const wasm=new WebAssembly.Instance(new WebAssembly.Module(buffer)).exports;
const heap=new Float64Array(wasm.memory.buffer);
// Cache proven immutable transposed weights for the CPU training job lifetime
// using small bounded circular epochs. No generated data, platform toolchain,
// shared memory, or install. Scratch capacity: 512+384+384 values.
const x=0,b=512,z=896,start=1280;
let lookup=new WeakMap(),offset=start;
function compute(xs,ws,bs,result=null){
  if(!(xs instanceof Float64Array)||!(ws instanceof Float64Array)||!(bs instanceof Float64Array))
    throw Error('Bad portable affine input type');
  const m=xs.length,n=bs.length;
  if(m===0||m>512||n===0||n>384||ws.length!==m*n)throw Error('Bad portable affine rank');
  if(n%2){
    const out=result??bs.slice();if(result)out.set(bs);
    for(let k=0;k<m;k++)for(let i=0;i<n;i++)out[i]+=ws[k*n+i]*xs[k];
    return out;
  }
  let address=lookup.get(ws);
  if(address===undefined){
    if(offset+ws.length>heap.length){offset=start;lookup=new WeakMap();}
    address=offset;heap.set(ws,address);offset+=ws.length;
    lookup.set(ws,address);
  }
  heap.set(xs,x);heap.set(bs,b);
  wasm.dot(m,n,x*8,address*8,b*8,z*8);
  if(result){result.set(heap.subarray(z,z+n));return result;}
  return heap.slice(z,z+n);
}
function linear(xs,ws,bs){return compute(xs,ws,bs);}
function lengths(specs){
  if(specs.some(([a,n])=>!(a instanceof Float64Array)||a.length!==n))
    throw Error('Bad frozen draft input dimensions');
}
function scorePrepared(xs,w0,b0,w2,b2,w4,b4,h0,h2){
  compute(xs,w0,b0,h0);
  for(let j=0;j<64;j++)if(h0[j]<=0)h0[j]=0;
  compute(h0,w2,b2,h2);
  for(let j=0;j<32;j++)if(h2[j]<=0)h2[j]=0;
  let y=b4[0];for(let j=0;j<32;j++)y+=h2[j]*w4[j];
  return y;
}
function score(xs,w0,b0,w2,b2,w4,b4){
  lengths([[xs,148],[w0,148*64],[b0,64],[w2,64*32],[b2,32],[w4,32],[b4,1]]);
  return scorePrepared(xs,w0,b0,w2,b2,w4,b4,new Float64Array(64),new Float64Array(32));
}
function many(hidden,avg,sq,codes,meta,credit,w0,b0,w2,b2,w4,b4){
  const h=hidden.length,n=codes.length;
  if(h<288||h>4800||h%48||!(codes instanceof Uint32Array)||n<6||n>360000||n%6||
     !Number.isFinite(credit)||credit<0)throw Error('Invalid portable draft candidate header');
  lengths([[hidden,h],[avg,h],[sq,h],[meta,h/16],[w0,148*64],[b0,64],[w2,64*32],[b2,32],[w4,32],[b4,1]]);
  const out=new Float64Array(n/6),xs=new Float64Array(148),h0=new Float64Array(64),h2=new Float64Array(32);
  for(let i=0;i<out.length;i++){
    xs.fill(0,0,96);xs.fill(-Infinity,96,144);
    let spent=0,carry=0,mega=0;
    for(let j=0;j<6;j++){
      const q=codes[i*6+j];
      if(q>=h/48)throw Error('Invalid portable draft code');
      for(let k=0;k<j;k++)if(q===codes[i*6+k])throw Error('Duplicated portable draft code');
      spent+=meta[3*q];carry+=meta[3*q+1];mega+=meta[3*q+2];
      for(let k=0;k<48;k++){
        const u=q*48+k,y=hidden[u];xs[k]+=avg[u];xs[48+k]+=sq[u];
        if(Number.isNaN(y)||y>xs[96+k])xs[96+k]=y;
      }
    }
    if(spent>credit)throw Error('Portable draft exceeds budget');
    xs[144]=spent/100;xs[145]=(credit-spent)/100;xs[146]=carry/6;xs[147]=mega/6;
    out[i]=scorePrepared(xs,w0,b0,w2,b2,w4,b4,h0,h2);
  }
  return out;
}
module.exports={linear,score,many};
