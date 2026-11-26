import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { availableParallelism } from 'node:os';
import { mkdirSync, writeFileSync } from 'node:fs';
import { gzipSync, gunzipSync } from 'node:zlib';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';

const sha=b=>createHash('sha256').update(b).digest('hex');
const clock=()=>Math.round(performance.now());
if (!isMainThread) {
  const n=128, turns=1536;
  let s=workerData.seed;
  const next=()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)/2**32)-0.5;
  const mat=Float64Array.from({length:n*n},next);
  let a=Float64Array.from({length:n},next), b=new Float64Array(n);
  const t=clock();
  for(let r=0;r<turns;r++){
    for(let i=0;i<n;i++){
      let c=0;
      for(let j=0;j<n;j++)c+=mat[i*n+j]*a[j];
      b[i]=c/(1+Math.abs(c))+next()*0.05;
    }
    [a,b]=[b,a];
  }
  parentPort.postMessage({cpuMs:clock()-t,hash:sha(JSON.stringify(Array.from(a)))});
} else {
  const shard=Number(process.env.PROBE_SHARD);
  if (!Number.isInteger(shard)||shard<0||shard>7) throw Error('Invalid synthetic shard');
  const t=clock();
  const lanes=await Promise.all([0,1].map(i=>new Promise((ok,bad)=>{
    const w=new Worker(new URL(import.meta.url),{workerData:{seed:12345+2*shard+i}});
    w.once('message',ok);w.once('error',bad);
    w.once('exit',n=>{if(n!==0)bad(Error('Worker failed '+n))});
  })));
  const computeMs=clock()-t;
  // Fully synthetic, deterministic bytes. No model, actual matches, or user data.
  const raw=Buffer.alloc(4*1024*1024);
  let x=(32332+shard)>>>0;
  for(let i=0;i<raw.length;i+=4){
    x=(x^(x<<13))>>>0;x=(x^(x>>>17))>>>0;x=(x^(x<<5))>>>0;
    raw.writeUInt32LE(x,i);
  }
  const packing=clock();
  const blob=gzipSync(raw,{level:1,mtime:0});
  const decoded=gunzipSync(blob);
  if(!raw.equals(decoded)||lanes.some(x=>!/^[0-9a-f]{64}$/.test(x.hash)))throw Error('Synthetic readback failed');
  mkdirSync('probe-output');
  writeFileSync('probe-output/synthetic.bin.gz',blob,{flag:'wx'});
  const report={seedShard:shard,kind:'synthetic-infrastructure-probe',computeMs,lanes,
    packingAndReadbackMs:clock()-packing,archiveBytes:blob.length,rawBytes:raw.length,
    packedSha256:sha(blob),rawSha256:sha(raw),node:process.version,
    parallelism:availableParallelism(),completedAt:new Date().toISOString()};
  const line=JSON.stringify(report);
  writeFileSync('probe-output/report.json',line+'\n',{flag:'wx'});
  console.log(line);
}
