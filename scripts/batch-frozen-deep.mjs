// Parallelize independent frozen full Matches; never reorder online updates.
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
import {availableParallelism} from 'node:os';
import {runStrategyMatch} from './dev/simulator/strategy-match.js';
import {createSession,validateModel} from '../models/deep.mjs';

export function frozenDeepMatch(model,seed,{profile=false}={}) {
  const result=runStrategyMatch({
    seed,policies:['pure-rl-linear','pure-rl-linear'],
    strategyMode:'p2',pureRlSessions:[createSession(model),createSession(model)],
    pureRlAllPhases:false,capturePhaseTimings:profile,
  });
  if(!result.completed||result.score[result.winner]!==4||
     result.pureRlStats.some(s=>s.train||s.updates))
    throw new Error('Frozen Match must finish with zero training updates');
  return result;
}

// Validate the exact expected indices, row shape, completeness and uniqueness
// before mutating shared ordered results. Array.some skips sparse holes, so
// it must NEVER be used for the final completeness assertion.
export function acceptWorkerRows(chunk,rows,ordered,filled) {
  if(!Array.isArray(chunk)||!Array.isArray(rows)||rows.length!==chunk.length)
    throw new Error('Frozen worker returned an incomplete batch');
  const expected=new Set(chunk.map(t=>t.index)),seen=new Set();
  if(expected.size!==chunk.length)throw new Error('Duplicate task index');
  for(const row of rows){
    if(!row||!Number.isSafeInteger(row.index)||!expected.has(row.index)||
       seen.has(row.index)||filled.has(row.index))
      throw new Error('Frozen worker returned duplicate or unexpected index');
    if(!row.result||row.result.completed!==true||
       ![0,1].includes(row.result.winner)||
       !Array.isArray(row.result.score)||
       row.result.score[row.result.winner]!==4)
      throw new Error('Frozen worker returned invalid Match');
    seen.add(row.index);
  }
  if(seen.size!==expected.size)
    throw new Error('Frozen worker omitted a Match result');
  for(const row of rows){filled.add(row.index);ordered[row.index]=row.result;}
}
if(!isMainThread) {
  const results=workerData.tasks.map(t=>({index:t.index,
    result:frozenDeepMatch(workerData.model,t.seed,{profile:workerData.profile})}));
  parentPort.postMessage(results);
}
export async function frozenDeepBatch(model,seeds,
  {workers=Math.min(4,availableParallelism()),profile=false}={}) {
  validateModel(model);
  const parallelism=availableParallelism();
  if(!Number.isInteger(workers)||workers<1||workers>16||workers>parallelism)
    throw new Error('Worker count exceeds supported logical CPU count');
  // Defensive upper bound on raw float storage only; JS object/worker runtime
  // overhead is larger. For huge models use serial evaluation or more memory.
  if(Object.keys(model.weights).length*16*8*workers>128*1024*1024)
    throw new Error('Frozen worker model memory estimate exceeds 128 MiB');
  const tasks=seeds.map((seed,index)=>({seed,index}));
  if(workers===1||tasks.length<2)
    return tasks.map(t=>frozenDeepMatch(model,t.seed,{profile}));
  const chunks=Array.from({length:Math.min(workers,tasks.length)},()=>[]);
  for(let i=0;i<tasks.length;i++)chunks[i%chunks.length].push(tasks[i]);
  const ordered=new Array(tasks.length),filled=new Set(),running=[];
  try{
    await Promise.all(chunks.map(chunk=>new Promise((resolve,reject)=>{
      const worker=new Worker(new URL(import.meta.url),{
        workerData:{model,tasks:chunk,profile}});
      running.push(worker);
      let received=false;
      worker.on('message',rows=>{
        if(received){reject(new Error('Frozen worker emitted duplicate batches'));return;}
        try{acceptWorkerRows(chunk,rows,ordered,filled);received=true;}
        catch(error){reject(error);}
      });
      worker.on('error',reject);
      worker.on('exit',code=>code===0&&received?resolve():reject(
        new Error('Frozen Match worker ended before completing all jobs: '+code)));
    })));
  }catch(error){
    await Promise.allSettled(running.map(worker=>worker.terminate()));
    throw error;
  }
  if(filled.size!==tasks.length||
     Array.from({length:tasks.length},(_,i)=>i).some(i=>!filled.has(i)||!(i in ordered)))
    throw new Error('Missing frozen Match result');
  return ordered;
}
