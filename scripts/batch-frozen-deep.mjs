// Parallelize independent, frozen complete Matches only.
// Never parallelize online SARSA/Actor-Critic updates (which are order-dependent).
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
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
if(!isMainThread) {
  const results=workerData.tasks.map(t=>({index:t.index,
    result:frozenDeepMatch(workerData.model,t.seed,{profile:workerData.profile})}));
  parentPort.postMessage(results);
}
export async function frozenDeepBatch(model,seeds,{workers=4,profile=false}={}) {
  validateModel(model);
  if(!Number.isInteger(workers)||workers<1||workers>16)throw new Error('Invalid worker count');
  const tasks=seeds.map((seed,index)=>({seed,index}));
  if(workers===1||tasks.length<2)
    return tasks.map(t=>frozenDeepMatch(model,t.seed,{profile}));
  const chunks=Array.from({length:Math.min(workers,tasks.length)},()=>[]);
  for(let i=0;i<tasks.length;i++)chunks[i%chunks.length].push(tasks[i]);
  const ordered=new Array(tasks.length);
  await Promise.all(chunks.map(chunk=>new Promise((resolve,reject)=>{
    const worker=new Worker(new URL(import.meta.url),{
      workerData:{model,tasks:chunk,profile}});
    let received=false;
    worker.on('message',rows=>{
      for(const row of rows)ordered[row.index]=row.result;
      received=true;
    });
    worker.on('error',reject);
    worker.on('exit',code=>code===0&&received?resolve():reject(
      new Error('Frozen Match worker ended before completing all jobs: '+code)));
  })));
  if(ordered.some(x=>!x))throw new Error('Missing frozen Match result');
  return ordered;
}
