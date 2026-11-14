// Frozen historical PPO instance; distinct from this repository's RL-Deep.
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {validateDeep,encodeDeep,forwardDeep,createDeepSession} from '../aq-runtime/src/ai/rl-deep.js';
import {rankPureRlAllPhases} from '../aq-runtime/src/ai/pure-rl-phases.js';

export const featureVersion='rl-deep-hashed-public-256';
export const weightsSha256='55673679fe773e4489e1dec505fb434525322175af5cb2ee1c2bf919adaf5d41';
export function loadModel(text) {
  const model=validateDeep(JSON.parse(text));
  if(createHash('sha256').update(JSON.stringify(model)).digest('hex')!==weightsSha256)throw new Error('RL-Deep-001 model SHA mismatch');
  return model;
}
export const createModel=()=>loadModel(readFileSync(new URL('../aq-runtime/models/RL-Deep-001.json',import.meta.url),'utf8'));
export const saveModel=model=>JSON.stringify(loadModel(JSON.stringify(model)));
export const createSession=(model,options)=>createDeepSession(loadModel(JSON.stringify(model)),options);
export function rankActions(model,view,legal,context={}) {
  loadModel(JSON.stringify(model));
  const neutral={schema:'i975-pure-linear-sarsa-v1',weights:{}};
  const allowed=new Set(legal.map(action=>JSON.stringify(action)));
  const candidates=rankPureRlAllPhases(neutral,view,{seat:context.seat??0,consecutiveOwnSwitches:context.consecutiveOwnSwitches??0}).filter(row=>allowed.has(JSON.stringify(row.action)));
  if(!candidates.length){if(legal.length)throw new Error('No permitted action');return [];}
  const scores=forwardDeep(model,candidates.map(c=>encodeDeep(c.features))).probs;
  return candidates.map((c,i)=>({action:c.action,q:scores[i]}));
}
