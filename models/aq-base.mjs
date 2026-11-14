import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {validatePublicActionQ,rankPublicActionQ} from '../aq-runtime/src/ai/v3-public-action-q.js';
import {chooseFrozenAQBase,AQ_BASE_REPORT_SHA256} from '../aq-runtime/scripts/dev/simulator/rl-linear-aq-base-opponent.js';

export const featureVersion='aq-base-61d';
export const weightsSha256='5ccc11a3362c7f5a8874847e3afc1f19c539d4ef258bd82704f53d8a1db1946e';
const sha=x=>createHash('sha256').update(x).digest('hex');
export function loadModel(text) {
  const model=validatePublicActionQ(JSON.parse(text));
  if(model.featureCount!==61||sha(JSON.stringify(model))!==weightsSha256)throw new Error('AQ-Base-001 model SHA mismatch');
  return model;
}
export const saveModel=model=>JSON.stringify(loadModel(JSON.stringify(model)));
export const createModel=()=>loadModel(readFileSync(new URL('../aq-runtime/models/AQ-Base-001.json',import.meta.url),'utf8'));
export function createOpponent(){const model=createModel();return {model,modelSha256:weightsSha256,reportSha256:AQ_BASE_REPORT_SHA256};}
// AQ-Base is a bounded correction over the frozen V2E candidate scores; the
// policy's own public-seat features are built by the upstream 61D adapter.
export function rankActions(model,view,legal,{baselineScores,...context}) {
  loadModel(JSON.stringify(model));
  const result=chooseFrozenAQBase(model,view,legal,baselineScores,context);
  const ranked=rankPublicActionQ(model,result.candidates,baselineScores);
  return legal.map((action,i)=>({action,q:ranked.scores[i]})).sort((a,b)=>b.q-a.q);
}
