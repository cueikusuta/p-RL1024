import {rankStrategyV2Actions} from '../src/ai/strategy-v2.js';
import {legalPureRlActions} from '../src/ai/pure-rl-linear.js';
import {playableSets} from '../src/production-runtime.js';
import data from '../data/catalog/battle-data.json' with {type:'json'};

export const featureVersion='v2e-rules';
export const createModel=()=>({blend:0.28,exposure:0.75,continuation:0.40});
export function loadModel(text) {
  const model=JSON.parse(text),expected=createModel();
  if(JSON.stringify(model)!==JSON.stringify(expected))throw new Error('Invalid fixed V2E weights');
  return model;
}
export const saveModel=model=>JSON.stringify(loadModel(JSON.stringify(model)));
export function rankActions(model,view,legal=legalPureRlActions(view),context={}) {
  const weights=loadModel(JSON.stringify(model));
  const allowed=new Set(legal.map(action=>JSON.stringify(action)));
  const rows=rankStrategyV2Actions(view,{...context,data,catalog:playableSets,variant:'strategy-v2-e',weights}).rows
    .filter(row=>allowed.has(JSON.stringify(row.action)))
    .map(row=>({action:row.action,q:row.score}));
  if(!rows.length&&legal.length)throw new Error('No permitted V2E action');
  return rows;
}
