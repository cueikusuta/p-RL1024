import {createPureRlModel,validatePureRlModel,rankPureRlActions,legalPureRlActions} from '../src/ai/pure-rl-linear.js';

export const featureVersion='pure-rl-sparse-public';
export const createModel=()=>createPureRlModel();
export const loadModel=text=>validatePureRlModel(JSON.parse(text));
export const saveModel=model=>JSON.stringify(validatePureRlModel(model));
export function rankActions(model,view,legal=legalPureRlActions(view)) {
  const permitted=new Set(legal.map(action=>JSON.stringify(action)));
  const rows=rankPureRlActions(model,view).filter(row=>permitted.has(JSON.stringify(row.action)));
  if (!rows.length && legal.length) throw new Error('No permitted action');
  return rows;
}
