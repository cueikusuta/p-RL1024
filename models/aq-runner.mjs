import {SeededRandom} from '../src/battle.js';
import {mixSeed} from '../aq-runtime/scripts/dev/simulator/runner.js';
import {runStrategyMatch} from '../scripts/optimized/strategy-match.mjs';
import {createPureRlModel,validatePureRlModel,createPureRlSession} from '../src/ai/pure-rl-linear.js';
import {rankPureRlPhaseActions} from '../src/ai/pure-rl-phases.js';
import {rankActions as rankLinear} from './linear.mjs';
import {createModel as createDeep,validateModel as validateDeep,createSession as deepSession} from './deep.mjs';
import {createOpponent,weightsSha256} from './aq-base.mjs';

export {mixSeed,weightsSha256};
export const createModel=(id,seed)=>id==='RL-Linear'?createPureRlModel():id==='RL-Deep'?createDeep(seed):(()=>{throw new Error('Unsupported model');})();
export const validateModel=(id,model)=>id==='RL-Linear'?validatePureRlModel(model):id==='RL-Deep'?validateDeep(model):(()=>{throw new Error('Unsupported model');})();
export function match({id,model,seed,seat,train=false,epsilon=0.35}) {
  validateModel(id,model);
  const rng=new SeededRandom(mixSeed(seed,0xa1,seat));
  const ranker=(m,v)=>['draft','selection'].includes(v.phase)?rankPureRlPhaseActions(m,v,{seat}):rankLinear(m,v);
  const session=id==='RL-Linear'?createPureRlSession(model,{train,epsilon:train?epsilon:0,rng,ranker}):deepSession(model,{train,rng});
  const policies=['aq-base','aq-base'],sessions=[null,null];policies[seat]='pure-rl-linear';sessions[seat]=session;
  const result=runStrategyMatch({seed,policies,pureRlSessions:sessions,pureRlAllPhases:id==='RL-Linear',strategyMode:'integrated',v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:createOpponent()});
  if(!result.completed||![0,1].includes(result.winner)||result.score[result.winner]!==4)throw new Error('AQ-Base Match did not complete');
  return result;
}
