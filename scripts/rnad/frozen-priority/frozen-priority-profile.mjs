// Frozen inference only. Do not install this in a stochastic training Actor
// without a new behavior-probability contract.
import {firstThreatGate} from './g421-threat-gate.mjs';
export function createFrozenPriorityProfile({mode='off',deps,onAccepted=()=>{}}={}){
 if(!['off','shadow','public-lethal-priority-v1'].includes(mode))throw new TypeError('Unknown priority profile');
 let used=false,pending=null;
 const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 return Object.freeze({
  id:mode,needsPublicFrame:mode!=='off',
  decide({observation,legalActions,logits,index,publicContext}){
   if(pending)throw new Error('Prior priority recommendation not settled');
   if(mode==='off'||used||observation.phase!=='battle'||observation.requestType!=='turn')return index;
   const reveal=publicContext?.publicFrame?.publicObservation;
   if(!reveal)return index; // Missing or non-seat-scoped reveal fails closed.
   const visible={...observation,opponent:{...observation.opponent,publicObservation:reveal}};
   const g=firstThreatGate(visible,legalActions,logits,index,deps);
   if(!g?.priority)return index;
   const i=legalActions.findIndex(a=>same(a,g.priority.action));
   if(i<0||i===index)throw new Error('Priority choice absent from supported actions');
   const chosen=mode==='shadow'?index:i;
   pending={profile:mode,original:legalActions[index],proposed:legalActions[i],accepted:legalActions[chosen],gap:g.priority.gap,turn:observation.turn};
   return chosen;
  },
  accept(action){
   if(!pending)return;
   if(!same(action,pending.accepted))throw new Error('Authority accepted a different priority action');
   used=true;const event=pending;pending=null;onAccepted(event);
  },
  // Rejected/aborted proposal does not consume the per-match intervention.
  reject(){pending=null;},
 });
}
