import {validateTeam} from '../../../src/draft.js';
import {setValue} from '../../../src/ai/selection-basic.js';
export {setValue,publicRosterTypes,chooseBasicSelection as chooseSelection}
  from '../../../src/ai/selection-basic.js';

export function chooseTeam(view,{data,forcedSetId=null,rng=null}={}) {
  const carry=view.self.carry??[],fresh=view.self.fresh??[];
  const options=[...carry,...fresh], carryIds=new Set(carry.map(p=>p.instanceId));
  const credits=view.self.credits;
  if(forcedSetId && !options.some(p=>p.setId===forcedSetId))
    throw new Error('Forced set is absent from the legal candidate pool: '+forcedSetId);
  const scores=options.map(p=>setValue(p,data)+(carryIds.has(p.instanceId)?5:0)+(rng?(rng.integer(1000)/1000-0.5)*0.3:0));
  // Exact six-member optimization, without altering the roster scoring rules:
  // cache the pair bonuses and prune branches whose optimistic upper bound
  // cannot beat the incumbent. This avoids repeated 15-pair leaf scoring.
  const pair=Array.from({length:options.length},()=>Array(options.length).fill(0));
  for(let i=0;i<options.length;i++)for(let j=i+1;j<options.length;j++) {
    const shared=(options[i].types??[]).filter(t=>(options[j].types??[]).includes(t)).length;
    pair[i][j]=shared?-4.5*shared:0.65;
  }
  const optimistic=Array.from({length:options.length+1},()=>Array(7).fill(-Infinity));
  const forcedRemaining=Array(options.length+1).fill(false);
  for(let i=options.length;i>=0;i--) {
    optimistic[i][0]=0;
    if(i===options.length)continue;
    forcedRemaining[i]=forcedRemaining[i+1]||options[i].setId===forcedSetId;
    const ranked=scores.slice(i).sort((a,b)=>b-a);
    let sum=0;
    for(let k=1;k<=Math.min(6,ranked.length);k++) {
      sum+=ranked[k-1];optimistic[i][k]=sum;
    }
  }
  let best=null,bestScore=-Infinity,team=[],indices=[];
  function search(index,cost,score,forced) {
    if(team.length===6) {
      if(!forcedSetId||forced) {
        if(score>bestScore){bestScore=score;best=team.map(p=>p.instanceId);}
      }
      return;
    }
    const missing=6-team.length;
    if(options.length-index<missing)return;
    if(forcedSetId&&!forced&&!forcedRemaining[index])return;
    // At most 0.65 synergy is available per additional teammate pair.
    const remainingPairs=15-team.length*(team.length-1)/2;
    if(score+optimistic[index][missing]+0.65*remainingPairs<=bestScore)return;
    const p=options[index],next=cost+(carryIds.has(p.instanceId)?0:p.cost);
    if(next<=credits) {
      const bonus=indices.reduce((sum,j)=>sum+pair[j][index],0);
      team.push(p);indices.push(index);
      search(index+1,next,score+scores[index]+bonus,forced||p.setId===forcedSetId);
      indices.pop();team.pop();
    }
    search(index+1,cost,score,forced);
  }
  search(0,0,0,false);
  if(!best)throw new Error('No affordable six-member team'+(forcedSetId?' containing '+forcedSetId:''));
  // Always validate the final roster through the authoritative draft rule.
  validateTeam(best,fresh,carry,view.round,credits);
  return best;
}
