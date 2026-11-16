// Opt-in, seat-local human review of authority-accepted draft picks.
// Never include opponent hidden state. This is diagnostic evidence, not a policy.
import {analyzeSetCapabilities} from '../../src/ai/set-capabilities.js';

const safeInt=x=>Number.isSafeInteger(x)&&x>=0;
export function summarizeAcceptedDraftPick(observation,legalActions,choice,utilities,data){
  if(observation?.phase!=='draft'||!Array.isArray(legalActions)||
     !legalActions.length||!safeInt(choice)||choice>=legalActions.length||
     !Array.isArray(utilities)||utilities.length!==legalActions.length||
     utilities.some(x=>!Number.isFinite(x)))
    throw new TypeError('Invalid accepted draft review inputs');
  const carry=observation.self?.carry??[];
  const fresh=observation.self?.fresh??[];
  const credits=observation.self?.credits;
  if(!Array.isArray(carry)||!Array.isArray(fresh)||!safeInt(credits))
    throw new TypeError('Invalid seat-local draft offer');
  const pool=[...carry,...fresh];
  const ids=pool.map(p=>p.instanceId);
  const byId=new Map(ids.map((id,i)=>[id,i]));
  if(!pool.length||byId.size!==pool.length||ids.some(x=>x==null))
    throw new TypeError('Duplicate or missing draft offer IDs');
  const candidates=legalActions.map(action=>{
    if(!Array.isArray(action)||action.length!==6||new Set(action).size!==6||
       action.some(id=>!byId.has(id)))
      throw new TypeError('Draft candidate is not a unique own-pool six');
    return action.map(id=>byId.get(id));
  });
  const carried=new Set(carry.map(p=>p.instanceId));
  const roster=pool.map(p=>({
    id:String(p.instanceId),
    species:String(p.speciesName??p.speciesId??p.setId??'unknown'),
    set_id:String(p.setId??''),
    credit_cost:carried.has(p.instanceId)?0:p.cost,
    is_carry:carried.has(p.instanceId),
    types:Array.isArray(p.types)?p.types.slice():[],
    item:String(p.itemId??''),
    ability:String(p.abilityId??''),
    mega:Boolean(p.mega),
    roles:[...analyzeSetCapabilities(p,data).tags].sort()
  }));
  if(roster.some(p=>!safeInt(p.credit_cost)))
    throw new TypeError('Invalid draft offered credit price');
  const ranks=legalActions.map((_,i)=>i).sort((a,b)=>
    utilities[b]-utilities[a]||a-b);
  const picked=candidates[choice];
  const pickedSpent=picked.reduce((s,i)=>s+roster[i].credit_cost,0);
  if(pickedSpent>credits||utilities[ranks[0]]-utilities[choice]>1e-5)
    throw new Error('Untrusted draft utility or accepted choice');
  return {
    schema:'rnad-accepted-draft-pick-review-v1',
    round:Number.isSafeInteger(observation.round)?observation.round:null,
    credits,roster,candidate_codes:candidates,
    accepted_choice_index:choice,
    accepted_roster_codes:picked,
    accepted_spent:pickedSpent,
    teacher_rank:ranks.indexOf(choice)+1,
    teacher_top10:ranks.slice(0,10).map(i=>({candidate_index:i,utility:utilities[i]})),
    teacher_best_utility:utilities[ranks[0]],
    accepted_utility:utilities[choice]
  };
}
