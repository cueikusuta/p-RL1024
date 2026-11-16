// Opt-in zero-sum reward shaping for seat-local ACTUALLY ACCEPTED voluntary
// switches. This is a research objective, not a change to Pokémon rules.
// An attacking opportunity is an approximation from the public opponent's
// type(s) and own known move power; it is NOT a counterfactual damage oracle.
import {effectiveness} from '../../src/battle-mechanics.js';
import catalog from '../../data/catalog/battle-data.json' with {type:'json'};
export const SWITCH_OPPORTUNITY_OBJECTIVE='opportunity-v1';
export const MAX_SWITCH_COST=0.02;
export const MAX_MATCH_SWITCH_COST_PER_SEAT=0.20;
const clip=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,x));
const movesOf=p=>Array.isArray(p?.moves)?p.moves:[];
const moveId=m=>typeof m==='string'?m:m?.id;
const typeSet=p=>Array.isArray(p?.types)?p.types.filter(t=>typeof t==='string'):[];
function attackPressure(p,foeTypes,legalMoveIndices=null) {
  if(!p||!foeTypes.length)return 0;
  let best=0;
  for(let i=0;i<movesOf(p).length;i++){
    if(legalMoveIndices&&!legalMoveIndices.has(i))continue;
    const move=catalog.moves?.[moveId(p.moves[i])];
    if(!move||move.category==='Status'||!(move.basePower>0))continue;
    const accuracy=move.accuracy===true?1:clip((move.accuracy??100)/100);
    const stab=typeSet(p).includes(move.type)?1.3:1;
    const mult=clip(effectiveness(move.type,foeTypes,catalog.typeChart),0,4);
    const score=clip((move.basePower/150)*accuracy*stab*mult/2.8);
    best=Math.max(best,score);
  }
  return best;
}
export function acceptedVoluntarySwitchOpportunity(row) {
  const zero={cost:0,pressure:0,target_pressure:0,reason:'not-penalized'};
  if(row?.phase!=='battle'||row?.accepted_action?.kind!=='switch'||
     row?.observation?.requestType!=='turn')return zero;
  const obs=row.observation;
  const team=obs.self?.team,active=obs.self?.active;
  const target=team?.[row.accepted_action.index];
  const p=team?.[active];
  const foeTypes=typeSet(obs.opponent);
  if(!p||!target||!foeTypes.length||
     !Number.isFinite(p.hp)||!Number.isFinite(p.maxHP)||p.maxHP<=0||
     p.hp/p.maxHP<.65||p.status)return zero;
  // Danger of a super-effective incoming attack makes protective switching
  // strategically legitimate even if the active Pokémon could attack.
  if(foeTypes.some(type=>
       effectiveness(type,typeSet(p),catalog.typeChart)>=2))return zero;
  const moves=new Set(row.legal_actions.filter(a=>
    a.kind==='move'&&Number.isInteger(a.index)).map(a=>a.index));
  if(!moves.size)return zero;
  const pressure=attackPressure(p,foeTypes,moves);
  const targetPressure=attackPressure(target,foeTypes);
  if(pressure<=.30||targetPressure>pressure+.10)
    return {...zero,pressure,target_pressure:targetPressure};
  const cost=MAX_SWITCH_COST*clip((pressure-.30)/.70);
  return {cost,pressure,target_pressure:targetPressure,
    reason:'lost-available-attack'};
}
// Apply to an already verified accepted-action joint timeline: no simulator
// internals, opponent hidden sets or proposed/rejected actions are used.
export function shapeSwitchOpportunity(events,originalDecisions){
  if(!Array.isArray(events)||events.length!==originalDecisions.length||
     !events.length)throw new Error('Expected aligned accepted timeline');
  const consumed=[0,0];
  for(let i=0;i<events.length;i++){
    const row=originalDecisions[i],event=events[i];
    if(event.actor!==row.player_seat||event.phase!==row.phase)
      throw new Error('Mismatched accepted authority seat/phase');
    const raw=acceptedVoluntarySwitchOpportunity(row);
    const seat=event.actor;
    const cost=Math.min(raw.cost,
      Math.max(0,MAX_MATCH_SWITCH_COST_PER_SEAT-consumed[seat]));
    consumed[seat]+=cost;
    event.action_kind=row.accepted_action?.kind??null;
    event.request_type=row.observation?.requestType??null;
    event.switch_opportunity_cost=cost;
    event.reward=cost===0?[0,0]:
      [seat===0?-cost:cost,seat===1?-cost:cost];
  }
  return {objective:SWITCH_OPPORTUNITY_OBJECTIVE,
    max_per_switch:MAX_SWITCH_COST,
    max_per_seat:MAX_MATCH_SWITCH_COST_PER_SEAT,
    cost_per_seat:consumed};
}
