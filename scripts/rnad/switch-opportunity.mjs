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
  // Diagnostic reason codes are metadata only: never alter the reward,
  // executable legal support, original observations or checkpoint format.
  const zero=reason=>({cost:0,pressure:0,target_pressure:0,reason});
  if(row?.phase!=='battle'||row?.accepted_action?.kind!=='switch')
    return zero('not-voluntary-switch');
  if(row?.observation?.requestType!=='turn')
    return zero('non-turn-switch');
  const obs=row.observation;
  const team=obs.self?.team,active=obs.self?.active;
  const target=team?.[row.accepted_action.index];
  const p=team?.[active];
  const foeTypes=typeSet(obs.opponent);
  if(!p||!target)return zero('missing-known-team');
  if(!foeTypes.length)return zero('opponent-type-unrevealed');
  if(!Number.isFinite(p.hp)||!Number.isFinite(p.maxHP)||p.maxHP<=0)
    return zero('invalid-own-hp');
  if(p.hp/p.maxHP<.65)return zero('own-hp-below-65-percent');
  if(p.status)return zero('own-status');
  // Danger of a super-effective incoming attack makes protective switching
  // strategically legitimate even if the active Pokémon could attack.
  if(foeTypes.some(type=>
       effectiveness(type,typeSet(p),catalog.typeChart)>=2))
    return zero('public-type-threat');
  const moves=new Set(row.legal_actions.filter(a=>
    a.kind==='move'&&Number.isInteger(a.index)).map(a=>a.index));
  if(!moves.size)return zero('no-legal-attack');
  const pressure=attackPressure(p,foeTypes,moves);
  const targetPressure=attackPressure(target,foeTypes);
  if(pressure<=.30)return {...zero('low-attack-pressure'),
    pressure,target_pressure:targetPressure};
  if(targetPressure>pressure+.10)return {...zero('better-switch-attack'),
    pressure,target_pressure:targetPressure};
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

// Read-only diagnostic over an authority-accepted, complete Match decision
// timeline. The caller must separately verify its source/engine/seed/receipt;
// this helper cannot establish provenance from arbitrary JSON.
export function summarizeAcceptedSwitchOpportunities(rows){
  if(!Array.isArray(rows)||!rows.length||rows.some(row=>
    !row||![0,1].includes(row.player_seat)||
    !['draft','selection','battle'].includes(row.phase)||
    !Array.isArray(row.legal_actions)||!row.observation||
    !row.accepted_action))
    throw new TypeError('Expected trusted accepted two-seat Match decisions');
  // Compute the exact existing per-seat cap without mutating accepted rows.
  const clonedEvents=rows.map(row=>({
    actor:row.player_seat,phase:row.phase,reward:[0,0]
  }));
  const shaped=shapeSwitchOpportunity(clonedEvents,rows);
  const bySeat=[0,1].map(seat=>({
    seat,voluntary_switches:0,shaped_switches:0,
    cost:shaped.cost_per_seat[seat],reason_counts:{}
  }));
  for(let i=0;i<rows.length;i++){
    const row=rows[i],seat=row.player_seat;
    if(row.phase!=='battle'||row.accepted_action?.kind!=='switch'||
       row.observation.requestType!=='turn')continue;
    const original=acceptedVoluntarySwitchOpportunity(row);
    const actual=clonedEvents[i].switch_opportunity_cost;
    const reason=original.cost>0&&actual===0?
      'match-budget-exhausted':
      original.cost>actual+1e-12?'match-budget-clipped':original.reason;
    const entry=bySeat[seat];
    entry.voluntary_switches++;
    if(actual>0)entry.shaped_switches++;
    entry.reason_counts[reason]=(entry.reason_counts[reason]??0)+1;
  }
  return {
    schema:'rnad-switch-opportunity-reason-diagnostic-v1',
    objective:SWITCH_OPPORTUNITY_OBJECTIVE,
    voluntary_switches:bySeat.reduce((n,s)=>n+s.voluntary_switches,0),
    shaped_switches:bySeat.reduce((n,s)=>n+s.shaped_switches,0),
    cost_per_seat:shaped.cost_per_seat,
    by_seat:bySeat
  };
}
