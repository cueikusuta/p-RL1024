// RNAD-002 opt-in three-phase numeric features, versioned separately from
// battle-only RNAD-001. No opponent roster, hidden move or Match internals.
import {RNAD_OBSERVATION_SCHEMA} from './legal-action-adapter.js';
import {encodeRnadDecision} from './rnad-features.js';

export const RNAD_THREE_PHASE_FEATURE_SCHEMA='rnad-three-phase-features-v1';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const safe=x=>finite(x)?x:0;
const clamp=(x,lo=0,hi=1)=>Math.min(hi,Math.max(lo,safe(x)));
const fraction=(x,d)=>clamp(safe(x)/d);
const mean=(a,f)=>a.length?a.reduce((s,x)=>s+safe(f(x)),0)/a.length:0;
const hash=s=>{
  if(typeof s!=='string'||!s)return 0;
  let v=2166136261;
  for(const letter of s)v=Math.imul(v^letter.charCodeAt(0),16777619)>>>0;
  return (v%1000000)/1000000;
};
function stat(p,k){return safe(p?.stats?.[k]??p?.baseStats?.[k]);}
function typed(p){return Array.isArray(p?.types)?p.types.filter(x=>typeof x==='string'):[];}
function teamSummary(list){
  const allTypes=new Set(list.flatMap(typed));
  const species=new Set(list.map(p=>p.speciesId));
  const items=new Set(list.map(p=>p.itemId).filter(Boolean));
  return {
    types:allTypes.size,species:species.size,items:items.size,
    mega:list.filter(p=>p.mega).length,
    cost:list.reduce((sum,p)=>sum+safe(p.cost),0),
    atk:mean(list,p=>stat(p,'atk')),
    def:mean(list,p=>stat(p,'def')),
    spa:mean(list,p=>stat(p,'spa')),
    spd:mean(list,p=>stat(p,'spd')),
    spe:mean(list,p=>stat(p,'spe')),
    hp:mean(list,p=>safe(p.hp)/Math.max(1,safe(p.maxHP))),
    hash:mean(list,p=>hash(p.speciesId))
  };
}
function assertFinite(row,n){
  if(row.length!==n||row.some(x=>!finite(x)))
    throw new TypeError('Invalid three-phase numeric features');
  return row;
}
export function encodeRnadThreePhaseDecision(observation,legalActions){
  if(observation?.schema!==RNAD_OBSERVATION_SCHEMA||
    ![0,1].includes(observation.seat)||
    !['draft','selection','battle'].includes(observation.phase)||
    !Array.isArray(legalActions)||legalActions.length===0||
    legalActions.length>60000)
    throw new TypeError('Invalid three-phase actor observation or legal set');
  if(observation.phase==='battle'){
    const old=encodeRnadDecision(observation,legalActions);
    // Versioned three-phase network: stable phase indicator at 0,1,2.
    const state=old.state.slice();
    state[0]=0;state[1]=0;state[2]=1;
    return {schema:RNAD_THREE_PHASE_FEATURE_SCHEMA,state,
      actions:old.actions};
  }
  const own=observation.self;
  if(!own||typeof own!=='object')throw new TypeError('Missing own phase roster');
  const roster=observation.phase==='draft'?
    [...(own.carry??[]),...(own.fresh??[])]:own.team;
  if(!Array.isArray(roster)||roster.length<6||roster.length>24||
    roster.some(p=>typeof p?.instanceId!=='string')||
    new Set(roster.map(p=>p.instanceId)).size!==roster.length)
    throw new TypeError('Malformed own phase roster');
  const byId=new Map(roster.map(p=>[p.instanceId,p]));
  const t=teamSummary(roster);
  const score=Array.isArray(observation.score)?observation.score:[0,0];
  const obsOpp=observation.opponent??{};
  const round=safe(observation.round);
  const state=assertFinite([
    observation.phase==='draft'?1:0,
    observation.phase==='selection'?1:0,0,
    fraction(score[observation.seat],4),
    fraction(score[1-observation.seat],4),fraction(round,7),
    fraction(own.credits,40),fraction(observation.grant,30),
    fraction(own.fresh?.length,20),fraction(own.carry?.length,6),
    fraction(roster.length,20),
    fraction(obsOpp.species?.length,6),
    fraction(t.cost,42),fraction(t.mega,6),fraction(t.types,12),
    fraction(t.items,6),fraction(t.species,6),
    fraction(t.atk,300),fraction(t.def,300),
    fraction(t.spa,300),fraction(t.spd,300),
    fraction(t.spe,300),fraction(t.hp,1),t.hash,
    hash(obsOpp.species?.[0]),
    hash(obsOpp.species?.[1]),hash(obsOpp.species?.[2]),
    fraction(observation.budget,40),
    obsOpp.ready?1:0,obsOpp.teamConfirmed?1:0,
    obsOpp.selectionConfirmed?1:0,1
  ],32);
  const carryIds=new Set((own.carry??[]).map(p=>p.instanceId));
  const actions=legalActions.map(ids=>{
    if(!Array.isArray(ids)||ids.length!==(observation.phase==='draft'?6:3)||
      new Set(ids).size!==ids.length||ids.some(id=>!byId.has(id)))
      throw new TypeError('Invalid own draft/selection candidate');
    const selected=ids.map(id=>byId.get(id)),a=teamSummary(selected);
    const lead=selected[0];
    const spent=observation.phase==='draft'?
      selected.filter(p=>!carryIds.has(p.instanceId)).reduce((s,p)=>s+safe(p.cost),0):0;
    const row=[
      observation.phase==='draft'?1:0,observation.phase==='selection'?1:0,
      fraction(spent,40),fraction(a.cost,42),fraction(a.mega,6),
      fraction(a.types,12),fraction(a.items,6),fraction(a.species,6),
      fraction(a.atk,300),fraction(a.def,300),
      fraction(a.spa,300),fraction(a.spd,300),
      fraction(a.spe,300),
      fraction(selected.filter(p=>carryIds.has(p.instanceId)).length,6),
      hash(lead.speciesId),a.hash
    ];
    return assertFinite(row,16);
  });
  return {schema:RNAD_THREE_PHASE_FEATURE_SCHEMA,state,actions};
}
