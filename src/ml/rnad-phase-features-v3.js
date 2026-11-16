// Experimental v3: 48 public/own state dimensions and 32 action dimensions.
// Frozen v1/v2 encoders and model bytes are unaffected.  The extra 16 action
// coordinates are bounded categorical bucket indicators (not scalar hashes).
// Model's trainable policy projection learns coefficients for those buckets.
// History is built by the trusted receipt-backed per-seat Match runner only.
import {encodeRnadThreePhaseDecisionV2}
  from './rnad-phase-features-v2.js';
export const RNAD_THREE_PHASE_FEATURE_SCHEMA_V3='rnad-three-phase-features-v3';
export const RNAD_V3_STATE_FEATURES=48;
export const RNAD_V3_ACTION_FEATURES=32;
const num=x=>typeof x==='number'&&Number.isFinite(x)?x:0;
const clamp=(x,lo=0,hi=1)=>Math.min(hi,Math.max(lo,num(x)));
const ratio=(n,d)=>clamp(num(n)/Math.max(1,num(d)));
function bucket(value,n){
  if(typeof value!=='string'||!value)return -1;
  let h=2166136261;
  for(const c of value)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;
  return h%n;
}
function species(p){return p?.speciesId??'';}
function ids(p,kind){return p?.[kind]??'';}
const ownTeam=obs=>obs?.phase==='draft'?
  [...(obs.self?.carry??[]),...(obs.self?.fresh??[])]:obs.self?.team??[];
function categorical(selection,{move=null}={}){
  const values=new Array(16).fill(0);
  const groups=[
    selection.map(species),
    selection.map(p=>ids(p,'abilityId')),
    selection.map(p=>ids(p,'itemId')),
    move!==null?[move]:selection.flatMap(p=>
      (p.moves??[]).map(m=>typeof m==='string'?m:m?.id))
  ];
  for(let g=0;g<4;g++){
    const distinct=[...new Set(groups[g].filter(x=>typeof x==='string'&&x))];
    for(const name of distinct){
      const b=bucket(name,4);
      if(b>=0)values[g*4+b]+=1/Math.max(1,distinct.length);
    }
  }
  return values;
}
function extras(obs){
  const team=ownTeam(obs);
  const alive=team.filter(p=>num(p?.hp)>0);
  const fraction=(xs,fn)=>xs.length?
    xs.reduce((s,x)=>s+fn(x),0)/xs.length:0;
  const types=new Set(team.flatMap(p=>Array.isArray(p?.types)?p.types:[]));
  const abilities=new Set(team.map(p=>p?.abilityId).filter(Boolean));
  const items=new Set(team.map(p=>p?.itemId).filter(Boolean));
  const hist=obs.acceptedHistory??{lastKind:'none',consecutiveSwitches:0,
    lastSwitchTarget:-1,lastActive:-1};
  if(hist===null||!['none','move','switch','draft','selection'].includes(hist.lastKind)||
     !Number.isSafeInteger(hist.consecutiveSwitches)||
     hist.consecutiveSwitches<0||hist.consecutiveSwitches>3000||
     !Number.isSafeInteger(hist.lastSwitchTarget)||
     hist.lastSwitchTarget< -1||hist.lastSwitchTarget>5||
     !Number.isSafeInteger(hist.lastActive)||
     hist.lastActive< -1||hist.lastActive>5)
    throw new TypeError('Invalid trusted accepted-history v3 structure');
  const count=team.length;
  const opp=obs?.opponent??{};
  const shown=opp.publicObservation??{};
  return [
    hist.lastKind==='move'?1:0,
    hist.lastKind==='switch'?1:0,
    clamp(hist.consecutiveSwitches/3),
    clamp((hist.lastSwitchTarget+1)/6),
    clamp((hist.lastActive+1)/6),
    fraction(team,p=>ratio(p?.hp,p?.maxHP)),
    count?alive.length/count:0,
    clamp(types.size/12),
    clamp(abilities.size/6),
    clamp(items.size/6),
    clamp((shown.usedMoves??[]).length/4),
    clamp((shown.observedItemIds??[]).length/3),
    clamp((shown.observedAbilityIds??[]).length/3),
    clamp((obs.phase==='draft'?(obs.self?.fresh??[]).length:0)/20),
    clamp((obs.phase==='selection'?(obs.opponent?.species??[]).length:0)/6),
    obs.phase==='battle'?1:0
  ];
}
export function encodeRnadThreePhaseDecisionV3(obs,legalActions){
  const prior=encodeRnadThreePhaseDecisionV2(obs,legalActions);
  const state=[...prior.state,...extras(obs)];
  const roster=ownTeam(obs);
  const byId=new Map(roster.map(p=>[p.instanceId,p]));
  const active=obs.self?.team?.[obs.self?.active];
  const actions=legalActions.map((choice,i)=>{
    let selection,move=null;
    if(obs.phase==='battle'){
      if(choice.kind==='switch'){
        const p=roster[choice.index];
        if(!p)throw new TypeError('Missing known switch candidate');
        selection=[p];
      }else if(choice.kind==='move'){
        if(!Number.isSafeInteger(choice.index)||!active?.moves?.[choice.index])
          throw new TypeError('Missing known move candidate');
        selection=[active];
        const m=active.moves[choice.index];
        move=typeof m==='string'?m:m?.id;
      }else{
        selection=[active].filter(Boolean);
      }
    }else{
      selection=choice.map(id=>byId.get(id));
      if(selection.some(x=>!x))throw new TypeError('Unknown own roster choice');
    }
    return [...prior.actions[i],...categorical(selection,{move})];
  });
  if(state.length!==RNAD_V3_STATE_FEATURES||
     actions.some(row=>row.length!==RNAD_V3_ACTION_FEATURES)||
     [...state,...actions.flat()].some(v=>!Number.isFinite(v)))
    throw new TypeError('Nonfinite or malformed v3 features');
  return {schema:RNAD_THREE_PHASE_FEATURE_SCHEMA_V3,state,actions};
}
