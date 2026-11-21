// Read-only diagnostic on authority-accepted, seat-local 3-phase decisions.
// Measures feature-vector ties for the exact same engine-legal policy support.
// No raw opponent secrets, model weights or privileged battle state are used.
import {encodeRnadThreePhaseDecisionV2}
  from '../../src/ml/rnad-phase-features-v2.js';
import {encodeRnadThreePhaseDecisionV3}
  from '../../src/ml/rnad-phase-features-v3.js';

const PHASES=['draft','selection','battle'];
const fresh=()=>({decisions:0,legal_candidates:0,
  v2_duplicate_candidates:0,v3_duplicate_candidates:0,
  v2_tied_chosen_decisions:0,v3_tied_chosen_decisions:0,
  v2_any_collision_decisions:0,v3_any_collision_decisions:0,
  v3_reduced_collisions_decisions:0});

export function compareAcceptedRnadCandidateCollisions(rows){
  if(!Array.isArray(rows)||!rows.length)
    throw new TypeError('Missing trusted accepted three-phase timeline');
  const phases=Object.fromEntries(PHASES.map(x=>[x,fresh()]));
  for(const row of rows){
    const phase=row?.phase;
    const supported=row?.policy_eligible_actions??row?.legal_actions;
    if(!PHASES.includes(phase)||!row.observation||
       row.observation.phase!==phase||
       ![0,1].includes(row.player_seat)||
       !Array.isArray(supported)||!supported.length||
       !row.accepted_action)
      throw new TypeError('Invalid accepted seat/phase or legal support');
    const chosen=supported.findIndex(a=>
      JSON.stringify(a)===JSON.stringify(row.accepted_action));
    if(chosen<0||supported.filter(a=>
      JSON.stringify(a)===JSON.stringify(row.accepted_action)).length!==1)
      throw new Error('Missing or duplicate accepted legal action');
    const v2=encodeRnadThreePhaseDecisionV2(row.observation,supported);
    const v3=encodeRnadThreePhaseDecisionV3(row.observation,supported);
    if(v2.actions.length!==supported.length||
       v3.actions.length!==supported.length||
       v3.actions.some((a,i)=>JSON.stringify(a.slice(0,16))!==
         JSON.stringify(v2.actions[i])))
      throw new Error('V3 prefix/authority candidate mismatch');
    const s=phases[phase];
    s.decisions++;s.legal_candidates+=supported.length;
    const seen=[];
    for(const [prefix,encoded] of [['v2',v2],['v3',v3]]){
      const keys=encoded.actions.map(a=>JSON.stringify(a));
      const counts=new Map();
      for(const key of keys)counts.set(key,(counts.get(key)??0)+1);
      const duplicated=keys.length-counts.size;
      s[prefix+'_duplicate_candidates']+=duplicated;
      s[prefix+'_tied_chosen_decisions']+=Number(counts.get(keys[chosen])>1);
      s[prefix+'_any_collision_decisions']+=Number(duplicated>0);
      seen.push(duplicated);
    }
    if(seen[1]>seen[0])throw new Error('V3 added impossible prefix collision');
    if(seen[1]<seen[0])s.v3_reduced_collisions_decisions++;
  }
  if(PHASES.some(phase=>phases[phase].decisions===0))
    throw new Error('Missing draft/selection/battle in accepted Match');
  for(const s of Object.values(phases)){
    s.v2_duplicate_fraction=s.v2_duplicate_candidates/s.legal_candidates;
    s.v3_duplicate_fraction=s.v3_duplicate_candidates/s.legal_candidates;
  }
  return {schema:'rnad-i5-three-phase-accepted-collision-diagnostic-v1',
    phases,model_strength_improvement_confirmed:false};
}
