// I857: anonymous OWN-set capability funnel for exploratory CPU matches.
// These counts are availability, not desirability or proven tactical success.
// Never inspect the opponent's true team, set IDs or hidden selections.
import {analyzeSetCapabilities} from '../../../src/ai/set-capabilities.js';

export const TRACKED_TACTIC_TAGS=Object.freeze([
  'hazard','yawn','screen','batonPass','trickRoom',
]);
const MOVE_IDS=Object.freeze({
  hazard:new Set(['stealthrock','spikes','toxicspikes','stickyweb']),
  yawn:new Set(['yawn']),
  screen:new Set(['reflect','lightscreen','auroraveil']),
  batonPass:new Set(['batonpass']),
  trickRoom:new Set(['trickroom']),
});
const freshRow=()=>({
  offeredRounds:0,draftedRounds:0,selectedRounds:0,
  activeDecisions:0,usedMoveDecisions:0,
  inspectedDecisions:0,legalMoveDecisions:0,
  structurallyOpenDecisions:0,clearlyLowValueDecisions:0,
  indeterminateDecisions:0,usedWhenStructurallyOpenDecisions:0,
  usedDespiteClearlyLowValueDecisions:0,
});
export function createTacticCapabilityFunnel() {
  return {rounds:0,byTag:Object.fromEntries(
    TRACKED_TACTIC_TAGS.map(tag=>[tag,freshRow()])),
    caution:'OWN capability and structural opportunity only; structurally open is NOT a verified profitable action. Counts do not prove tactical effect or success; no usage quota.',
  };
}
const ids=sets=>{
  if(!Array.isArray(sets)||sets.some(s=>typeof s?.instanceId!=='string'))
    throw new TypeError('Expected own confirmed set stage');
  const out=new Set(sets.map(s=>s.instanceId));
  if(out.size!==sets.length)throw new TypeError('Duplicate own set');
  return out;
};
const subset=(set,sup)=>[...set].every(id=>sup.has(id));
const tags=(sets,data)=>new Set(sets.flatMap(set=>
  analyzeSetCapabilities(set,data).tags.filter(tag=>
    TRACKED_TACTIC_TAGS.includes(tag))));

// Call once per round, after six-member draft and three-member selection.
// The stages are generated from the SAME seat's authorized Match.view only.
export function recordTacticCapabilityRound(funnel,
  {offered,drafted,selected,data}) {
  if(!funnel?.byTag||!data?.moves)throw new TypeError('Missing funnel or data');
  const a=ids(offered),b=ids(drafted),c=ids(selected);
  if(!subset(b,a)||!subset(c,b)||drafted.length!==6||selected.length!==3)
    throw new TypeError('Own roster stages must be nested confirmed teams');
  const stageTags={
    offered:tags(offered,data),
    drafted:tags(drafted,data),
    selected:tags(selected,data),
  };
  funnel.rounds++;
  for(const tag of TRACKED_TACTIC_TAGS) {
    const row=funnel.byTag[tag];
    if(stageTags.offered.has(tag))row.offeredRounds++;
    if(stageTags.drafted.has(tag))row.draftedRounds++;
    if(stageTags.selected.has(tag))row.selectedRounds++;
  }
}

// A high-confidence NO-GO check, not a judge of tactical advantage.
// Open means the move is available without an obvious structural blocker.
// Unknown means even this limited inference is unsafe.
export function classifyStructuralTacticOpportunity(tag,view,{moveId=null}={}) {
  if(!TRACKED_TACTIC_TAGS.includes(tag)||!view?.self||!view?.opponent||
     view.requestType!=='turn')return 'unavailable-observation';
  const slots=view.self.team?.[view.self.active]?.moves;
  if(!Array.isArray(slots))return 'unavailable-observation';
  const eligible=slots.filter(slot=>MOVE_IDS[tag].has(slot?.id)&&
    (moveId===null||slot.id===moveId)&&slot.disabled!==true);
  if(!eligible.length)return 'no-legal-move';
  const side=(index)=>view.field?.sideEffects?.[index]??[];
  if(tag==='hazard'){
    if(view.opponent.remaining<=1)return 'clearly-low-value';
    const max={stealthrock:1,spikes:3,toxicspikes:2,stickyweb:1};
    if(eligible.every(slot=>{
      const current=side(1).find(x=>x.id===slot.id);
      return current&&Number(current.layers??1)>=max[slot.id];
    }))return 'clearly-low-value';
  }
  if(tag==='yawn'&&view.opponent.status)return 'clearly-low-value';
  if(tag==='screen'){
    const missing=eligible.some(slot=>!side(0).some(e=>
      e.id===slot.id&&(e.duration??1)>0));
    if(!missing)return 'clearly-low-value';
    // If Aurora Veil is the only eligible screen in wrong weather,
    // failure is obvious, but other screen moves remain valid.
    if(eligible.every(slot=>slot.id==='auroraveil')&&
       !['hail','snowscape'].includes(view.field?.weather))
      return 'clearly-low-value';
  }
  if(tag==='batonPass'&&
     !view.self.team.some((p,i)=>i!==view.self.active&&p.hp>0))
    return 'clearly-low-value';
  // Trick Room can be tactically reversed, so an already active Room is
  // uncertain rather than treated as a useless or beneficial action.
  if(tag==='trickRoom'&&(view.field?.trickRoomTurns??0)>0)
    return 'indeterminate';
  return 'structurally-open';
}

// Called only when Strategy submits a legal action. The active set is
// resolved from the authorized OWN selection, never from opponent truth.
export function recordTacticCapabilityAction(funnel,
  {activeSet,action,moveId,data,requestType,view=null}) {
  if(!funnel?.byTag||!activeSet||!data?.moves)
    throw new TypeError('Missing own tactic evidence');
  if(requestType!=='turn')return;
  const available=tags([activeSet],data);
  for(const tag of TRACKED_TACTIC_TAGS) {
    if(!available.has(tag))continue;
    const row=funnel.byTag[tag];
    row.activeDecisions++;
    const used=action?.kind==='move'&&MOVE_IDS[tag].has(moveId);
    if(used)row.usedMoveDecisions++;
    if(view){
      row.inspectedDecisions++;
      const verdict=classifyStructuralTacticOpportunity(tag,view);
      if(verdict!=='no-legal-move'&&verdict!=='unavailable-observation')
        row.legalMoveDecisions++;
      if(verdict==='structurally-open'){
        row.structurallyOpenDecisions++;
        // At the TAG level a move may be open even when the chosen move
        // itself is redundant (e.g. Rocks already up, Spikes absent). Count
        // actual choice against the selected move's public state instead.
        if(used&&classifyStructuralTacticOpportunity(tag,view,{moveId})===
            'structurally-open')row.usedWhenStructurallyOpenDecisions++;
      } else if(verdict==='clearly-low-value')
        row.clearlyLowValueDecisions++;
      else if(verdict==='indeterminate')
        row.indeterminateDecisions++;
      if(used&&classifyStructuralTacticOpportunity(tag,view,{moveId})===
          'clearly-low-value')row.usedDespiteClearlyLowValueDecisions++;
    }
  }
}
