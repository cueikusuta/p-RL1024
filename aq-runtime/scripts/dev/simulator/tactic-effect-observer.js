// I737 P5: public-state-only evidence of visible tactical transitions.
// Observable field change is not a verified multi-turn strategy success.
// Never emit raw Battle views, secret selections, instance IDs or opponent sets.
const SCREENS=new Set(['reflect','lightscreen','auroraveil']);
const BOOSTS=new Set(['swordsdance','nastyplot','dragondance','agility',
  'quiverdance','bulkup','calmmind','shellsmash','cosmicpower','irondefense']);
const PHASING=new Set(['whirlwind','roar','dragontail','circlethrow']);
const PLAN_IDS=new Set([
  'hazard-yawn-phaze','screens-ace','boost-baton','trickroom-slow',
]);
const sideEffects=(view,index)=>view?.field?.sideEffects?.[index]??[];
const sideEffect=(view,side,id)=>sideEffects(view,side).find(e=>e?.id===id);
const duration=(view,side,id)=>sideEffect(view,side,id)?.duration??0;
const hasEffect=(view,side,id)=>Boolean(sideEffect(view,side,id));
const active=(view)=>view?.self?.team?.[view.self.active];
const boosted=(before,after)=>[...new Set([...Object.keys(before??{}),...Object.keys(after??{})])].some(stat=>
  (Number(after?.[stat])||0)>(Number(before?.[stat])||0));
const observed=(effect,observation)=>({effect,observation});
// Only a completed transition into a new ordinary turn permits saying that
// a planned effect was not visible. Pending pivots, forced replacements,
// unfinished simultaneous actions and terminal snapshots are inconclusive.
// A positive visible field change may be reported at any boundary.
const noVisible=(before,after)=>
  before.phase==='turn'&&before.requestType==='turn'&&
  after.phase==='turn'&&after.requestType==='turn'&&
  !after.self?.confirmed&&
  Number.isSafeInteger(before.turn)&&Number.isSafeInteger(after.turn)&&
  after.turn>before.turn
    ?'not-observed':'not-verifiable';

// The caller captures BEFORE the selected legal action and AFTER both seats
// have had a chance to submit. No identity or move history is returned.
// 'not-observed' means "no visible transition on a resolved next-turn boundary";
// it does not establish a move failed or that a strategy was unsuccessful.
export function observeTacticalPublicEffect({planId,action,moveId,before,after}) {
  if(!PLAN_IDS.has(planId)||!before?.self||!after?.self)
    throw new TypeError('Expected a known plan and own public AI snapshots');
  const sameActive=before.self.active===after.self.active;
  const isMove=action?.kind==='move';
  const isSwitch=action?.kind==='switch';
  if(planId==='hazard-yawn-phaze') {
    if(isMove&&moveId==='stealthrock') {
      const had=hasEffect(before,1,'stealthrock');
      const has=hasEffect(after,1,'stealthrock');
      return observed('opponent-stealthrock-installed',
        !had&&has?'visible-transition':had?'not-verifiable':noVisible(before,after));
    }
    if(isMove&&moveId==='yawn')
      return observed('yawn-volatile-not-in-compact-view','not-verifiable');
    if(isMove&&PHASING.has(moveId))
      return observed('coincident-opponent-public-species-change',
        typeof before.opponent?.speciesId==='string'&&
        typeof after.opponent?.speciesId==='string'&&
        before.opponent.speciesId!==after.opponent.speciesId
          ?'visible-transition':'not-verifiable');
  }
  if(planId==='screens-ace') {
    if(isMove&&SCREENS.has(moveId)) {
      const was=duration(before,0,moveId),now=duration(after,0,moveId);
      const newlyActive=!hasEffect(before,0,moveId)&&hasEffect(after,0,moveId);
      return observed('own-screen-became-active',
        newlyActive||now>was?'visible-transition':noVisible(before,after));
    }
    if(isSwitch) {
      const screenRemains=[...SCREENS].some(id=>
        hasEffect(before,0,id)&&hasEffect(after,0,id));
      return observed('own-switch-under-screen',
        before.self.active!==after.self.active&&screenRemains
          ?'visible-transition':noVisible(before,after));
    }
  }
  if(planId==='boost-baton') {
    if(isMove&&BOOSTS.has(moveId)) {
      const increased=boosted(active(before)?.boosts,active(after)?.boosts);
      return observed('own-boost-applied',
        sameActive&&increased?'visible-transition':
          !sameActive?'not-verifiable':noVisible(before,after));
    }
    // Baton Pass resolves through an additional pivot request, so inspecting
    // the immediate next boundary cannot prove transferred stat stages.
    if(isMove&&moveId==='batonpass')
      return observed('baton-transfer-needs-pivot-observation','not-verifiable');
  }
  if(planId==='trickroom-slow') {
    if(isMove&&moveId==='trickroom')
      return observed('trickroom-became-active',
        (before.field?.trickRoomTurns??0)===0&&
        (after.field?.trickRoomTurns??0)>0
          ?'visible-transition':noVisible(before,after));
    if(isSwitch) {
      const oldSpeed=active(before)?.stats?.spe;
      const newSpeed=active(after)?.stats?.spe;
      const enteredSlow=before.self.active!==after.self.active&&
        Number.isFinite(oldSpeed)&&Number.isFinite(newSpeed)&&
        newSpeed<oldSpeed&&(after.field?.trickRoomTurns??0)>0;
      return observed('slow-participant-entered-under-room',
        enteredSlow?'visible-transition':noVisible(before,after));
    }
  }
  return observed('no-public-transition-contract','not-verifiable');
}

export function summarizeTacticalEffectObservations(events) {
  const totals={followedDecisions:0,visibleTransitions:0,
    noVisibleTransitions:0,notVerifiable:0,byPlan:{},
    caution:'Field transitions are public-state observations, NOT proof of a completed multi-turn combo, causal attribution, or move success. Yawn and Baton Pass need additional state-aware validation.'};
  for(const {planId,effect,observation} of events) {
    if(!PLAN_IDS.has(planId)||
       !['visible-transition','not-observed','not-verifiable'].includes(observation))
      throw new TypeError('Invalid tactic transition evidence');
    const row=totals.byPlan[planId]??={
      followedDecisions:0,visibleTransitions:0,
      noVisibleTransitions:0,notVerifiable:0,effects:{},
    };
    const field=observation==='visible-transition'?'visibleTransitions':
      observation==='not-observed'?'noVisibleTransitions':'notVerifiable';
    totals.followedDecisions++;
    row.followedDecisions++;
    totals[field]++;
    row[field]++;
    const e=row.effects[effect]??={visibleTransitions:0,noVisibleTransitions:0,notVerifiable:0};
    e[field]++;
  }
  return totals;
}
