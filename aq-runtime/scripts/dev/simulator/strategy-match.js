// I737 P3 opt-in Simulator runner. No change to the Basic/Intermediate defaults.
// Use only Match.view(seat) for draft/selection and the seat-local compact
// decision view for actions. Opponent truth is never passed to a strategy policy.
import {createMatch,mixSeed} from './runner.js';
import {SeededRandom} from '../../../src/battle.js';
import {playableSets} from '../../../src/production-runtime.js';
import {chooseTeam,chooseSelection} from './draft-ai.js';
import {chooseBattleAction} from './battle-ai.js';
import {chooseIntermediateAction} from './intermediate-ai.js';
import {chooseStrategyTeam,chooseStrategySelection} from '../../../src/ai/team-assessment.js';
import {chooseHybridSelection,normalizeHybridWeights} from '../../../src/ai/hybrid-selection.js';
import {chooseTacticalSelection} from '../../../src/ai/selection-tactical.js';
import {diagnoseSelectionDifference} from '../../../src/ai/hybrid-selection-diagnostics.js';
import {chooseStrategyAction} from '../../../src/ai/action-evaluator.js';
import {chooseIntegratedStrategyAction} from '../../../src/ai/macro-strategy.js';
import {V2_POLICIES,chooseStrategyV2Action,rankStrategyV2Actions,commitStrategyV2Choice,normalizeV2Weights} from '../../../src/ai/strategy-v2.js';
import {chooseFrozenAQBase,AQ_BASE_REPORT_SHA256} from './rl-linear-aq-base-opponent.js';
import {createI877AntiStall} from '../../../src/ai/v3-research-anti-stall.js';
import {assessCompactMatchup,assessMatchup} from '../../../src/ai/matchup-assessment.js';
import data from '../../../data/catalog/battle-data.json' with {type:'json'};
import {observeTacticalPublicEffect,summarizeTacticalEffectObservations}
  from './tactic-effect-observer.js';
import {createTacticCapabilityFunnel,recordTacticCapabilityRound,
  recordTacticCapabilityAction} from './tactic-capability-funnel.js';
import {createHumanQualityObserver,observeHumanQualityDecision,
  summarizeHumanQualityObservations} from './human-quality-observer.js';

const POLICY_IDS=new Set(['basic','intermediate','strategy','hybrid','strategy-hybrid','strategy-tactical','pure-rl-linear','aq-base',...V2_POLICIES]);
const v2Policy=p=>V2_POLICIES.includes(p);
const strategyPolicy=p=>p==='strategy'||p==='strategy-hybrid'||p==='strategy-tactical'||p==='aq-base'||v2Policy(p);
const hybridPolicy=p=>p==='hybrid'||p==='strategy-hybrid';
function summarizeTacticEvents(events) {
  const byPlan={};
  const ensure=id=>byPlan[id]??={
    feasibleDecisions:0,proposedDecisions:0,followedDecisions:0,
    startsFollowed:0,continuesFollowed:0,
    completedStateSignals:0,interruptedStateSignals:0,
  };
  let followedDecisions=0;
  for(const event of events) {
    for(const state of event.planStates) {
      const counts=ensure(state.id);
      if(state.feasible)counts.feasibleDecisions++;
      if(state.state==='completed')counts.completedStateSignals++;
      if(state.state==='interrupted')counts.interruptedStateSignals++;
    }
    if(event.proposed) {
      const counts=ensure(event.proposed.id);
      counts.proposedDecisions++;
      if(event.followed) {
        followedDecisions++;
        counts.followedDecisions++;
        if(event.proposed.state==='start')counts.startsFollowed++;
        if(event.proposed.state==='continue')counts.continuesFollowed++;
      }
    }
  }
  return {observedDecisions:events.length,followedDecisions,byPlan,
    caution:'Planner completed/interrupt states are NOT verified tactical success.'};
}
const percentile=(values,p)=>{
  if(!values.length)return null;
  const ordered=[...values].sort((a,b)=>a-b);
  return ordered[Math.ceil(p*ordered.length)-1];
};

// One seed produces one authoritative Match, including its draft, selection,
// battles and carryovers. maxRounds supports cheap contract pilots, while the
// default runs to the fourth victory. This is NOT the legacy strength model.
export function runStrategyMatch({seed=737,policies=['strategy','intermediate'],
  strategyMode='p2',maxRounds=64,maxDecisionsPerRound=3000,
  captureDecisionSamples=false,captureSelectionMetrics=false,
  captureTacticEffects=false,captureTacticFunnel=false,
  captureHumanQuality=false,capturePhaseTimings=false,captureV2Diagnostics=false,
  fastDraftValidation=true,hybridWeights={},v2Weights=null,
  pureRlSessions=null,pureRlAllPhases=false,
  aqBase=null,capturePureRlPhaseTimings=false}={}) {
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff)
    throw new TypeError('Expected a positive 32-bit seed');
  if(!Array.isArray(policies)||policies.length!==2||
     policies.some(policy=>!POLICY_IDS.has(policy)))
    throw new TypeError('Expected two known Basic/Intermediate/Strategy experiment policies');
  if(!['p2','integrated'].includes(strategyMode)||
     typeof captureDecisionSamples!=='boolean')
    throw new TypeError('Invalid strategy mode or timing capture');
  if(typeof captureSelectionMetrics!=='boolean'||
     typeof captureTacticEffects!=='boolean'||
     typeof captureTacticFunnel!=='boolean'||
     typeof captureHumanQuality!=='boolean'||typeof capturePhaseTimings!=='boolean'||
     typeof fastDraftValidation!=='boolean'||typeof captureV2Diagnostics!=='boolean')
    throw new TypeError('Invalid selection or tactic effect metrics flag');
  const normalizedWeights=normalizeHybridWeights(hybridWeights);
  if(!policies.some(hybridPolicy)&&Object.values(normalizedWeights).some(w=>w!==0))
    throw new TypeError('Hybrid weights supplied without a Hybrid policy');
  if((policies.some(v2Policy)||policies.includes('aq-base'))&&strategyMode!=='integrated')
    throw new TypeError('V2 requires integrated Strategy mode');
  if((policies.includes('strategy-v2-e')||policies.includes('aq-base'))&&!v2Weights)
    throw new TypeError('V2-E requires trained weights');
  if(v2Weights!==null){
    if(!policies.includes('strategy-v2-e')&&!policies.includes('aq-base'))
      throw new TypeError('V2-E weights supplied without V2-E policy');
    normalizeV2Weights(v2Weights);
  }
  const aqSeats=policies.map((p,i)=>p==='aq-base'?i:-1).filter(i=>i>=0);
  if(aqSeats.length>1||(aqSeats.length===1&&(
      aqBase?.reportSha256!==AQ_BASE_REPORT_SHA256||
      aqBase?.model?.featureCount!==61||
      typeof aqBase?.modelSha256!=='string'))||
      (aqSeats.length===0&&aqBase!==null))
    throw new TypeError('AQ-Base requires an original SHA-pinned frozen opponent');
  if(typeof capturePureRlPhaseTimings!=='boolean'||
    (capturePureRlPhaseTimings&&!pureRlAllPhases))
    throw new TypeError('Pure RL timing needs a full-phase policy');
  if(typeof pureRlAllPhases!=='boolean'||
    pureRlAllPhases&&!policies.includes('pure-rl-linear'))
    throw new TypeError('Full-phase RL requires independent pure RL policy');
  if(policies.includes('pure-rl-linear') && (
    !Array.isArray(pureRlSessions)||pureRlSessions.length!==2 ||
    policies.some((policy,seat)=>policy==='pure-rl-linear' && (
      !pureRlSessions[seat]||
      ['select','commit','finish','stats'].some(k=>
        typeof pureRlSessions[seat][k]!=='function')))))
    throw new TypeError('Pure RL requires a separate authorized session per seat');
  if(pureRlSessions!==null&&!policies.includes('pure-rl-linear'))
    throw new TypeError('Pure RL sessions supplied without pure RL policy');
  if(!Number.isSafeInteger(maxRounds)||maxRounds<1||maxRounds>64||
     !Number.isSafeInteger(maxDecisionsPerRound)||maxDecisionsPerRound<1)
    throw new TypeError('Invalid bounded match settings');

  // Timing is an opt-in experiment, not part of game decisions or RNG.
  // Aggregate phase durations only: never record private views, sets or actions.
  const phaseBegin=capturePhaseTimings?performance.now():0;
  const cpuBegin=capturePhaseTimings?process.cpuUsage():null;
  const phaseDurations=capturePhaseTimings?Object.create(null):null;
  const phaseCalls=capturePhaseTimings?Object.create(null):null;
  const measure=(name,fn)=>{
    if(!phaseDurations)return fn();
    const start=performance.now();
    try{return fn();}finally {
      phaseDurations[name]=(phaseDurations[name]??0)+(performance.now()-start);
      phaseCalls[name]=(phaseCalls[name]??0)+1;
    }
  };
  // Simulator-only default after same-seed full Match proof + measured speedup.
  // Production chooseStrategyTeam() keeps its legacy default false.
  const {match,getBattle}=measure('matchSetup',()=>
    createMatch({seed,mode:'control',headlessPresentation:true}));
  measure('matchReady',()=>{match.ready(0);match.ready(1);});
  const matchView=seat=>measure('matchView',()=>match.view(seat,{includeBattle:false}));
  const timings=[[],[]],actions=[0,0],roundWinners=[];
  const tacticEvents=[[],[]],tacticEffectEvents=[[],[]];
  const selectionTimes=[[],[]],selectionDiagnoses=[[],[]];
  const pureRlPhaseTimes=[0,1].map(()=>({draft:[],selection:[]}));
  const tacticFunnels=policies.map(policy=>captureTacticFunnel&&
    strategyPolicy(policy)?createTacticCapabilityFunnel():null);
  const qualityObservers=policies.map(policy=>captureHumanQuality&&
    strategyPolicy(policy)?createHumanQualityObserver():null);
  const aqBaseGuards=policies.map(p=>p==='aq-base'?createI877AntiStall({maxNonAttacks:5}):null);
  const v2Stats=policies.map(policy=>captureV2Diagnostics&&v2Policy(policy)
    ?{calls:0,changedDecisions:0,coveredSum:0}:null);
  let rounds=0;
  for(;rounds<maxRounds;rounds++) {
    const views=[matchView(0),matchView(1)];
    if(views[0].phase==='complete')break;
    if(views[0].phase!=='draft')throw new Error('Expected draft phase');
    const round=views[0].round;
    for(const seat of [0,1]) {
      const pureTiming=capturePureRlPhaseTimings&&
        policies[seat]==='pure-rl-linear';
      const draftStart=pureTiming?performance.now():0;
      const chosen=measure('draftChoice',()=>
        pureRlAllPhases&&policies[seat]==='pure-rl-linear'
          ?pureRlSessions[seat].select(views[seat])
          :strategyPolicy(policies[seat])
          ?chooseStrategyTeam(views[seat],{data,fastDraftValidation})
          :chooseTeam(views[seat],{data,rng:new SeededRandom(mixSeed(seed,0xd0,seat,round))}));
      if(pureTiming)pureRlPhaseTimes[seat].draft.push(performance.now()-draftStart);
      measure('draftCommit',()=>match.confirmTeam(seat,chosen));
      if(pureRlAllPhases&&policies[seat]==='pure-rl-linear')
        pureRlSessions[seat].commit();
    }
    const selectionViews=[matchView(0),matchView(1)];
    const selections=selectionViews.map((view,seat)=>{
      const policy=policies[seat],seedForSelection=mixSeed(seed,0x50,seat,round);
      const options={data,sets:playableSets,
        rng:new SeededRandom(seedForSelection)};
      const start=performance.now();
      const chosen=measure('selectionChoice',()=>
        pureRlAllPhases&&policy==='pure-rl-linear'
          ?pureRlSessions[seat].select(view)
          :(policy==='strategy-tactical'||policy==='aq-base'||v2Policy(policy))
          ?chooseTacticalSelection(view,{data,sets:playableSets})
        :strategyPolicy(policy)&&!hybridPolicy(policy)
          ?chooseStrategySelection(view,{catalog:playableSets,data})
          :hybridPolicy(policy)
          ?chooseHybridSelection(view,{...options,weights:normalizedWeights})
          :chooseSelection(view,options));
      const elapsed=performance.now()-start;
      if(capturePureRlPhaseTimings&&policy==='pure-rl-linear')
        pureRlPhaseTimes[seat].selection.push(elapsed);
      if(captureSelectionMetrics) {
        selectionTimes[seat].push(elapsed);
        // The independent Basic RNG does not consume the match policy's
        // RNG. Only anonymous aggregate classifications reach the report.
        measure('optionalSelectionDiagnosis',()=>{
          const baseline=chooseSelection(view,{data,sets:playableSets,
            rng:new SeededRandom(seedForSelection)});
          selectionDiagnoses[seat].push(diagnoseSelectionDifference(view,{
            basic:baseline,alternative:chosen,data,sets:playableSets,
          }));
        });
      }
      return chosen;
    });
    const ownSets=selectionViews.map((view,seat)=>selections[seat].map(id=>{
      const set=view.self.team.find(p=>p.instanceId===id);
      if(!set)throw new Error('Strategy selection omitted a confirmed own set');
      return set;
    }));
    if(captureTacticFunnel)for(const seat of [0,1]) {
      if(!tacticFunnels[seat])continue;
      recordTacticCapabilityRound(tacticFunnels[seat],{
        offered:[...(views[seat].self.fresh??[]),
          ...(views[seat].self.carry??[])],
        drafted:selectionViews[seat].self.team,
        selected:ownSets[seat],
        data,
      });
    }
    for(const seat of [0,1]){
      measure('selectionCommit',()=>match.confirmSelection(seat,selections[seat]));
      if(pureRlAllPhases&&policies[seat]==='pure-rl-linear')
        pureRlSessions[seat].commit();
    }
    const battle=getBattle();
    if(!battle)throw new Error('Authoritative battle was not created');
    const memories=[{},{}],beliefs=[null,null];
    const streams=[0,1].map(seat=>
      new SeededRandom(mixSeed(seed,0xb0,seat,round)));
    let decisions=0;
    // Public, aggregate action-kind counters only. This diagnostic never
    // stores private rosters, submitted move IDs or opponent hidden data.
    const actionKindCounts=[0,1].map(()=>({
      moves:0,voluntarySwitches:0,forcedSwitches:0,other:0,recent:[],
    }));
    while(!battle.result()) {
      let acted=false;
      const pendingTacticEffects=[];
      for(const seat of [0,1]) {
        if(battle.result())break;
        const view=measure('decisionViewForAI',()=>battle.decisionViewForAI(seat));
        const started=performance.now();
        let action=null,chosenPlanEvent=null,acceptedAQBaseCommit=null;
        const active=view.self.team?.[view.self.active];
        const canDecide=!view.self.confirmed&&
          !['wait','finished'].includes(view.requestType)&&
          !(view.requestType==='replacement'&&active?.hp>0&&!view.self.needsSwitch);
        if(policies[seat]==='pure-rl-linear'&&canDecide) {
          action=measure('pureRlDecision',()=>pureRlSessions[seat].select(view));
        } else if(strategyPolicy(policies[seat])&&canDecide) {
          // Resolve the caller's OWN selected set, never an opponent set.
          const ownSet=ownSets[seat][view.self.active];
          const evaluated=measure('publicBeliefMatchup',()=>assessCompactMatchup(view,{
            ownSet,catalog:playableSets,data,previousBelief:beliefs[seat],
          }));
          beliefs[seat]=evaluated.beliefState;
          const options={data,memory:memories[seat],
            matchup:evaluated.assessment,roundWins:views[seat].score[seat]};
          if(strategyMode==='integrated') {
            options.onTacticalChoice=event=>{
              tacticEvents[seat].push(event);
              chosenPlanEvent=event;
            };
            // Reuse one public opponent belief for own active and reserves.
            // Never use the opponent's hidden selection, team or true set.
            const ownMatchups=view.self.team.map((member,index)=>{
              if(member.hp<=0)return null;
              if(index===view.self.active)return evaluated.assessment;
              return measure('reserveMatchup',()=>assessMatchup({
                ownSet:ownSets[seat][index],own:member,
                opponentBelief:evaluated.beliefState.belief,
                opponentPublic:view.opponent,
                catalog:playableSets,data,field:view.field,
              }));
            });
            options.ownMatchups=ownMatchups;
            options.switchMatchups=ownMatchups;
          }
          if(v2Stats[seat])options.onV2Choice=event=>{
            v2Stats[seat].calls++;
            v2Stats[seat].changedDecisions+=Number(event.changed);
            v2Stats[seat].coveredSum+=event.covered;
          };
          if(policies[seat]==='aq-base'&&view.requestType==='turn'){
            const ranking=rankStrategyV2Actions(view,{...options,
              variant:'strategy-v2-e',catalog:playableSets,weights:v2Weights});
            const legal=ranking.rows.map(row=>row.action);
            const scores=ranking.rows.map(row=>row.score);
            const chosen=chooseFrozenAQBase(aqBase.model,view,legal,scores,{
              data,roundWins:views[seat].score[seat],
              roundLosses:views[seat].score[1-seat]});
            const index=aqBaseGuards[seat].choose(
              chosen.candidates,chosen.selectedIndex).index;
            action=legal[index];
            acceptedAQBaseCommit=()=>commitStrategyV2Choice(view,{
              ...options,variant:'strategy-v2-e'},ranking,action);
          }else action=measure('strategyDecision',()=>
            (v2Policy(policies[seat])||policies[seat]==='aq-base')
            ?chooseStrategyV2Action(view,{...options,
              variant:policies[seat]==='aq-base'?'strategy-v2-e':policies[seat],
              catalog:playableSets,
              ...(policies[seat]==='strategy-v2-e'||policies[seat]==='aq-base'
                ?{weights:v2Weights}:{})})
            :strategyMode==='integrated'
              ?chooseIntegratedStrategyAction(view,options)
              :chooseStrategyAction(view,options));
        } else if(policies[seat]==='intermediate') {
          action=measure('intermediateDecision',()=>chooseIntermediateAction(view,{
            data,rng:streams[seat],memory:memories[seat],
          }));
        } else if(policies[seat]==='basic'||policies[seat]==='hybrid') {
          action=measure('basicDecision',()=>chooseBattleAction(view,{
            data,rng:streams[seat],memory:memories[seat],
          }));
        }
        // Empty/wait decisions are not action thought-time samples.
        // Recording those near-zero calls biases p50/p95 downward.
        const elapsed=performance.now()-started;
        if(action) {
          if(captureTacticEffects&&strategyMode==='integrated'&&
             chosenPlanEvent?.followed&&chosenPlanEvent.proposed) {
            const moveId=action.kind==='move'&&typeof action.index==='number'
              ?active?.moves?.[action.index]?.id:null;
            pendingTacticEffects.push({
              seat,planId:chosenPlanEvent.proposed.id,
              action,moveId,before:view,
            });
          }
          timings[seat].push(elapsed);
          // Match delegates legality to the authoritative Battle engine.
          measure('battleTransition',()=>
            match.battleAction(seat,action,round,view.request));
          const audit=actionKindCounts[seat];
          const code=action.kind==='move'?'M'
            :action.kind==='switch'&&view.requestType==='turn'?'S'
            :action.kind==='switch'?'F':'O';
          if(code==='M')audit.moves++;
          else if(code==='S')audit.voluntarySwitches++;
          else if(code==='F')audit.forcedSwitches++;
          else audit.other++;
          audit.recent.push(code);
          if(audit.recent.length>12)audit.recent.shift();
          if(acceptedAQBaseCommit)acceptedAQBaseCommit();
          // Only accepted actions enter the pure RL SARSA trajectory.
          if(policies[seat]==='pure-rl-linear')pureRlSessions[seat].commit();
          // Count ACCEPTED commands, not attempts: the authoritative Match
          // must accept the action before it can enter the observation funnel.
          if(tacticFunnels[seat]) {
            const moveId=action.kind==='move'&&Number.isInteger(action.index)
              ?active?.moves?.[action.index]?.id:null;
            recordTacticCapabilityAction(tacticFunnels[seat],{
              activeSet:ownSets[seat][view.self.active],
              action,moveId,data,requestType:view.requestType,view,
            });
          }
          if(qualityObservers[seat]){
            const moveId=action.kind==='move'&&Number.isInteger(action.index)
              ?active?.moves?.[action.index]?.id:null;
            observeHumanQualityDecision(qualityObservers[seat],{
              view,action,moveId,
            });
          }
          actions[seat]++;
          decisions++;
          acted=true;
        }
      }
      // Compare only public compact snapshots after both seats submitted.
      // A visible transition is NOT an end-to-end tactic success or proof of
      // which actor caused it. The opt-in path never changes Battle behavior.
      for(const event of pendingTacticEffects) {
        const after=measure('optionalEffectView',()=>battle.decisionViewForAI(event.seat));
        const finding=measure('optionalEffectDiagnosis',()=>
          observeTacticalPublicEffect({...event,after}));
        tacticEffectEvents[event.seat].push({planId:event.planId,...finding});
      }
      if(!acted)throw new Error('Match stalled without legal action in round '+round);
      if(decisions>maxDecisionsPerRound)
        throw new Error('Decision cap exceeded in round '+round+
          ' [public action-kind summary; M=move S=voluntary switch F=forced switch] '+
          JSON.stringify({decisions,perSeat:actionKindCounts}));
    }
    const after=matchView(0);
    const last=after.history.at(-1);
    if(!last||![0,1].includes(last.winner))
      throw new Error('Expected a resolved non-draw battle');
    roundWinners.push(last.winner);
    if(after.phase==='complete') {
      rounds++;
      break;
    }
    if(after.phase!=='result')throw new Error('Unexpected round phase '+after.phase);
    if(rounds+1<maxRounds) {
      measure('nextRoundReady',()=>{
        match.readyNext(0,round);
        match.readyNext(1,round);
      });
    }
  }
  const final=matchView(0);
  const completed=final.phase==='complete';
  if(maxRounds===64&&!completed)throw new Error('Match did not finish in 64 rounds');
  const pureRlStats=policies.map((policy,seat)=>policy==='pure-rl-linear'
    ?completed?pureRlSessions[seat].finish(final.score.indexOf(4)===seat?1:-1)
      :pureRlSessions[seat].stats()
    :null);
  return {
    schemaVersion:1,kind:'i737-strategy-match-pilot',
    seed,policies:[...policies],strategyMode,completed,rounds,
    score:[...final.score],winner:completed?final.score.indexOf(4):null,
    roundWinners,actionCounts:actions,
    ...(policies.includes('pure-rl-linear')?{pureRlStats}:{}),
    ...(capturePureRlPhaseTimings?{
      pureRlPhaseTimeMs:pureRlPhaseTimes.map(stages=>Object.fromEntries(
        Object.entries(stages).map(([name,samples])=>[name,{
          calls:samples.length,samples:[...samples],
          p50:percentile(samples,0.5),p95:percentile(samples,0.95),
        }]))),
    }:{}),
    ...(captureSelectionMetrics?{
      selectionTimeMs:selectionTimes.map(samples=>({
        calls:samples.length,p50:percentile(samples,0.5),
        p95:percentile(samples,0.95),
        // Per-Match samples are used only to pool measurement-time summaries.
        samples:[...samples],
      })),
      selectionDiagnostics:selectionDiagnoses.map(rows=>({
        rounds:rows.length,trioChanged:rows.filter(r=>r.trioChanged).length,
        leadChanged:rows.filter(r=>r.leadChanged).length,
        benchChanged:rows.filter(r=>r.benchChanged).length,
        unknownFoeRounds:rows.filter(r=>r.unknownPublicSpecies>0).length,
        byPotentialCause:Object.fromEntries([...new Set(
          rows.flatMap(r=>r.possibleCauses))].map(cause=>[
          cause,rows.filter(r=>r.possibleCauses.includes(cause)).length,
        ])),
      })),
    }:{}),
    tacticalPlanStats:tacticEvents.map((events,seat)=>
      strategyMode==='integrated'&&strategyPolicy(policies[seat])
        ?summarizeTacticEvents(events):null),
    ...(captureTacticFunnel?{tacticCapabilityFunnel:tacticFunnels}:{}),
    ...(captureHumanQuality?{humanQualityStats:qualityObservers.map(row=>
      row?summarizeHumanQualityObservations(row):null)}:{}),
    ...(captureV2Diagnostics?{v2Diagnostics:v2Stats.map(row=>row
      ?{calls:row.calls,changedDecisions:row.changedDecisions,
        meanHypothesisCoverage:row.calls>0?row.coveredSum/row.calls:0}:null)}:{}),
    ...(captureTacticEffects?{
      tacticalEffectStats:tacticEffectEvents.map((events,seat)=>
        strategyMode==='integrated'&&strategyPolicy(policies[seat])
          ?summarizeTacticalEffectObservations(events):null),
    }:{}),
    decisionTimeMs:timings.map(t=>({
      calls:t.length,p50:percentile(t,0.5),p95:percentile(t,0.95),
      maximum:t.length?Math.max(...t):null,
      ...(captureDecisionSamples?{samples:[...t]}:{}),
    })),
    ...(capturePhaseTimings?{phaseProfile:{
      schemaVersion:1,
      unit:'full-four-win-Match',
      totalWallMs:performance.now()-phaseBegin,
      cpuMs:Object.fromEntries(Object.entries(process.cpuUsage(cpuBegin)).map(
        ([name,microseconds])=>[name,microseconds/1000])),
      heapUsedMiB:process.memoryUsage().heapUsed/1048576,
      phases:Object.fromEntries(Object.keys(phaseDurations).sort().map(name=>[
        name,{calls:phaseCalls[name],wallMs:phaseDurations[name]},
      ])),
      // Exclusive timing for each phase; nested calls are not double-counted.
      // Unclassified overhead = loop control/validation/summary + timer work.
    }}:{}),
  };
}
