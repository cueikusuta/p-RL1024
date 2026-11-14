import { Match } from '../../../src/match.js';
import { SeededRandom } from '../../../src/battle.js';
import { validateOffer, budgetFor, speciesKey } from '../../../src/draft.js';
import { productionDependencies, playableSets } from '../../../src/production-runtime.js';
import { calculateChampionsStats } from '../../../src/champions-stats.js';
import { genderForSpeciesInstance } from '../../../src/pokemon-gender.js';
import data from '../../../data/catalog/battle-data.json' with {type:'json'};
import { chooseTeam, chooseSelection } from './draft-ai.js';
import { chooseBattleAction } from './battle-ai.js';
import { analyze } from './analysis.js';

// Independent deterministic streams: offers/battles are engine-owned;
// AI choice noise is derived from a disjoint seed for each stage/seat/round.
export function mixSeed(seed,...keys) {
  let s=(seed>>>0)||1;
  for(const n of keys) {
    s=(s^Math.imul((n>>>0)+1,0x9e3779b9))>>>0;
    s=Math.imul(s^(s>>>16),0x85ebca6b)>>>0;
    s=Math.imul(s^(s>>>13),0xc2b2ae35)>>>0;
    s=(s^(s>>>16))>>>0;
  }
  return s||1;
}
export function minimumRound(set) {
  // The normal R1 pool has no cost-6/7 slot; R1 Megas have max cost 3.
  if(set.cost===7)return 3; // R3 grant can afford one cost-7 plus five cost-1.
  if(set.mega&&set.cost>3)return 2;
  if(!set.mega&&set.cost>5)return 2;
  return 1;
}
export function resolveSet(setId) {
  const found=playableSets.find(p=>p.setId===setId);
  if(!found)throw new Error('Unknown / currently unplayable setId: '+setId+'. Use --list.');
  return found;
}
export function instantiate(set,round,seat) {
  const species=data.species[set.speciesId];
  const publicName=playableSets.find(s=>s.speciesId===set.speciesId&&!s.mega)?.speciesNameJa;
  if(!species||set.mega&&!publicName)throw new Error('Set lacks production species data: '+set.setId);
  return {...structuredClone(set),instanceId:'r'+round+'-p'+seat+'-'+set.setId,
    speciesName:set.speciesNameJa,publicSpeciesName:publicName??set.speciesNameJa,
    baseStats:structuredClone(species.baseStats),types:[...species.types],
    gender:genderForSpeciesInstance(species,'r'+round+'-p'+seat+'-'+set.setId),
    stats:calculateChampionsStats(species.baseStats,set.statPoints,data.natures[set.nature]),
    ...(set.mega?{megaPreview:{...data.species[set.mega.speciesId],
      speciesId:set.mega.speciesId,abilityId:set.mega.abilityId,
      stats:calculateChampionsStats(data.species[set.mega.speciesId].baseStats,set.statPoints,
        data.natures[set.nature])}}:{})};
}
function injectOffer(fresh,carry,round,seat,target,{forced=false}={}) {
  if(fresh.some(p=>p.setId===target.setId))return fresh;
  const eligible=fresh.filter(p=>p.cost===target.cost&&Boolean(p.mega)===Boolean(target.mega));
  if(!eligible.length)return null;
  const candidate=instantiate(target,round,seat);
  for(const old of eligible) {
    const changed=fresh.map(p=>p.instanceId===old.instanceId?candidate:p);
    try {
      validateOffer(changed,carry,round);
      if(forced) {
        // Conservative feasibility at the round's fresh grant; actual unused
        // credits can only increase it. A failed witness triggers retry.
        chooseTeam({round,self:{fresh:changed,carry,credits:budgetFor(round)}},
          {data,forcedSetId:target.setId});
      }
      return changed;
    } catch { /* Try another donor of the same cost and Mega classification. */ }
  }
  return null;
}
export function createMatch({seed,mode='control',target=null,targetSeat=0,targetRound=1,
  headlessPresentation=true}) {
  const baseline=productionDependencies(seed,{headlessPresentation});
  const audits=new Map();
  let battle=null,rerolls=0;
  const provider=args=>{
    const original=baseline.provider(args);
    let offered=original;
    let audit=baseline.provider.auditFor(args.seat);
    if(mode!=='control'&&args.round===targetRound&&args.seat===targetSeat &&
      !args.carry.some(p=>p.setId===target.setId)) {
      // An existing survivor with the same species or held item makes this
      // intervention impossible for this *entire match seed*. Re-rolling only
      // the current round's fresh offer cannot remove that carryover.
      if(args.carry.some(p=>(p.speciesClauseId??p.speciesId)===(target.speciesClauseId??target.speciesId)
        || (target.itemId && p.itemId===target.itemId)))
        throw new Error('Intervention carryover conflict for '+target.setId+
          '; retry the entire paired trial with a new deterministic seed');
      const maxAttempts=64;
      let changed=null;
      for(let attempt=0;attempt<maxAttempts;attempt++) {
        const attemptProvider=attempt===0?null:productionDependencies(mixSeed(seed,0x51a7,attempt)).provider;
        const fresh=attemptProvider?attemptProvider(args):original;
        changed=injectOffer(fresh,args.carry,args.round,args.seat,target,{forced:mode==='force'||mode==='play'});
        if(changed){
          if(attemptProvider)audit=attemptProvider.auditFor(args.seat);
          rerolls+=attempt;break;
        }
      }
      if(!changed)throw new Error('No legal offer for guaranteed '+target.setId+
        ' after '+maxAttempts+' attempts; change seed/round');
      offered=changed;
      audit={...audit,injected:target.setId,rerolls};
    }
    audits.set(args.seat,audit);
    return offered;
  };
  provider.auditFor=seat=>structuredClone(audits.get(seat)??null);
  const match=new Match(provider,{...baseline.options,
    battleFactory:(teams,meta)=>{
      battle=baseline.options.battleFactory(teams,meta);
      return battle;
    }});
  return {match,getBattle:()=>battle,getRerolls:()=>rerolls};
}
export function runOne({seed,trial=0,seat=0,mode='control',setId,round=1,maxDecisions=3000,onRound=null,onTiming=null,useCompactView=true,useCompactMatchView=true,
  useHeadlessPresentation=true}) {
  const matchStarted=performance.now();
  const target=resolveSet(setId);
  if(round<minimumRound(target))throw new Error('Set '+setId+' requires round '+minimumRound(target)+' or later');
  if(!Number.isInteger(round)||round>7)throw new Error('Round must be 1..7 in the current no-draw game');
  const {match,getBattle,getRerolls}=createMatch({seed,mode,target,targetSeat:seat,targetRound:round,
    headlessPresentation:useHeadlessPresentation});
  const matchView=s=>match.view(s,{includeBattle:!useCompactMatchView});
  match.ready(0);match.ready(1);
  const record={trial,seat,mode,seed,offered:false,drafted:false,selected:false,
    roundWon:false,survived:false,matchWon:false,reachedRound:false,deployed:false,rounds:0,rerolls:0};
  let final=null,battleSeconds=0;
  for(let roundCount=0;roundCount<64;roundCount++) {
    const publicViews=[matchView(0),matchView(1)];
    if(publicViews[0].phase==='complete')break;
    if(publicViews[0].phase!=='draft')throw new Error('Unexpected match phase '+publicViews[0].phase);
    const currentRound=publicViews[0].round;
    if(currentRound===round)record.reachedRound=true;
    if(currentRound===round)record.offered=[...publicViews[seat].self.fresh,...publicViews[seat].self.carry]
      .some(p=>p.setId===setId);
    const drafts=publicViews.map((view,s)=>chooseTeam(view,{data,
      forcedSetId:(mode==='force'||mode==='play')&&s===seat&&currentRound===round?setId:null,
      rng:new SeededRandom(mixSeed(seed,0xd0,s,currentRound))}));
    for(const s of [0,1])match.confirmTeam(s,drafts[s]);
    if(currentRound===round) {
      const ids=new Set(drafts[seat]);
      record.drafted=[...publicViews[seat].self.fresh,...publicViews[seat].self.carry]
        .some(p=>p.setId===setId&&ids.has(p.instanceId));
    }
    const selectionViews=[matchView(0),matchView(1)];
    const selections=selectionViews.map((view,s)=>chooseSelection(view,{
      data,sets:playableSets,rng:new SeededRandom(mixSeed(seed,0x50,s,currentRound)),
      forcedSetId:mode==='play'&&s===seat&&currentRound===round?setId:null,
      forceLead:mode==='play'&&s===seat&&currentRound===round,
      // Apply the same target-conditioned carryover eligibility to A/B/C/D,
      // preserving paired comparisons without violating production offer rules.
      excludedSpeciesKey:s===seat&&currentRound<round?speciesKey(target):null,
      excludedItemId:s===seat&&currentRound<round?target.itemId:null}));
    for(const s of [0,1])match.confirmSelection(s,selections[s]);
    if(currentRound===round) {
      record.selected=selections[seat].some(id=>
        selectionViews[seat].self.team.find(p=>p.instanceId===id)?.setId===setId);
      if(mode==='play') {
        if(!record.selected||selections[seat][0]!==selectionViews[seat].self.team.find(p=>p.setId===setId)?.instanceId)
          throw new Error('Forced-lead selection guarantee was violated');
        record.deployed=true;
      }
    }
    const battle=getBattle();
    const battleStarted=performance.now();
    if(!battle)throw new Error('Production battle was not created');
    const memories=[{},{}],streams=[0,1].map(s=>new SeededRandom(mixSeed(seed,0xb0,s,currentRound)));
    let decisions=0;
    // Avoid materializing the entire public Match/Battle view every action.
    while(!battle.result()) {
      let acted=false;
      for(const s of [0,1]) {
        if(battle.result())break;
        const view=useCompactView?battle.decisionViewForAI(s):battle.decisionView(s);
        const action=chooseBattleAction(view,{data,rng:streams[s],memory:memories[s]});
        if(action){match.battleAction(s,action,currentRound,view.request);acted=true;decisions++;}
      }
      if(!acted)throw new Error('Battle stalled with no legal action, round '+currentRound);
      if(decisions>maxDecisions)throw new Error('Battle exceeded decision limit '+maxDecisions+
        ' at round '+currentRound);
    }
    battleSeconds+=(performance.now()-battleStarted)/1000;
    const after=matchView(seat),last=after.history.at(-1);
    final=after;
    if(currentRound===round) {
      record.roundWon=last.winner===seat;
      record.survived=record.selected&&after.self.carry.some(p=>p.setId===setId);
    }
    record.rounds++;
    if(onRound)onRound({round:currentRound,winner:last.winner,
      // Preserve lead/bench ordering; team ordering is not battle order.
      selected:selections.map(ids=>[...ids]),
      carries:[0,1].map(s=>(s===seat?after:matchView(s)).self.carry.map(p=>p.instanceId)),
      battleRecord:battle.record(),outcome:battle.result()});
    if(after.phase==='complete')break;
    if(after.phase!=='result')throw new Error('Unexpected post-battle phase '+after.phase);
    match.readyNext(0,currentRound);match.readyNext(1,currentRound);
  }
  if(final?.phase!=='complete')throw new Error('Match did not finish within 64 rounds');
  record.matchWon=final.score[seat]===4;
  record.rerolls=getRerolls();
  if(onTiming){const elapsedSeconds=(performance.now()-matchStarted)/1000;
    onTiming({elapsedSeconds,battleSeconds,otherSeconds:elapsedSeconds-battleSeconds,
      rounds:record.rounds});}
  return record;
}
export function runExperiment({setId,trials=1,seed=1,round=1,mode='compare',progress=null}={}) {
  const set=resolveSet(setId);
  const minimum=minimumRound(set);
  if(round<minimum)throw new Error('This set cannot be tested before round '+minimum);
  if(!Number.isInteger(round)||round>7)throw new Error('Round must be 1..7 in the current no-draw game');
  const modes=mode==='compare'?['control','offer','force']:
    mode==='compare-all'?['control','offer','force','play']:[mode];
  if(!modes.every(m=>['control','offer','force','play'].includes(m)))throw new Error('Invalid experiment mode');
  if(!Number.isSafeInteger(trials)||trials<1||!Number.isInteger(seed)||seed<1||seed>0xffffffff)
    throw new Error('Invalid trial count or unsigned nonzero seed');
  const records=[];
  const started=performance.now(),total=trials*2*modes.length;
  let trialSeedRerolls=0;
  for(let trial=0;trial<trials;trial++) {
    // All arms AND both target seats must share one valid base match seed.
    // If a carried species/item makes an injected offer impossible, retry the
    // entire paired block, never only the treatment arm. No failed run is
    // counted in the rate denominator. Trial rerolls are reported explicitly.
    const maxTrialSeedAttempts=64;
    let trialRecords=null,lastRetryable=null;
    for(let attempt=0;attempt<maxTrialSeedAttempts;attempt++) {
      const gameSeed=attempt===0?mixSeed(seed,trial,0x9876)
        :mixSeed(seed,trial,0x9876,attempt);
      const candidate=[];
      try {
        for(const seat of [0,1])for(const experiment of modes)
          candidate.push(runOne({seed:gameSeed,trial,seat,mode:experiment,setId,round}));
        trialRecords=candidate;
        trialSeedRerolls+=attempt;
        break;
      } catch(error) {
        if(!/Intervention carryover conflict|No legal offer for guaranteed/.test(error.message)) {
          error.message+=' [setId='+setId+' trial='+trial+' seed='+gameSeed+']';
          throw error;
        }
        lastRetryable=error;
      }
    }
    if(!trialRecords)throw new Error('Could not find a valid paired match seed after '+
      maxTrialSeedAttempts+' attempts for '+setId+' trial='+trial+
      '; last error: '+lastRetryable?.message);
    records.push(...trialRecords);
    if(progress)progress({completed:records.length,total,
      matchesPerSecond:records.length/Math.max(0.001,(performance.now()-started)/1000),
      seconds:(performance.now()-started)/1000,trial});
  }
  const result=analyze({records,setId,setName:set.speciesNameJa,cost:set.cost,trials,seed,round});
  result.metadata.elapsedSeconds=(performance.now()-started)/1000;
  result.metadata.matchesPerSecond=total/Math.max(0.001,result.metadata.elapsedSeconds);
  result.metadata.interventionRerolls=records.reduce((sum,row)=>sum+row.rerolls,0);
  result.metadata.trialSeedRerolls=trialSeedRerolls;
  return result;
}
