// Independent opt-in Match runner for heterogeneous phase AIs.
// The provenance-pinned I975/I737 runners and their model contracts stay intact.
import {createMatch,mixSeed} from './runner.js';
import {SeededRandom} from '../../../src/battle.js';
import {createPhasePipeline} from '../../../src/ai/phase-pipeline.js';
// Nonforgeable, one-use receipts are minted only following Match acceptance.
// No public receipt factory: generic collectors cannot self-certify proposals.
const authorityReceipts=new WeakMap();
function receipt(kind,seat,round,action){
  const opaque=Object.freeze({});
  authorityReceipts.set(opaque,{kind,seat,round,action:JSON.stringify(action)});
  return opaque;
}
export function consumePhaseAuthorityReceipt(token,{kind,seat,round,action}){
  const expected=authorityReceipts.get(token);
  if(!expected||expected.kind!==kind||expected.seat!==seat||
     expected.round!==round||expected.action!==JSON.stringify(action))
    throw new Error('Missing authoritative Match acceptance receipt');
  authorityReceipts.delete(token);
  return true;
}

export function runPhaseMatch({seed=737,agents,maxRounds=64,
  maxDecisionsPerRound=3000}={}){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
    !Number.isSafeInteger(maxRounds)||maxRounds<1||maxRounds>64||
    !Number.isSafeInteger(maxDecisionsPerRound)||maxDecisionsPerRound<1||
    !Array.isArray(agents)||agents.length!==2)
    throw new TypeError('Invalid bounded phase Match configuration');
  const pipelines=agents.map((definition,seat)=>
    createPhasePipeline(definition,{seat}));
  const {match,getBattle}=createMatch({seed,mode:'control',
    headlessPresentation:true});
  const view=seat=>match.view(seat,{includeBattle:false});
  match.ready(0);
  match.ready(1);
  const roundWinners=[],actionCounts=[0,0];
  let rounds=0;
  for(;rounds<maxRounds;rounds++){
    const drafts=[view(0),view(1)];
    if(drafts[0].phase==='complete')break;
    if(drafts[0].phase!=='draft')throw new Error('Expected draft');
    const round=drafts[0].round;
    for(const seat of [0,1]){
      const rng=new SeededRandom(mixSeed(seed,0xd0,seat,round));
      const decision=pipelines[seat].select('draft',drafts[seat],{round,rng});
      match.confirmTeam(seat,decision);
      pipelines[seat].accepted('draft',drafts[seat],decision,
        receipt('draft',seat,round,decision));
    }
    const selections=[view(0),view(1)];
    for(const seat of [0,1]){
      const rng=new SeededRandom(mixSeed(seed,0x50,seat,round));
      const decision=pipelines[seat].select('selection',selections[seat],{round,rng});
      match.confirmSelection(seat,decision);
      pipelines[seat].accepted('selection',selections[seat],decision,
        receipt('selection',seat,round,decision));
    }
    const battle=getBattle();
    if(!battle)throw new Error('Battle not created');
    const streams=[0,1].map(seat=>
      new SeededRandom(mixSeed(seed,0xb0,seat,round)));
    let decisions=0;
    while(!battle.result()){
      let acted=false;
      for(const seat of [0,1]){
        if(battle.result())break;
        const observation=battle.decisionViewForAI(seat);
        const active=observation.self.team?.[observation.self.active];
        const canDecide=!observation.self.confirmed&&
          !['wait','finished'].includes(observation.requestType)&&
          !(observation.requestType==='replacement'&&
            active?.hp>0&&!observation.self.needsSwitch);
        if(!canDecide)continue;
        const action=pipelines[seat].select('battle',observation,
          {round,rng:streams[seat]});
        // Authoritative validation comes before either the cross-AI handoff
        // or a learning session's accepted-action commit.
        match.battleAction(seat,action,round,observation.request);
        pipelines[seat].accepted('battle',observation,action,
          receipt('battle',seat,round,action));
        actionCounts[seat]++;
        decisions++;
        acted=true;
      }
      if(!acted)throw new Error('Battle stalled in round '+round);
      if(decisions>maxDecisionsPerRound)
        throw new Error('Decision cap exceeded in round '+round);
    }
    const after=view(0);
    const winner=after.history.at(-1)?.winner;
    if(![0,1].includes(winner))
      throw new Error('Expected a resolved non-draw battle');
    roundWinners.push(winner);
    for(const seat of [0,1])pipelines[seat].roundEnd(round,winner);
    if(after.phase==='complete'){rounds++;break;}
    if(after.phase!=='result')throw new Error('Expected round result');
    if(rounds+1<maxRounds){
      match.readyNext(0,round);
      match.readyNext(1,round);
    }
  }
  const final=view(0),completed=final.phase==='complete';
  if(maxRounds===64&&!completed)
    throw new Error('Match did not finish in 64 rounds');
  const winner=completed?final.score.indexOf(4):null;
  const outcome={completed,winner,score:[...final.score],rounds};
  return {schema:'p-rl1024-phase-match-v1',seed,...outcome,
    roundWinners,actionCounts,
    agentIds:pipelines.map(pipeline=>pipeline.ids),
    agentStats:pipelines.map((pipeline,seat)=>pipeline.finish(outcome,
      completed?receipt('finish',seat,rounds,{winner,score:outcome.score}):null))};
}
