import {Battle,SeededRandom} from '../../aq-runtime/src/battle.js';
import {playableSets} from '../../aq-runtime/src/production-runtime.js';
import {chooseBattleAction} from '../../aq-runtime/scripts/dev/simulator/battle-ai.js';
import {instantiate,mixSeed} from '../../aq-runtime/scripts/dev/simulator/runner.js';
import data from '../../aq-runtime/data/catalog/battle-data.json' with {type:'json'};

const setById=new Map(playableSets.map(set=>[set.setId,set]));
const publicNames=new Map(playableSets.filter(set=>!set.mega)
  .map(set=>[set.speciesId,set.speciesNameJa]));
const byName=new Map();
for(const set of playableSets){
  const name=publicNames.get(set.speciesId);
  if(!name)continue;
  if(!byName.has(name))byName.set(name,[]);
  byName.get(name).push(set);
}
const SELECT_WORLDS=3, MAX_CANDIDATES=3, ACTION_CAP=140;
const seedsFor=obs=>[obs.round,obs.seat,...obs.opponent.species
  .flatMap(x=>Array.from(x,c=>c.codePointAt(0)))];
function publicWorlds(obs){
  const names=obs.opponent.species;
  if(names.length!==6||new Set(names).size!==6)return [];
  const seed=mixSeed(955143,...seedsFor(obs));
  const options=names.map(name=>byName.get(name)??[]);
  if(options.some(sets=>!sets.length))return [];
  // Sample distinct trinities from the six public species and only catalog
  // variants. Preserve uniqueness of held items in each hypothesized trio.
  const worlds=[],seen=new Set();
  for(let attempt=0;attempt<20&&worlds.length<SELECT_WORLDS;attempt++){
    const rng=new SeededRandom(mixSeed(seed,attempt));
    const pool=options.slice(),picked=[],items=new Set();
    while(picked.length<3){
      const [row]=pool.splice(rng.integer(pool.length),1);
      const candidates=row.filter(s=>!s.itemId||!items.has(s.itemId));
      if(!candidates.length)break;
      const set=candidates[rng.integer(candidates.length)];
      picked.push(set);
      if(set.itemId)items.add(set.itemId);
    }
    if(picked.length!==3)continue;
    const key=picked.map(p=>p.setId).join('|');
    if(seen.has(key))continue;
    seen.add(key);
    worlds.push({sets:picked,seed:mixSeed(seed,1000+worlds.length)});
  }
  return worlds;
}
function finishBattle(own,foe,seed,metrics){
  const b=new Battle([own,foe],seed,{data,headlessPresentation:true});
  const memories=[{},{}];
  const streams=[0,1].map(seat=>new SeededRandom(mixSeed(seed,seat)));
  while(!b.result()&&metrics.remaining>0){
    let acted=false;
    for(const seat of [0,1]){
      if(b.result()||metrics.remaining<=0)break;
      const view=b.decisionViewForAI(seat);
      if(view.requestType==='revival')return null;
      const next=chooseBattleAction(view,{data,rng:streams[seat],memory:memories[seat]});
      if(!next)continue;
      // The frozen rollout policy has no Mega decision. This is intentionally
      // a model of the round, not an assertion about real G421 win odds.
      b.submit(seat,next,view.request);
      metrics.remaining--;
      acted=true;
    }
    if(!acted)break;
  }
  const r=b.result();
  return r?.winner===0?1:r?.winner===1?0:null;
}
export function createSelectionReranker(metrics){
  return ({observation:obs,legalActions,logits,baseline})=>{
    metrics.calls++;
    const ownMap=new Map(obs.self.team.map(p=>[p.instanceId,setById.get(p.setId)]));
    if(ownMap.size!==6||[...ownMap.values()].some(s=>!s))return baseline;
    const worlds=publicWorlds(obs);
    if(worlds.length!==SELECT_WORLDS)return baseline;
    const ranks=logits.map((x,i)=>i).sort((a,b)=>logits[b]-logits[a]||a-b).slice(0,MAX_CANDIDATES);
    if(ranks[0]!==baseline)throw Error('Not a frozen greedy selection shortlist');
    const opponents=worlds.map(w=>w.sets.map(p=>instantiate(p,obs.round,1)));
    let best=baseline,bestWins=-1;
    for(const i of ranks){
      let wins=0;
      const own=legalActions[i].map(id=>instantiate(ownMap.get(id),obs.round,0));
      for(let j=0;j<worlds.length;j++){
        const budget={remaining:ACTION_CAP};
        const outcome=finishBattle(own,opponents[j],worlds[j].seed,budget);
        metrics.simulations++;
        metrics.simDecisions+=ACTION_CAP-budget.remaining;
        if(outcome===null){metrics.inconclusive++;return baseline;}
        wins+=outcome;
      }
      if(wins>bestWins){best=i;bestWins=wins;}
    }
    metrics.scored++;
    if(best!==baseline)metrics.changed++;
    return best;
  };
}
