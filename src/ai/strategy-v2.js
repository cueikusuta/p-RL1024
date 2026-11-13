// I871 experimental CPU battle policies. No hidden Battle state, no opponent
// checkpoint, no synthetic true opposing selection and no change to rules.
// A: bounded one-turn model; B: public action scenarios; C: soft belief;
// D: conservative continuation; E: offline self-play tuned scalar weights.
import {calculateChampionsStats} from '../champions-stats.js';
import {fieldSpeed} from '../battle-field.js';
import {estimateAttackPercent} from './matchup-assessment.js';
import {publicMoveDamage} from './action-evaluator.js';
import {rankIntegratedStrategyActions} from './macro-strategy.js';

export const V2_POLICIES=Object.freeze([
  'strategy-v2-a','strategy-v2-b','strategy-v2-c',
  'strategy-v2-d','strategy-v2-e',
]);
export const V2_DEFAULT_WEIGHTS=Object.freeze({
  blend:0.28,exposure:0.75,continuation:0.22,
});
const LIMITS=Object.freeze({ownActions:6,opponentHypotheses:8,opponentMoves:4});
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const clamp=(n,lo,hi)=>Math.min(hi,Math.max(lo,n));
const health=p=>100*p.hp/Math.max(1,p.maxHP);
const signal=(row,key)=>row?.modelVersion==='matchup-p1-v1'&&
  row[key]?.status==='evaluated'&&finite(row[key].score)
  ?clamp(row[key].score,-1,1):null;
const accuracy=m=>m?.accuracy===true?1:finite(m?.accuracy)
  ?clamp(m.accuracy/100,0,1):1;
const actionKey=(action,view)=>action.kind==='switch'
  ?'switch:'+action.index:action.kind==='move'&&typeof action.index==='number'
    ?'move:'+(view.self.team[view.self.active].moves[action.index]?.id??'?')
    :'move:struggle';

// Return order only when the public field and a catalog hypothesis permit a
// reliable comparison. Never equate a guessed speed with a known turn order.
function expectedAttackOrder(view,own,foe,ownMove,foeMove) {
  if(!ownMove||!foeMove)return null;
  const dynamic=new Set(['unburden','quickdraw','galewings','triage']);
  if([own,foe].some(p=>dynamic.has(p.abilityId)||
     ['quickclaw','custapberry'].includes(p.itemId)))return null;
  const ownPriority=finite(ownMove.priority)?ownMove.priority:0;
  const foePriority=finite(foeMove.priority)?foeMove.priority:0;
  if(ownPriority!==foePriority)return ownPriority>foePriority?'own':'foe';
  const sideEffects=view.field?.sideEffects;
  if(!Array.isArray(sideEffects)||sideEffects.length!==2||
     !sideEffects.every(Array.isArray))return null;
  const weatherSpeed={chlorophyll:'sunnyday',sandrush:'sandstorm',
    swiftswim:'raindance',slushrush:'snowscape'};
  if([own,foe].some(p=>weatherSpeed[p.abilityId]&&
     (weatherSpeed[p.abilityId]===view.field.weather||
       p.abilityId==='slushrush'&&view.field.weather==='hail'))&&
     typeof view.field.weatherSuppressed!=='boolean')return null;
  if(!finite(own.stats?.spe)||!finite(foe.stats?.spe))return null;
  const field={...view.field,
    weatherSuppressed:view.field.weatherSuppressed===true,
    sides:sideEffects.map(rows=>Object.fromEntries(rows
      .filter(effect=>typeof effect?.id==='string')
      .map(effect=>[effect.id,effect])))};
  const a=fieldSpeed(own,field,0),b=fieldSpeed(foe,field,1);
  if(!finite(a)||!finite(b)||a===b)return null; // speed tie is random
  const trickRoom=(view.field.trickRoomTurns??0)>0;
  return (trickRoom?a<b:a>b)?'own':'foe';
}

export function normalizeV2Weights(weights={}) {
  if(!weights||typeof weights!=='object'||Array.isArray(weights)||
     Object.keys(weights).some(k=>!(k in V2_DEFAULT_WEIGHTS)))
    throw new TypeError('Invalid experimental v2 weights');
  const values={...V2_DEFAULT_WEIGHTS,...weights};
  for(const [key,lo,hi] of [['blend',0,0.65],['exposure',0,2],
    ['continuation',0,0.8]]) {
    if(!finite(values[key])||values[key]<lo||values[key]>hi)
      throw new TypeError('Out-of-range v2 weight '+key);
  }
  return values;
}

function hypotheses(view,options,tier) {
  const catalog=options.catalog;
  const belief=options.matchup?.opponentBelief;
  if(!Array.isArray(catalog)||!Array.isArray(belief?.hypotheses)||
     belief.unknown)return [];
  const byId=new Map(catalog.map(set=>[set.setId,set]));
  const observations=belief.observations??{};
  // A partially malformed public adapter must not sneak a nonmatching move
  // into a catalog hypothesis. Conflicting confirmed facts fail closed.
  if(observations.usedMoves!==undefined&&observations.usedMoves!==null&&
     (!Array.isArray(observations.usedMoves)||
      observations.usedMoves.some(id=>typeof id!=='string')))return [];
  if(['confirmedOriginalItemId','confirmedBaseAbilityId'].some(key=>
     observations[key]!=null&&typeof observations[key]!=='string'))
    return [];
  if(Array.isArray(observations.conflictingConfirmedFields)&&
     observations.conflictingConfirmedFields.length)return [];
  const usedMoves=observations.usedMoves??[];
  const rows=[];
  for(const h of belief.hypotheses) {
    if(!(h.weight>0)||!finite(h.weight))continue;
    const set=byId.get(h.setId);
    if(!set||!Array.isArray(set.moves)||!set.moves.length||
       usedMoves.some(id=>!set.moves.includes(id))||
       (typeof observations.confirmedOriginalItemId==='string'&&
         observations.confirmedOriginalItemId!==set.itemId)||
       (typeof observations.confirmedBaseAbilityId==='string'&&
         observations.confirmedBaseAbilityId!==set.abilityId))continue;
    const speciesId=view.opponent.speciesId===set.mega?.speciesId
      ?set.mega.speciesId:set.speciesId;
    const base=options.data.species?.[speciesId];
    if(!base?.baseStats)continue;
    const actor={
      stats:calculateChampionsStats(base.baseStats,set.statPoints??{},
        options.data.natures?.[set.nature]??{}),
      types:Array.isArray(view.opponent.types)&&view.opponent.types.length
        ?view.opponent.types:base.types,
      boosts:view.opponent.stages??{},status:view.opponent.status??'',
      itemId:set.itemId,abilityId:speciesId===set.mega?.speciesId
        ?set.mega.abilityId:set.abilityId,
      moves:set.moves.slice(0,LIMITS.opponentMoves),
    };
    // C: bounded likelihood reweighting of public observations. It NEVER
    // eliminates a surviving catalog hypothesis and is not calibrated.
    let weight=h.weight;
    if(tier>=3) {
      // Public matchup context supplies a weak selection prior even before
      // a move is revealed. This is a HYPOTHESIS, not an observed fact.
      const pressures=actor.moves.map(id=>{
        const move=options.data.moves[id];
        return move?estimateAttackPercent(actor,
          view.self.team[view.self.active],move,options.data,view.field):null;
      }).filter(finite);
      if(pressures.length)weight*=0.8+0.4*
        clamp(Math.max(0,...pressures)/80,0,1);
      // If an authorized public-history adapter supplies observed moves,
      // reweight softly by their plausible utility for this candidate.
      if(usedMoves.length){
        const observedPressure=usedMoves.reduce((sum,id)=>{
          const move=options.data.moves[id];
          const damage=move?estimateAttackPercent(actor,
            view.self.team[view.self.active],move,options.data,view.field):null;
          return sum+(damage===null?0.2:clamp(damage/100,0,1));
        },0)/usedMoves.length;
        weight*=0.75+0.5*observedPressure;
      }
      // Do not prune a valid catalog hypothesis using these soft priors.
    }
    rows.push({actor,weight,setId:set.setId});
  }
  rows.sort((a,b)=>b.weight-a.weight||
    a.setId.localeCompare(b.setId,'en'));
  const chosen=rows.slice(0,LIMITS.opponentHypotheses);
  // The omitted mass is explicitly left unknown; do not re-normalize it as
  // if the selected hypotheses exhaust all possible opposing sets.
  const total=rows.reduce((sum,r)=>sum+r.weight,0);
  return total>0?chosen.map(h=>({...h,weight:h.weight/total})):[];
}

function opponentOptions(h,view,data,tier) {
  const own=view.self.team[view.self.active];
  const attacks=h.actor.moves.map(id=>{
    const move=data.moves[id];
    const percent=move?estimateAttackPercent(h.actor,own,move,data,view.field):null;
    return {id,percent,accuracy:accuracy(move),category:move?.category};
  }).filter(row=>row.percent!==null&&row.percent>0)
    .sort((a,b)=>b.percent*b.accuracy-a.percent*a.accuracy);
  const best=attacks[0];
  // If any plausible damaging move is not modelled, a known weak move must
  // not be mistaken for the opponent's maximum damage. Preserve uncertainty.
  const unmodelledAttack=h.actor.moves.some(id=>{
    const move=data.moves[id];
    return move&&move.category!=='Status'&&
      estimateAttackPercent(h.actor,own,move,data,view.field)===null;
  });
  if(unmodelledAttack||!best)
    return [{kind:'unknown',prob:1,percent:0}];
  if(tier===1)return [{kind:'attack',prob:1,percent:best.percent*
    best.accuracy,rawPercent:best.percent,moveAccuracy:best.accuracy,
    moveId:best.id}];
  // B+: deterministic, bounded scenarios for attack, setup and a possible
  // retreat. The retreat is a generic unknown target, never a secret bench.
  const hasSetup=h.actor.moves.some(id=>{
    const m=data.moves[id];
    return m?.category==='Status'&&
      (m.self?.boosts||m.boosts&&m.target==='self');
  });
  const canSwitch=view.opponent.remaining>1;
  const pressure=best?clamp(best.percent*best.accuracy,0,150):0;
  let attack=best?0.75:0.3;
  const setup=hasSetup?0.13:0;
  const switchChance=canSwitch?0.12:0;
  if(!best)attack=1-setup-switchChance;
  const total=attack+setup+switchChance;
  return [
    {kind:'attack',prob:attack/total,percent:pressure,
      rawPercent:best.percent,moveAccuracy:best.accuracy,moveId:best.id},
    ...(setup?[{kind:'setup',prob:setup/total,percent:0}]:[]),
    ...(switchChance?[{kind:'switch',prob:switchChance/total,percent:0}]:[]),
  ];
}

function ownAttackPercent(row,view,options) {
  if(row.action.kind!=='move'||typeof row.action.index!=='number')return null;
  const id=view.self.team[view.self.active].moves[row.action.index]?.id;
  const forecast=options.matchup?.moveEstimates?.find(m=>m.id===id);
  const expected=finite(forecast?.expectedPercent)?
    forecast.expectedPercent:publicMoveDamage(
      view.self.team[view.self.active],view.opponent,
      options.data.moves[id],options.data,view.field);
  return expected===null||!finite(expected)?null:
    clamp(expected*accuracy(options.data.moves[id]),0,300);
}

function project(row,view,options,hypothesesBySet,tier,weights) {
  const own=view.self.team[view.self.active],selfHP=health(own);
  const reserve=row.action.kind==='switch'
    ?view.self.team[row.action.index]:null;
  const ownDamage=ownAttackPercent(row,view,options);
  // Unsupported moves are evaluated by the existing Integrated strategy.
  const isAttack=ownDamage!==null;
  const ownId=isAttack?own.moves[row.action.index]?.id:null;
  const ownMove=ownId?options.data.moves[ownId]:null;
  const forecast=options.matchup?.modelVersion==='matchup-p1-v1'
    ?options.matchup.moveEstimates?.find(m=>m.id===ownId):null;
  // KO is a certainty bonus only with the conservative minimum forecast.
  const secureOwnKO=!!ownMove&&own.status!=='slp'&&own.status!=='frz'&&
    finite(forecast?.minPercent)&&forecast.minPercent>=
      view.opponent.hpPercent*1.12&&
    forecast.immuneCandidateWeight===0&&accuracy(ownMove)===1;
  const foeHP=view.opponent.hpPercent;
  let utility=0,covered=0;
  for(const h of hypothesesBySet) {
    for(const scenario of opponentOptions(h,view,options.data,tier)) {
      // An unknown opposing reserve is NOT today's active Pokémon: its HP,
      // typing, item and ability are unknown. Retain this probability mass
      // with the original Integrated score, not invented 35%-of-HP damage.
      if(scenario.kind==='unknown'||scenario.kind==='switch')continue;
      let outgoing=isAttack?ownDamage:0;
      let incoming=scenario.percent;
      if(reserve&&scenario.kind==='attack') {
        const hits=h.actor.moves.filter(id=>
          options.data.moves[id]?.category!=='Status').map(id=>{
          const move=options.data.moves[id];
          const damage=move?estimateAttackPercent(h.actor,reserve,
            move,options.data,view.field):null;
          return damage===null?null:damage*accuracy(move);
        });
        // If ANY possible attack against the reserve is unsupported,
        // discard the entire scenario. A null estimate is not 0 damage.
        if(!hits.length||hits.some(d=>d===null))continue;
        incoming=Math.max(0,...hits);
      }
      const recipient=reserve??own;
      const hp=health(recipient);
      const order=!reserve&&isAttack&&scenario.kind==='attack'
        ?expectedAttackOrder(view,own,h.actor,ownMove,
          options.data.moves[scenario.moveId]):null;
      const foeLikelyKO=scenario.kind==='attack'&&
        scenario.moveAccuracy>=0.95&&
        scenario.rawPercent>=hp*1.12;
      // A credible first strike ends the turn: do not credit an attack that
      // cannot happen, or penalize a secure KO with a later enemy attack.
      if(order==='foe'&&foeLikelyKO)outgoing=0;
      if(order==='own'&&secureOwnKO)incoming=0;
      if(order===null&&secureOwnKO&&foeLikelyKO)continue;
      const probability=h.weight*scenario.prob;
      covered+=probability;
      const selfLoss=scenario.kind==='attack'?Math.min(hp,incoming):0;
      const foeLoss=Math.min(foeHP,outgoing);
      const opponentKO=isAttack&&secureOwnKO&&
        !(order==='foe'&&foeLikelyKO);
      const selfKO=selfLoss>=hp&&scenario.kind==='attack';
      // One-turn projected utility. Do not claim an exact Battle rollout:
      // move-order, procs, hazards and ability effects remain approximate.
      let value=0.65*foeLoss-weights.exposure*0.65*selfLoss+
        (opponentKO?28:0)-(selfKO?30:0);
      if(scenario.kind==='setup'&&!opponentKO)value-=12;
      if(scenario.kind==='switch'&&row.category==='setup')value+=6;
      if(reserve) {
        const m=options.ownMatchups?.[row.action.index]??null;
        const safety=signal(m,'switchIn');
        value+=(safety??0)*12;
        value-=Math.max(0,100-health(reserve))*0.06;
      }
      if(tier>=4) {
        // D: bounded second decision in the approximate state. Inspect the
        // next legal own attack forecast or a reserve retreat. This is NOT
        // an authoritative Battle rollout; hidden outcomes remain unknown.
        const ownAfter=Math.max(0,hp-selfLoss);
        const foeAfter=Math.max(0,foeHP-foeLoss);
        const currentIndex=reserve?row.action.index:view.self.active;
        const nextActor=view.self.team[currentIndex];
        const nextModel=options.ownMatchups?.[currentIndex]??
          options.matchup;
        const futureAttacks=(nextActor.moves??[]).map(slot=>{
          const id=slot.id??slot;
          const forecast=nextModel?.moveEstimates?.find(m=>m.id===id);
          if(!finite(forecast?.expectedPercent))return null;
          return clamp(forecast.expectedPercent*
            accuracy(options.data.moves[id]),0,300);
        }).filter(finite);
        const nextDamage=futureAttacks.length?Math.max(...futureAttacks):0;
        const nextAttackValue=0.55*Math.min(foeAfter,nextDamage)+
          (nextDamage>=foeAfter*1.12&&foeAfter>0?18:0)-
          0.25*Math.min(ownAfter,incoming);
        const retreatValue=Math.max(-15,...view.self.team.map((member,i)=>{
          if(i===currentIndex||member.hp<=0)return -15;
          const m=options.ownMatchups?.[i];
          const safety=signal(m,'switchIn');
          return safety===null?-10:safety*18-6;
        }));
        const continued=foeAfter>0&&ownAfter>0
          ?Math.max(nextAttackValue,retreatValue):0;
        const lastOwn=view.self.team.filter(p=>p.hp>0).length===1;
        const closing=foeAfter===0&&view.opponent.remaining===1;
        value+=weights.continuation*clamp(continued+
          0.14*ownAfter-0.12*foeAfter+(closing?48:0)-
          (lastOwn&&ownAfter===0?48:0),-50,75);
      }
      utility+=probability*value;
    }
  }
  return {utility,coverage:clamp(covered,0,1)};
}

// Ranks the SAME legal action set as Integrated Strategy. No action can be
// synthesized by the approximation. All versions are opt-in and deterministic.
export function rankStrategyV2Actions(view,{variant,...options}={}) {
  if(!V2_POLICIES.includes(variant))
    throw new TypeError('Unknown v2 policy');
  if(!options.data?.moves)throw new TypeError('Missing battle definitions');
  const weights=normalizeV2Weights(options.weights);
  if(variant==='strategy-v2-e'&&!options.weights)
    throw new TypeError('v2-E requires externally trained weights');
  const baseline=rankIntegratedStrategyActions(view,options);
  const baselineAction=baseline.rows[0]?.action??null;
  if(!baseline.rows.length)return {...baseline,variant,baselineAction};
  const tier=V2_POLICIES.indexOf(variant)+1;
  const hypothesesBySet=hypotheses(view,options,tier);
  if(!hypothesesBySet.length||view.requestType!=='turn')
    return {...baseline,variant,baselineAction};
  const shortlist=baseline.rows.slice(0,LIMITS.ownActions);
  // Preserve a verified KO/viable switch even if its base rank is lower.
  const forced=baseline.rows.filter(row=>row.macroDelta>=24)
    .slice(0,1);
  const candidates=new Set([...shortlist,...forced]);
  const rescored=baseline.rows.map(row=>{
    if(!candidates.has(row))return {...row,v2Delta:0,v2Coverage:0};
    const outcome=project(row,view,options,hypothesesBySet,tier,weights);
    // Unknown hypothesis mass retains the original integrated utility.
    const blend=weights.blend*outcome.coverage;
    const projected=outcome.coverage>0
      ?outcome.utility/outcome.coverage:row.score;
    return {...row,v2Delta:blend*(projected-row.score),
      v2Coverage:outcome.coverage,
      score:row.score+blend*(projected-row.score)};
  }).sort((a,b)=>b.score-a.score);
  return {...baseline,rows:rescored,variant,baselineAction};
}

export function chooseStrategyV2Action(view,options={}) {
  const result=rankStrategyV2Actions(view,options);
  const action=result.rows[0]?.action??null;
  if(!action)return null;
  const memory=options.memory??{};
  const key=actionKey(action,view);
  memory.macroRecent=[...(Array.isArray(memory.macroRecent)
    ?memory.macroRecent.slice(-5):[]),key];
  if(action.kind==='switch')memory.lastSwitchTurn=view.turn;
  else {
    memory.lastMoveId=typeof action.index==='number'
      ?view.self.team[view.self.active].moves[action.index]?.id:'struggle';
    memory.lastMoveTurn=view.turn;
  }
  if(typeof options.onV2Choice==='function') {
    const old=result.baselineAction;
    const chosen=result.rows[0];
    options.onV2Choice({variant:options.variant,
      changed:!!old&&(old.kind!==action.kind||old.index!==action.index||
        Boolean(old.mega)!==Boolean(action.mega)),
      covered:clamp(chosen.v2Coverage??0,0,1),
      evaluated:result.rows.length});
  }
  const p=result.plan;
  const followed=!!p&&p.candidate?.kind===action.kind&&
    p.candidate?.index===action.index;
  memory.tacticalPlan=followed?{id:p.id,turn:view.turn,state:p.state}:null;
  if(typeof options.onTacticalChoice==='function')
    options.onTacticalChoice({
      proposed:p?{id:p.id,state:p.state}:null,
      followed,
      planStates:result.plans.map(x=>({
        id:x.id,state:x.state,feasible:!!x.candidate,
      })),
    });
  return action;
}
