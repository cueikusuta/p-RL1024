// I737: opt-in three-member selection. Preserve the legacy Basic selection
// as the reference, adding only small, explicitly executable tactical packages.
// No enemy hidden team/sets, no Battle state, and no six-member draft edits.
import {rankBasicSelectionCandidates} from './selection-basic.js';
import {analyzeSetCapabilities} from './set-capabilities.js';

export const TACTICAL_SELECTION_LIMITS=Object.freeze({
  maxBasicDeficit:6,
  maxTacticBonus:6,
});
// Planned bonuses are uncalibrated selection utilities, not win probabilities.
// Only the HIGHEST viable plan applies; bonuses never stack.
const BONUSES=Object.freeze({
  'screens-ace':4,
  'screens-setup-ace':5,
  'boost-baton':5,
  'trickroom-slow':5,
});
const BOOST=new Set(['swordsdance','nastyplot','dragondance','agility',
  'quiverdance','bulkup','calmmind','shellsmash','cosmicpower','irondefense']);
const hasMove=(p,id)=>p.moves?.includes(id)??false;
const hasAny=(p,ids)=>p.moves?.some(id=>ids.has(id))??false;
const hasTag=(c,tag)=>c?.tags.includes(tag)??false;
// Match.view provides the own confirmed stats, including nature. Never
// estimate unknown natures or base stats: a fabricated speed ordering can
// cause an unjustified Trick Room selection.
const speed=p=>Number.isFinite(p.stats?.spe)&&p.stats.spe>0
  ?p.stats.spe:null;
function evaluateTrio(pokemon,data,capabilities) {
  const tagged=pokemon.map(p=>capabilities.get(p.instanceId));
  const plans=[];
  // Hazards and Yawn are situational individual moves, not a mandatory
  // three-move package. The generic action evaluator already ranks them.
  for(let i=0;i<3;i++) {
    const setter=pokemon[i],tags=tagged[i];
    for(let j=0;j<3;j++) {
      if(j===i)continue;
      const receiver=pokemon[j],recvTags=tagged[j];
      if(!hasTag(recvTags,'attack'))continue;
      if(hasTag(tags,'screen')) {
        const setup=hasTag(recvTags,'setup');
        plans.push({id:'screens-ace',
          bonus:BONUSES[setup?'screens-setup-ace':'screens-ace']});
      }
      const canPrepareBaton=hasAny(setter,BOOST)||
        (setter.abilityId==='speedboost'&&hasMove(setter,'protect'));
      if(hasTag(tags,'batonPass')&&canPrepareBaton)
        plans.push({id:'boost-baton',bonus:BONUSES['boost-baton']});
      // P5's runtime plan requires a slower ATTACKING recipient. Treat an
      // unknown speed as unknown, never presume it is slow or usable.
      const setterSpe=speed(setter),receiverSpe=speed(receiver);
      if(hasTag(tags,'trickRoom')&&setterSpe!==null&&receiverSpe!==null&&
         receiverSpe<setterSpe)
        plans.push({id:'trickroom-slow',bonus:BONUSES['trickroom-slow']});
    }
  }
  // Stable priority in equal utility: P5's own plan order is fixed, while
  // Basic's candidate enumeration remains the tie-break for team selection.
  const order=['screens-ace','boost-baton','trickroom-slow'];
  plans.sort((a,b)=>b.bonus-a.bonus||order.indexOf(a.id)-order.indexOf(b.id));
  return {bestPlan:plans[0]?.id??null,bonus:plans[0]?.bonus??0,
    eligiblePlans:[...new Set(plans.map(p=>p.id))]};
}
export function evaluateTacticalSelectionTrio(team,{data}={}) {
  if(!Array.isArray(team)||team.length!==3||!data?.moves||
     team.some(p=>typeof p?.instanceId!=='string'||!p.instanceId)||
     new Set(team.map(p=>p.instanceId)).size!==3)
    throw new TypeError('Expected three distinct own sets and battle data');
  const capabilities=new Map(team.map(p=>
    [p.instanceId,analyzeSetCapabilities(p,data)]));
  return evaluateTrio(team,data,capabilities);
}
export function rankTacticalSelectionCandidates(view,options={}) {
  const {data}=options;
  if(!data?.moves)throw new TypeError('Tactical selection requires battle data');
  // Exactly the same twenty candidates, forced-lead/exclusion constraints and
  // six RNG draws as Basic. Do not call Basic twice using the same RNG.
  const basics=rankBasicSelectionCandidates(view,options);
  if(!basics.length)throw new Error('No valid three-member selection');
  const team=view.self.team;
  const byId=new Map(team.map(p=>[p.instanceId,p]));
  const capabilities=new Map(team.map(p=>
    [p.instanceId,analyzeSetCapabilities(p,data)]));
  const strongest=Math.max(...basics.map(row=>row.score));
  return basics.map((row,index)=>{
    const trio=row.selection.map(id=>byId.get(id));
    const eligibility=evaluateTrio(trio,data,capabilities);
    const deficit=strongest-row.score;
    // A weak basic trio must NEVER win merely because it has a gimmick.
    const withinBasicMargin=deficit<=TACTICAL_SELECTION_LIMITS.maxBasicDeficit+1e-9;
    const bonus=withinBasicMargin?
      Math.min(eligibility.bonus,TACTICAL_SELECTION_LIMITS.maxTacticBonus):0;
    return {
      selection:row.selection,score:row.score+bonus,basicScore:row.score,
      tacticBonus:bonus,eligiblePlan:bonus>0?eligibility.bestPlan:null,
      withinBasicMargin,basicOrder:index,
    };
  }).sort((a,b)=>b.score-a.score||a.basicOrder-b.basicOrder);
}
export function chooseTacticalSelection(view,options={}) {
  return rankTacticalSelectionCandidates(view,options)[0].selection;
}
