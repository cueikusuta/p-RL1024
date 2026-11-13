// I737 Hybrid: Basic is the immutable reference; all corrections are opt-in.
// The optional diagnostic callback is developer-only and never sent to Match or clients.
import {chooseBasicSelection,rankBasicSelectionCandidates,publicRosterTypes}
  from './selection-basic.js';
import {effectiveness} from '../battle-mechanics.js';
import {scorePublicOpponentTrioHypotheses} from './team-assessment.js';

export const HYBRID_FEATURES=Object.freeze([
  'coverage','unfavorable','opponentHypotheses',
]);
export const ZERO_HYBRID_WEIGHTS=Object.freeze({
  coverage:0,unfavorable:0,opponentHypotheses:0,
});
export function normalizeHybridWeights(weights={}) {
  if(!weights||typeof weights!=='object'||Array.isArray(weights))
    throw new TypeError('Expected Hybrid feature weights');
  const normalized={...ZERO_HYBRID_WEIGHTS};
  for(const key of Object.keys(weights)) {
    if(!HYBRID_FEATURES.includes(key))throw new TypeError('Unknown Hybrid feature: '+key);
    const weight=weights[key];
    // A feature is on a [-1,1] scale, and each weight is bounded so no
    // single uncalibrated feature can arbitrarily dominate Basic's score.
    if(typeof weight!=='number'||!Number.isFinite(weight)||weight<0||weight>12)
      throw new TypeError('Hybrid weight must be finite in [0,12]: '+key);
    normalized[key]=weight;
  }
  return normalized;
}
function damagingScore(set,types,data) {
  if(!Array.isArray(types)||types.length===0)return null;
  if(!Array.isArray(set.moves)||!Array.isArray(set.types))return null;
  let best=0,hasAttack=false;
  for(const id of set.moves) {
    const move=data.moves?.[id];
    if(!move)return null; // An unknown attack is not a known zero.
    if(move.category==='Status')continue;
    if(!Number.isFinite(move.basePower)||move.basePower<=0)return null;
    hasAttack=true;
    const eff=effectiveness(move.type,types,data.typeChart);
    if(!Number.isFinite(eff)||eff<0)return null;
    const stab=set.types.includes(move.type)?1.2:1;
    best=Math.max(best,Math.min(1,eff*stab*move.basePower/150/2));
  }
  // Status-only sets can win indirectly. We cannot assert an offensive lock.
  return hasAttack?best:null;
}
export function createHybridCoverageContext(view,{data,sets}={}) {
  if(!data?.moves||!data?.typeChart||!Array.isArray(sets))
    throw new TypeError('Hybrid features require catalog and battle data');
  const team=view?.self?.team,foeNames=view?.opponent?.species;
  if(!Array.isArray(team)||team.length!==6||
     !Array.isArray(foeNames)||(foeNames.length!==0&&foeNames.length!==6)||
     foeNames.some(name=>typeof name!=='string'))
    throw new TypeError('Hybrid features require six own sets and public preview');
  const foeTypes=publicRosterTypes(foeNames,sets,data);
  // All 36 own-set/public-species estimates are calculated at most once.
  const matrix=team.map(p=>foeTypes.map(types=>damagingScore(p,types,data)));
  const byId=new Map(team.map((p,i)=>[p.instanceId,i]));
  return {matrix,byId,publicCount:foeNames.length,
    unknownPublicSpecies:foeTypes.filter(types=>!types.length).length};
}
// Zero=unavailable is NEVER silently treated as an average or neutral matchup.
export function evaluateHybridFeatures(selection,context) {
  const {matrix,byId,publicCount}=context;
  const indexes=selection.map(id=>byId.get(id));
  if(indexes.length!==3||indexes.some(i=>i===undefined)||
     new Set(indexes).size!==3)
    throw new TypeError('Hybrid features require three distinct own members');
  const coverage=[];
  let complement=0,unknown=0;
  for(let t=0;t<publicCount;t++) {
    const values=indexes.map(i=>matrix[i][t]);
    if(values.some(x=>x===null)) {
      coverage.push(null);unknown++;continue;
    }
    const best=Math.max(...values);
    coverage.push(best);
    complement+=best-(values[0]+values[1]+values[2])/3;
  }
  // Missing previews or unmodelled matchups disable the correction; they are
  // reported as unknown, NOT scored as neutral, zero or a fabricated 50%.
  const allKnown=publicCount===6&&unknown===0;
  if(!allKnown)return {
    coverage:null,unfavorable:null,opponentHypotheses:null,
    evaluated:publicCount-unknown,unknown,publicCount,
  };
  const average=coverage.reduce((a,b)=>a+b,0)/6;
  // Below 0.35 attack strength against any public foe is a worst-case risk,
  // not proof that the foe's hidden set would beat the team.
  const maxGap=Math.max(...coverage.map(n=>Math.max(0,0.35-n)/0.35));
  const hypotheses=scorePublicOpponentTrioHypotheses(coverage);
  return {
    coverage:Math.max(0,Math.min(1,complement/6)),
    unfavorable:-Math.min(1,maxGap),
    // Remove the simple average: only concentration risk across public 3-of-6
    // hypotheses remains. This is not a calibrated win probability.
    opponentHypotheses:Math.max(-1,Math.min(0,hypotheses-average)),
    evaluated:6,unknown:0,publicCount:6,
  };
}
function chooseFromRows(rows,{context,weights,onDiagnostics}) {
  let best=null,bestScore=-Infinity;
  const records=[];
  for(const row of rows) {
    const values=context?evaluateHybridFeatures(row.selection,context):null;
    const contributions=Object.fromEntries(HYBRID_FEATURES.map(key=>[
      key,weights[key]*(values?.[key]??0),
    ]));
    const correction=HYBRID_FEATURES.reduce((s,key)=>s+contributions[key],0);
    const score=row.score+correction;
    if(score>bestScore){bestScore=score;best=row.selection;}
    if(onDiagnostics)records.push({
      selection:[...row.selection],basic:row.score,
      features:values,contributions,correction,total:score,
    });
  }
  if(!best)throw new Error('No valid three-member selection');
  if(onDiagnostics)onDiagnostics({
    selected:[...best],weights:{...weights},rows:records,
    caution:'Own-team developer diagnostics only. Do not publish raw selections.',
  });
  return best;
}
export function chooseHybridSelection(view,{weights={},onDiagnostics=null,...options}={}) {
  const normalized=normalizeHybridWeights(weights);
  if(onDiagnostics!==null&&typeof onDiagnostics!=='function')
    throw new TypeError('Expected optional developer diagnostics callback');
  const enabled=HYBRID_FEATURES.some(key=>normalized[key]!==0);
  // Fast path is the EXACT Basic entrypoint, preserving RNG consumption and
  // all option validation and tie behavior when every correction is zero.
  if(!enabled&&!onDiagnostics)return chooseBasicSelection(view,options);
  const rows=rankBasicSelectionCandidates(view,options);
  const context=(enabled||onDiagnostics)?createHybridCoverageContext(view,options):null;
  return chooseFromRows(rows,{context,weights:normalized,onDiagnostics});
}
