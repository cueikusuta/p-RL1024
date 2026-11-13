import {isCpuSpecialSet} from './cpu-special-sets.js';

export class RuleError extends Error {
  constructor(message, code) {
    super(message);
    this.code=code;
  }
}
export function requireRule(condition, message, code) {
  if (!condition) throw new RuleError(message,code);
}
export function budgetFor(round) {
  requireRule(Number.isSafeInteger(round) && round >= 1 && round <= Number.MAX_SAFE_INTEGER - 9, 'Invalid round');
  return round === 1 ? 10 : 9 + round;
}

// Mega candidates are additional to the twelve normal sets.
// Carry count is the number offered at the start of the round, not the
// number the player eventually elects to keep.
export function megaOfferRule(round, carryCount=0) {
  budgetFor(round);
  requireRule(Number.isInteger(carryCount) && carryCount >= 0 && carryCount <= 3, 'Invalid carry count');
  requireRule(round!==1 || carryCount===0, 'No carry in first round');
  const count=Math.max(1,3-carryCount);
  return { min:count, max:count };
}

// Guaranteed candidate slots by round phase. The remaining one (R2+) or two
// (R1) slots are independently drawn by catalog-offers.js.
export function offerCostGuarantees(round) {
  budgetFor(round);
  const phase = round === 1 ? 0 : round === 2 ? 1 : round <= 4 ? 2 : round === 5 ? 3 : round === 6 ? 4 : 5;
  return [
    {1:6,2:1,3:1,4:1,5:1,6:0,7:0,draws:2,drawFrom:[2,3,4,5]},
    {1:6,2:1,3:1,4:1,5:1,6:1,7:0,draws:1,drawFrom:[2,3,4,5,7]},
    {1:5,2:2,3:1,4:1,5:1,6:1,7:0,draws:1,drawFrom:[2,3,4,5,7]},
    {1:4,2:2,3:2,4:1,5:1,6:1,7:0,draws:1,drawFrom:[2,3,4,5,7]},
    {1:2,2:3,3:2,4:2,5:1,6:1,7:0,draws:1,drawFrom:[2,3,4,5,7]},
    {1:0,2:3,3:3,4:2,5:2,6:1,7:0,draws:1,drawFrom:[2,3,4,5,7]},
  ][phase];
}

// Identity keys are normalized by the pinned catalog reference data.
export const speciesKey = p => p.speciesClauseId ?? p.speciesId;
export const isMegaSet = p => Boolean(p.mega);
export function validateInstance(instance,{allowCpuSpecial=false}={}) {
  requireRule(instance && typeof instance === 'object', 'Invalid instance');
  for (const key of ['instanceId', 'setId', 'speciesId']) {
    requireRule(typeof instance[key] === 'string' && instance[key].length > 0, `Missing ${key}`);
  }
  requireRule(typeof instance.itemId === 'string', 'Missing itemId');
  requireRule(Number.isInteger(instance.cost) && instance.cost >= 1 &&
    (instance.cost <= 7 || (allowCpuSpecial&&isCpuSpecialSet(instance))), 'Invalid basic cost');
}
function unique(values, message) {
  requireRule(new Set(values).size === values.length, message);
}
export function validateOffer(fresh, carry, round, {cpuHandicap=null,allowCpuSpecialCarry=false}={}) {
  budgetFor(round);
  requireRule(Array.isArray(fresh) && Array.isArray(carry), 'Invalid offer arrays');
  const megaRule=megaOfferRule(round,carry.length);
  const normalCount=12+(cpuHandicap?.extraNormalCandidates??0);
  const extraMega=cpuHandicap?.extraMegaCandidates??0;
  const specialCount=cpuHandicap?.specialLegendaryCandidates??0;
  requireRule(fresh.length===normalCount+megaRule.min+extraMega+specialCount,
    cpuHandicap?'Expected CPU candidates including special slots':'Expected 12 normal sets plus additional Mega candidates');
  fresh.forEach(instance=>validateInstance(instance,{allowCpuSpecial:specialCount>0}));
  carry.forEach(instance=>validateInstance(instance,{allowCpuSpecial:specialCount>0||allowCpuSpecialCarry}));
  unique([...fresh,...carry].map(p=>p.instanceId), 'Duplicate instance');
  unique([...fresh,...carry].map(speciesKey), 'Conflicting candidate species');
  unique([...fresh,...carry].map(p=>p.itemId).filter(Boolean), 'Conflicting candidate items');
  const specials=fresh.filter(isCpuSpecialSet);
  const normals=fresh.filter(p=>!isMegaSet(p)&&!isCpuSpecialSet(p)),
    megas=fresh.filter(p=>isMegaSet(p)&&!isCpuSpecialSet(p));
  requireRule(specials.length===specialCount,'Invalid CPU legendary candidate count');
  requireRule(normals.length===normalCount, cpuHandicap?'Invalid CPU normal candidate count':'Expected 12 normal candidates');
  requireRule(megas.length===megaRule.min+extraMega, 'Invalid mega count');
  requireRule(megas.every(p=>p.cost<= (round===1?3:7)), 'Invalid mega cost');
  const rule=offerCostGuarantees(round);
  const counts=Array(8).fill(0);
  normals.forEach(p=>counts[p.cost]++);
  const released=cpuHandicap?.costOneWeightMultiplier<1?Math.min(2,rule[1]):0;
  for(const cost of [1,2,3,4,5,6,7])requireRule(counts[cost]>=rule[cost]-(cost===1?released:0),`Missing cost-${cost} guarantee`);
  requireRule(Object.keys(rule).filter(key=>/^\d$/.test(key)).reduce((sum,cost)=>sum+rule[cost],0)+rule.draws===12,
    'Invalid candidate cost distribution rule');
  const extraCosts=counts.flatMap((count,cost)=>Array.from({length:Math.max(0,count-(rule[cost]??0)+(cost===1?released:0))},()=>cost));
  if(cpuHandicap){
    requireRule(extraCosts.length===rule.draws+released+cpuHandicap.extraNormalCandidates &&
      extraCosts.every(cost=>cost>=1&&cost<=7),'Invalid CPU extra cost tier');
  }else requireRule(extraCosts.length===rule.draws&&
    extraCosts.every(cost=>rule.drawFrom.includes(cost)), 'Invalid extra cost tier');
  // R1 has six cost-1 normals and a Mega of cost <=3: five normal
  // cost-1 sets + any offered Mega always form an affordable six-member
  // party (cost <=8), since duplicate species/items are rejected above.
}
export function teamCost(team, carry) {
  const carried = new Set(carry.map(p => p.instanceId));
  return team.reduce((total, p) => total + (carried.has(p.instanceId) ? 0 : p.cost), 0);
}
export function validateTeam(ids, fresh, carry, round, credits = budgetFor(round), {cpuHandicap=null}={}) {
  budgetFor(round);
  requireRule(Number.isSafeInteger(credits) && credits >= 0, 'Invalid credits');
  requireRule(Array.isArray(ids) && ids.length === 6, 'Expected six team members');
  unique(ids, 'Duplicate selection');
  const pool = new Map([...fresh, ...carry].map(p => [p.instanceId, p]));
  const team = ids.map(id => {
    requireRule(pool.has(id), 'Unknown candidate');
    return pool.get(id);
  });
  requireRule(team.every(p=>Number.isInteger(p.cost)&&p.cost>=1&&
    (p.cost<=7||(cpuHandicap?.specialLegendaryCandidates>0&&isCpuSpecialSet(p)))),
    'Invalid CPU special team cost');
  unique(team.map(speciesKey), 'Duplicate team species');
  unique(team.map(p => p.itemId).filter(Boolean), 'Duplicate team items');
  requireRule(teamCost(team, carry) <= credits, 'Insufficient credits');
  return structuredClone(team);
}
