import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

export function wilson(successes,total) {
  if(!total)return null;
  const z=1.95996398454, p=successes/total, z2=z*z;
  const mid=(p+z2/(2*total))/(1+z2/total);
  const half=z*Math.sqrt(p*(1-p)/total+z2/(4*total*total))/(1+z2/total);
  // Float cancellation at exact 0/total can otherwise produce tiny values
  // on the wrong side of the observed proportion (e.g. 2.8e-17 for 0/10).
  return [successes===0?0:Math.max(0,mid-half),
    successes===total?1:Math.min(1,mid+half)];
}
const rate=(numerator,denominator)=>denominator?numerator/denominator:null;
function pairedDifference(control,experiment) {
  // Two seat-swaps from the same seed are correlated. Make each seed one
  // statistical unit; treating both matches as independent shrinks intervals.
  const refs=new Map(control.map(row=>[row.trial+':'+row.seat,row]));
  const grouped=new Map();
  for(const row of experiment) {
    const base=refs.get(row.trial+':'+row.seat);
    if(!base)throw new Error('Unpaired experiment seed/seat');
    const values=grouped.get(row.trial)??[];
    values.push(Number(row.matchWon)-Number(base.matchWon));
    grouped.set(row.trial,values);
  }
  for(const [trial,values] of grouped)
    if(values.length!==2)throw new Error('Both target-seat swaps required for trial '+trial);
  const differences=[...grouped.values()].map(values=>(values[0]+values[1])/2);
  const n=differences.length;
  if(!n)return null;
  const effect=differences.reduce((s,x)=>s+x,0)/n;
  const variance=n>1?differences.reduce((s,x)=>s+(x-effect)**2,0)/(n-1):0;
  // Student-t two-sided 95% critical values for low degrees of freedom.
  const t95=[0,12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,
    2.228,2.201,2.179,2.160,2.145,2.131,2.120,2.110,2.101,2.093,
    2.086,2.080,2.074,2.069,2.064,2.060,2.056,2.052,2.048,2.045];
  // The lookup covers small samples. For larger samples, a third-order
  // Cornish-Fisher expansion approaches the normal 95% critical value
  // instead of incorrectly freezing the interval width at df=29.
  const df=n-1,z=1.95996398454;
  const critical=n<=1?null:df<=29?t95[df]
    :z+(z**3+z)/(4*df)+(5*z**5+16*z**3+3*z)/(96*df**2)
      +(3*z**7+19*z**5+17*z**3-15*z)/(384*df**3);
  const half=n>1?critical*Math.sqrt(variance/n):null;
  return {n,effect,ci95:half===null?null:[Math.max(-1,effect-half),Math.min(1,effect+half)],
    method:'paired target-seat differences, clustered by independent seed; Student-t 95% interval'};
}
export function analyze({records,setId,setName,cost,trials,seed,round}) {
  const modes=[...new Set(records.map(row=>row.mode))];
  const summaries=modes.map(mode=>{
    const sample=records.filter(row=>row.mode===mode);
    const n=sample.length;
    const reached=sample.filter(r=>r.reachedRound).length;
    const offered=sample.filter(r=>r.offered).length;
    const drafted=sample.filter(r=>r.drafted).length;
    const selected=sample.filter(r=>r.selected).length;
    const deployed=sample.filter(r=>r.deployed).length;
    const selectedWins=sample.filter(r=>r.selected&&r.roundWon).length;
    const surviving=sample.filter(r=>r.selected&&r.survived).length;
    const draftedWins=sample.filter(r=>r.drafted&&r.matchWon).length;
    const matchWins=sample.filter(r=>r.matchWon).length;
    return {mode,setId,setName,cost,trials:n,
      reachedRoundCount:reached,offerCount:offered,offerRate:rate(offered,reached),offerRateCI95:wilson(offered,reached),draftCount:drafted,draftRate:rate(drafted,offered),draftRateCI95:wilson(drafted,offered),
      selectionCount:selected,selectionRate:rate(selected,drafted),selectionRateCI95:wilson(selected,drafted),
      deployedCount:deployed,deployedRate:rate(deployed,reached),
      selectedRoundWins:selectedWins,selectedRoundWinRate:rate(selectedWins,selected),
      selectedRoundWinCI95:wilson(selectedWins,selected),survivorCount:surviving,
      selectedSurvivalRate:rate(surviving,selected),selectedSurvivalCI95:wilson(surviving,selected),
      draftedMatchWins:draftedWins,draftedMatchWinRate:rate(draftedWins,drafted),draftedMatchWinCI95:wilson(draftedWins,drafted),
      allMatchWins:matchWins,allMatchWinRate:rate(matchWins,n),allMatchWinCI95:wilson(matchWins,n)};
  });
  const control=records.filter(row=>row.mode==='control');
  const comparisons=control.length?modes.filter(mode=>mode!=='control')
    .map(mode=>({mode,baseline:'control',...pairedDifference(control,records.filter(r=>r.mode===mode))})):[];
  return {metadata:{setId,setName,cost,trials,seed,round,matchCount:records.length,
    confidence:0.95,experimentalUnit:'seed + designated seat',baseline:'control',
    limits:'Conditional draft/selection win rates are descriptive, not causal; intervention rerolls needed for legal offers can change candidate distributions.'},
    summaries,comparisons};
}
const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
export function toCSV(result) {
  const keys=['mode','setId','setName','cost','trials','reachedRoundCount','offerCount','offerRate','offerRateCI95','draftCount','draftRate','draftRateCI95',
    'selectionCount','selectionRate','deployedCount','deployedRate','selectedRoundWinRate','selectedRoundWinCI95',
    'selectedSurvivalRate','selectedSurvivalCI95','draftedMatchWinRate','draftedMatchWinCI95',
    'allMatchWinRate','allMatchWinCI95','versusControlDifference','versusControlCI95'];
  const comparison=new Map(result.comparisons.map(r=>[r.mode,r]));
  const values=result.summaries.map(row=>keys.map(key=>{
    const value=key==='versusControlDifference'?comparison.get(row.mode)?.effect
      :key==='versusControlCI95'?comparison.get(row.mode)?.ci95:row[key];
    return quote(Array.isArray(value)?value.join('..'):value);
  }).join(','));
  return [keys.join(','),...values].join('\n')+'\n';
}
export async function writeReport(result,output) {
  await mkdir(output,{recursive:true});
  await Promise.all([
    writeFile(join(output,'summary.json'),JSON.stringify(result,null,2)+'\n'),
    writeFile(join(output,'summary.csv'),toCSV(result)),
  ]);
}
