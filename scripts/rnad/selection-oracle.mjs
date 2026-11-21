// Map the original public AQ ranking (lead + ordered reserves, 120 rows)
// onto the authority's 60 selection actions (lead + unordered reserves).
// No secret opponent state or invented action is accepted here.
export function mapPublicAQSelectionScores(ranking, legalActions) {
  if(!Array.isArray(ranking)||ranking.length!==120||
     !Array.isArray(legalActions)||legalActions.length!==60)
    throw new TypeError('Expected AQ 120-row and authority 60-action selection');
  const keyOf=action=>{
    if(!Array.isArray(action)||action.length!==3||
       action.some(id=>typeof id!=='string'||!id)||
       new Set(action).size!==3)
      throw new TypeError('Invalid 3-member ordered lead selection');
    return JSON.stringify([action[0],[action[1],action[2]].sort()]);
  };
  const keys=legalActions.map(keyOf);
  if(new Set(keys).size!==60)
    throw new TypeError('Duplicate canonical authority selection');
  const indexes=new Map(keys.map((key,i)=>[key,i]));
  const counts=new Array(60).fill(0);
  const scores=new Array(60).fill(null);
  const orderedSeen=new Set();
  for(const row of ranking){
    if(!row||!Number.isFinite(row.score))
      throw new TypeError('Nonfinite AQ public selection score');
    const index=indexes.get(keyOf(row.selection));
    const orderedKey=JSON.stringify(row.selection);
    if(orderedSeen.has(orderedKey))
      throw new Error('Repeated AQ ordered reserve choice');
    orderedSeen.add(orderedKey);
    if(index===undefined)
      throw new Error('AQ ranking outside authority selection support');
    if(counts[index]===2)
      throw new Error('AQ ranking duplicates a reserve order');
    if(counts[index]===0)scores[index]=row.score;
    else if(Math.abs(scores[index]-row.score)>1e-8)
      throw new Error('AQ reserve order is not strategically equivalent');
    counts[index]++;
  }
  if(counts.some(n=>n!==2))
    throw new Error('AQ 120 ranking does not cover all 60 legal selections twice');
  let bestIndex=0;
  for(let i=1;i<scores.length;i++)
    if(scores[i]>scores[bestIndex])bestIndex=i;
  return {bestIndex,bestAction:legalActions[bestIndex],
    scores,scoreMax:scores[bestIndex]};
}
