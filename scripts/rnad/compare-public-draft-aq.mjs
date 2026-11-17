// Compare genuine authority-accepted public-AQ draft intervention against
// a frozen v3 actor on identical original AQ seed/seat complete 4-win Matches.
// Public-AQ is a proxy teacher, NOT the original full strategy actor.
import {DRAFT_AQ_SCREEN_SCHEMA}
  from './evaluate-deepsets-draft-vs-aq.mjs';
import {FULL_AQ_EVALUATION_SCHEMA}
  from './evaluate-vs-aq-full.mjs';

function pairs(rows,seed,count,seatKey){
  if(!Array.isArray(rows)||rows.length!==count*2)
    throw new Error('Missing seat-paired authority rows');
  const result=new Map();
  for(const row of rows){
    if(!Number.isSafeInteger(row.seed)||row.seed<seed||
       row.seed>=seed+count||![0,1].includes(row[seatKey])||
       typeof row.won!=='boolean'||!Array.isArray(row.score)||
       row.score.length!==2||
       row.score[row.won?row[seatKey]:1-row[seatKey]]!==4||
       row.score[row.won?1-row[seatKey]:row[seatKey]]>=4)
      throw new Error('Malformed original AQ four-win Match row');
    const key=row.seed+':'+row[seatKey];
    if(result.has(key))throw new Error('Duplicate seed/seat');
    result.set(key,Number(row.won));
  }
  return result;
}
export function comparePublicDraftAgainstV3(baseline,teacher,{familySize=4}={}){
  if(!Number.isSafeInteger(familySize)||familySize<1||familySize>16||
     baseline?.schema!==DRAFT_AQ_SCREEN_SCHEMA||
     baseline.mode!=='all-v3'||baseline.champion_promoted!==false||
     teacher?.schema!==FULL_AQ_EVALUATION_SCHEMA||
     teacher.phase_mode!=='aq-public-draft-neural-rest'||
     teacher.evaluation_policy!=='greedy-no-training'||
     teacher.switch_guard!=='anti-loop-v1'||
     teacher.champion_promoted!==false||
     teacher.benchmark!==baseline.benchmark||
     teacher.source_sha!==baseline.source_sha||
     teacher.engine_sha!==baseline.engine_sha||
     teacher.aq_base_model_sha256!==baseline.aq_base_model_sha256||
     teacher.model_sha256!==baseline.v3_model_sha256||
     teacher.seed!==baseline.seed||teacher.pairs!==baseline.pairs||
     !Number.isSafeInteger(baseline.pairs)||baseline.pairs<1||
     baseline.matches!==2*baseline.pairs||
     baseline.wins!==baseline.rows?.filter(x=>x.won).length||
     teacher.wins!==teacher.rows?.filter(x=>x.won).length)
    throw new Error('Unmatched public AQ draft reference');
  const a=pairs(baseline.rows,baseline.seed,baseline.pairs,'seat');
  const b=pairs(teacher.rows,teacher.seed,teacher.pairs,'candidateSeat');
  if(a.size!==b.size||[...a.keys()].some(k=>!b.has(k)))
    throw new Error('Incomplete paired public draft benchmark');
  const delta=[...a].reduce((sum,[k,v])=>sum+b.get(k)-v,0)/a.size;
  const half=Math.sqrt(2*Math.log(40*familySize)/baseline.pairs);
  return {schema:'rnad-i5-public-draft-vs-v3-paired-v1',
    public_teacher:'aq-public-draft-neural-rest',
    seed:baseline.seed,independent_seeds:baseline.pairs,
    matches_per_arm:2*baseline.pairs,
    baseline_wins:baseline.wins,teacher_wins:teacher.wins,
    paired_delta:delta,
    conservative_familywise_95pct_delta_bounds:[
      Math.max(-1,delta-half),Math.min(1,delta+half)],
    strength_established:false,champion_promoted:false};
}
