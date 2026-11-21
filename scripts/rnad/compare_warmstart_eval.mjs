// Strict seed-paired comparisons: compare models using exactly the same
// held-out Match seeds and seat assignments, never 2N independent iid rows.
// This module has no runtime effects unless called as a CLI.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';

const source=x=>typeof x==='string'&&/^[a-f0-9]{40}$/.test(x);
const hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const winRate=(wins,matches)=>wins/matches;
function validateReport(report,expectedPairs){
  if(report?.schema!=='rnad-paired-full-match-eval-v1'||
    !source(report.source_sha)||!source(report.engine_sha)||
    !hash(report.model_sha256)||report.opponent_id!=='stage0-fixed-untrained'||
    report.pairs!==expectedPairs||report.matches!==expectedPairs*2||
    !Number.isSafeInteger(report.seed)||report.seed<1||
    !Array.isArray(report.rows)||report.rows.length!==report.matches)
    throw new Error('Malformed paired held-out report');
  const rows=new Map();
  let wins=0;
  for(const row of report.rows){
    if(!Number.isSafeInteger(row.seed)||row.seed<report.seed||
      row.seed>=report.seed+expectedPairs||
      ![0,1].includes(row.candidateSeat)||typeof row.won!=='boolean'||
      !Array.isArray(row.score)||row.score.length!==2||
      !row.score.every(x=>Number.isSafeInteger(x)&&x>=0&&x<=4)||
      row.score[row.won?row.candidateSeat:1-row.candidateSeat]!==4||
      row.score[row.won?1-row.candidateSeat:row.candidateSeat]>=4||
      !Number.isSafeInteger(row.rounds)||row.rounds<4||row.rounds>64)
      throw new Error('Malformed authority Match row');
    const key=row.seed+':'+row.candidateSeat;
    if(rows.has(key))throw new Error('Duplicate evaluation seed/seat');
    rows.set(key,row.won);
    wins+=Number(row.won);
  }
  if(rows.size!==report.matches||
    !Number.isSafeInteger(report.wins)||report.wins!==wins||
    report.win_rate!==winRate(wins,report.matches))
    throw new Error('Paired report count mismatch');
  return rows;
}

/** Two disjoint fixed-size blocks for one evaluated model. */
function mergeBlocks(a,b,blockPairs){
  const x=validateReport(a,blockPairs),y=validateReport(b,blockPairs);
  if(a.model_sha256!==b.model_sha256||
    a.source_sha!==b.source_sha||a.engine_sha!==b.engine_sha||
    a.opponent_id!==b.opponent_id)
    throw new Error('Mixed policy, engine or code provenance');
  const joined=new Map(x);
  for(const [key,win] of y){
    if(joined.has(key))throw new Error('Reused held-out seat/seed');
    joined.set(key,win);
  }
  if(joined.size!==4*blockPairs)
    throw new Error('Incomplete two-block paired evaluation');
  for(const key of joined.keys()){
    const [seed,seat]=key.split(':');
    if(!joined.has(seed+':'+(1-Number(seat))))
      throw new Error('Unpaired holdout seat');
  }
  return {joined,model:a.model_sha256,
    engine:a.engine_sha,source:a.source_sha,opponent:a.opponent_id};
}

/** Two-level (seed then seat) paired Hoeffding; no Champion promotion. */
export function compareMatchedWarmstart(teacherA,teacherB,rnadA,rnadB,
  {blockPairs=64}={}){
  if(!Number.isSafeInteger(blockPairs)||blockPairs<1||blockPairs>100)
    throw new Error('Invalid independent evaluation block count');
  const t=mergeBlocks(teacherA,teacherB,blockPairs);
  const r=mergeBlocks(rnadA,rnadB,blockPairs);
  if(t.engine!==r.engine||t.source!==r.source||
    t.opponent!==r.opponent||t.model===r.model)
    throw new Error('Mismatched or identical model/engine/opponent provenance');
  if(t.joined.size!==r.joined.size||
    [...t.joined.keys()].some(k=>!r.joined.has(k)))
    throw new Error('Unmatched held-out evaluation seats');
  const teacherWins=[...t.joined.values()].filter(Boolean).length;
  const rnadWins=[...r.joined.values()].filter(Boolean).length;
  const uniquePairs=t.joined.size/2;
  const gain=(rnadWins-teacherWins)/t.joined.size;
  // Each seed contributes the mean of its two seat-specific differences,
  // bounded in [-1,1]. Hoeffding for n INDEPENDENT seeds (not 2n rows).
  const halfWidth=Math.sqrt(2*Math.log(40)/uniquePairs);
  return {schema:'rnad-warmstart-paired-128-seed-ab-v1',
    source_sha:t.source,engine_sha:t.engine,
    teacher_model_sha256:t.model,rnad_model_sha256:r.model,
    distinct_seed_pairs:uniquePairs,matches_per_model:t.joined.size,
    teacher_wins:teacherWins,rnad_wins:rnadWins,
    teacher_win_rate:winRate(teacherWins,t.joined.size),
    rnad_win_rate:winRate(rnadWins,t.joined.size),
    paired_win_rate_gain:gain,
    conservative_95pct_gain_bounds:[Math.max(-1,gain-halfWidth),
                                    Math.min(1,gain+halfWidth)],
    gain_statistically_established:(gain-halfWidth)>0,
    degradation_statistically_established:(gain+halfWidth)<0,
    is_champion:false};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const paths=process.argv.slice(2);
  if(paths.length!==5)throw new Error(
    'Usage: teacher1 teacher2 rnad1 rnad2 output.json');
  const input=paths.slice(0,4).map(p=>JSON.parse(readFileSync(p,'utf8')));
  const report=compareMatchedWarmstart(...input);
  writeFileSync(paths[4],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(report));
}
