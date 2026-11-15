// Fixed matched-seed comparison between two *different* candidate models.
// Analyze the paired per-seed gain, never confuse correlation with 2N iid.
import {readFileSync,writeFileSync} from 'node:fs';
const paths=process.argv.slice(2);
if(paths.length!==5)throw new Error('Usage: teacher1 teacher2 rnad1 rnad2 output.json');
const load=x=>JSON.parse(readFileSync(x,'utf8'));
const [teacherA,teacherB,rnadA,rnadB]=paths.slice(0,4).map(load);
function merge(a,b) {
  if([a,b].some(x=>x.schema!=='rnad-paired-full-match-eval-v1'||
      x.matches!==128||x.pairs!==64||
      x.opponent_id!=='stage0-fixed-untrained'))throw new Error('Wrong paired report');
  if(a.model_sha256!==b.model_sha256||a.engine_sha!==b.engine_sha||
      a.source_sha!==b.source_sha)throw new Error('Mixed policy/engine versions');
  const out=new Map(),seeds=new Map();
  for(const row of [...a.rows,...b.rows]){
    if(![0,1].includes(row.candidateSeat)||!Number.isInteger(row.seed)||
       typeof row.won!=='boolean')throw new Error('Invalid accepted evaluation');
    const key=row.seed+':'+row.candidateSeat;
    if(out.has(key))throw new Error('Duplicate evaluation seed/seat');
    out.set(key,row.won);
    const seats=seeds.get(row.seed)??new Set();
    seats.add(row.candidateSeat);seeds.set(row.seed,seats);
  }
  if(seeds.size!==128||[...seeds.values()].some(set=>set.size!==2))
    throw new Error('Expected 128 distinct complete seed pairs');
  return {out,model:a.model_sha256,engine:a.engine_sha,source:a.source_sha};
}
const t=merge(teacherA,teacherB),r=merge(rnadA,rnadB);
if(t.engine!==r.engine||t.source!==r.source)throw new Error('Mismatched code/engine SHA');
if(t.out.size!==r.out.size||[...t.out.keys()].some(k=>!r.out.has(k)))
  throw new Error('Unmatched evaluation seats');
const teacherWins=[...t.out.values()].filter(Boolean).length;
const rnadWins=[...r.out.values()].filter(Boolean).length;
const gain=(rnadWins-teacherWins)/t.out.size;
const halfWidth=Math.sqrt(2*Math.log(40)/128); // score-difference per seed in [-1,1]
const report={schema:'rnad-warmstart-paired-128-seed-ab-v1',
    source_sha:t.source,engine_sha:t.engine,
    teacher_model_sha256:t.model,rnad_model_sha256:r.model,
    distinct_seed_pairs:128,matches_per_model:256,
    teacher_wins:teacherWins,rnad_wins:rnadWins,
    teacher_win_rate:teacherWins/256,rnad_win_rate:rnadWins/256,
    paired_win_rate_gain:gain,
    conservative_95pct_gain_bounds:[Math.max(-1,gain-halfWidth),
                                    Math.min(1,gain+halfWidth)],
    gain_statistically_established:(gain-halfWidth)>0,
    is_champion:false};
writeFileSync(paths[4],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify(report));
