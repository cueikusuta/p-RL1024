// Strict exact-AQ A/B model comparison on identical unseen seed and seat rows.
// Reuses the existing original-AQ validator and treats 2 games from one seed
// as ONE independent cluster. This is a research screen, never promotion.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {aggregateOriginalAQReports} from './aggregate-original-aq.mjs';

export function compareOriginalAQPaired(baseline,candidate){
  if(!Number.isInteger(baseline?.pairs)||baseline.pairs<1||
     baseline.pairs>100||candidate?.pairs!==baseline.pairs||
     candidate.seed!==baseline.seed)
    throw new Error('Different seed schedules for exact-AQ paired comparison');
  const args={seedStart:baseline.seed,seedCount:baseline.pairs};
  const b=aggregateOriginalAQReports([baseline],args);
  const c=aggregateOriginalAQReports([candidate],args);
  if(b.source_sha!==c.source_sha||b.engine_sha!==c.engine_sha||
     b.aq_base_model_sha256!==c.aq_base_model_sha256||
     b.phase_mode!==c.phase_mode||b.switch_guard!==c.switch_guard||
     b.model_sha256===c.model_sha256)
    throw new Error('Different benchmark/code/mode or identical model');
  const key=x=>x.seed+':'+x.candidateSeat;
  const old=new Map(baseline.rows.map(x=>[key(x),Number(x.won)]));
  const next=new Map(candidate.rows.map(x=>[key(x),Number(x.won)]));
  if(old.size!==baseline.matches||next.size!==candidate.matches||
     [...old.keys()].some(k=>!next.has(k)))
    throw new Error('Missing paired seed/seat or duplicate rows');
  const n=baseline.pairs;
  let sum=0;
  for(const id of old.keys())sum+=next.get(id)-old.get(id);
  const gain=sum/(2*n);
  const half=Math.sqrt(2*Math.log(40)/n);
  return {schema:'rnad-original-aq-model-paired-ab-v1',
    baseline_model_sha256:b.model_sha256,
    candidate_model_sha256:c.model_sha256,
    source_sha:b.source_sha,engine_sha:b.engine_sha,
    aq_base_model_sha256:b.aq_base_model_sha256,
    independent_seeds:n,matches_per_model:2*n,
    baseline_wins:b.wins,candidate_wins:c.wins,
    baseline_win_rate:b.win_rate,candidate_win_rate:c.win_rate,
    paired_win_rate_gain:gain,
    conservative_95pct_gain_bounds:[
      Math.max(-1,gain-half),Math.min(1,gain+half)],
    gain_statistically_established:gain-half>0,
    strength_established:false,champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const argv=process.argv.slice(2);
  if(argv.length!==3)throw new Error('Usage: BASE.json CANDIDATE.json OUT.json');
  const b=JSON.parse(readFileSync(argv[0],'utf8'));
  const c=JSON.parse(readFileSync(argv[1],'utf8'));
  const report=compareOriginalAQPaired(b,c);
  writeFileSync(argv[2],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(report));
}
