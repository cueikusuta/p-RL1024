// Matched-seed independent 95% CI for a change in deployment action
// selection. Does not treat two seats sharing a seed as independent samples.
// No training, no policy updates, no Champion promotion.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
export function compareDeploymentModes(sample,greedy){
  function validate(x,mode){
    if(x?.schema!=='rnad-three-phase-vs-basic-paired-v1'||
      x?.opponent_id!=='basic-all-phases'||
      typeof x.model_sha256!=='string'||!/^[a-f0-9]{64}$/.test(x.model_sha256)||
      typeof x.engine_sha!=='string'||!/^[a-f0-9]{40,64}$/.test(x.engine_sha)||
      !Number.isInteger(x.pairs)||x.pairs<2||x.pairs>1024||
      x.matches!==2*x.pairs||!Array.isArray(x.rows)||
      x.rows.length!==x.matches||
      !Number.isInteger(x.wins)||x.wins<0||x.wins>x.matches||
      !finite(x.win_rate)||x.win_rate!==x.wins/x.matches||
      x.champion_promoted!==false||
      (mode==='greedy'&&x.inference_selection_mode!=='greedy')||
      (mode==='sample'&&!['sample',undefined].includes(x.inference_selection_mode)))
      throw new Error('Untrusted frozen paired deployment report');
    const map=new Map();
    for(const row of x.rows){
      if(!Number.isInteger(row.seed)||row.seed<1||row.seed>0xffffffff||
        ![0,1].includes(row.candidateSeat)||typeof row.won!=='boolean'||
        !Number.isInteger(row.rounds)||row.rounds<4||row.rounds>64||
        !Array.isArray(row.score)||row.score.length!==2||
        row.score.some(v=>!Number.isInteger(v)||v<0||v>4)||
        row.score[row.won?row.candidateSeat:1-row.candidateSeat]!==4||
        row.score[row.won?1-row.candidateSeat:row.candidateSeat]>=4)
        throw new Error('Non-authoritative paired Match result');
      const id=row.seed+':'+row.candidateSeat;
      if(map.has(id))throw new Error('Duplicate seat or seed');
      map.set(id,row.won);
    }
    if([...map.values()].filter(Boolean).length!==x.wins)
      throw new Error('Incorrect accepted Match totals');
    const seeds=new Set(x.rows.map(row=>row.seed));
    if(seeds.size!==x.pairs||
      [...seeds].some(seed=>!map.has(seed+':0')||!map.has(seed+':1')))
      throw new Error('Incomplete swapped seat seed blocks');
    return {map,seeds};
  }
  const a=validate(sample,'sample'),b=validate(greedy,'greedy');
  if(sample.model_sha256!==greedy.model_sha256||
     sample.engine_sha!==greedy.engine_sha||
     sample.pairs!==greedy.pairs||sample.matches!==greedy.matches||
     a.seeds.size!==b.seeds.size||
     [...a.map.keys()].some(id=>!b.map.has(id)))
    throw new Error('Different models, engines or held-out seed/seat blocks');
  const pairs=a.seeds.size;
  const diff=(greedy.wins-sample.wins)/(2*pairs);
  // Per-seed outcome difference is in [-1,+1]. For two-sided alpha=.05
  // Hoeffding radius is sqrt(2 * log(2/alpha) / independent_seed_count).
  const half=Math.sqrt(2*Math.log(40)/pairs);
  const ci=[Math.max(-1,diff-half),Math.min(1,diff+half)];
  return {schema:'rnad-deployment-sampling-greedy-paired-difference-v1',
    model_sha256:sample.model_sha256,engine_sha:sample.engine_sha,
    independent_seed_blocks:pairs,matches_each:2*pairs,
    sample_wins:sample.wins,greedy_wins:greedy.wins,
    win_rate_difference:diff,paired_seed_hoeffding95_difference:ci,
    improvement_statistically_established:ci[0]>0,
    champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2);
  if(args.length!==6||args[0]!=='--sample'||
     args[2]!=='--greedy'||args[4]!=='--out')
    throw new Error('Usage --sample FILE --greedy FILE --out FILE');
  const a=JSON.parse(readFileSync(args[1])),b=JSON.parse(readFileSync(args[3]));
  const result=compareDeploymentModes(a,b);
  writeFileSync(args[5],JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(result));
}
