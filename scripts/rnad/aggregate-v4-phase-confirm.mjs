// Independent held-out phase confirmation after exploratory selection alpha.
// Each seed contributes exactly two original AQ-Base four-win Matches.
// Requires two independently sharded reports per arm, same source/model/opponent.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {compareV4PhaseArms} from './compare-v4-phase-arms.mjs';

export function aggregateV4PairedConfirmation(shards,{seedStart,seedCount}={}){
  if(!Array.isArray(shards)||shards.length<1||
     !Number.isSafeInteger(seedStart)||seedStart<1||
     !Number.isSafeInteger(seedCount)||seedCount<1||seedCount>2048||
     seedStart+seedCount-1>0xffffffff)
    throw new Error('Invalid heldout confirmation schedule');
  const arms={baseline:[],candidate:[]};
  for(const shard of shards){
    if(!shard||!shard.baseline||!shard.candidate)
      throw new Error('Missing frozen pair source shard');
    // Verify every shard BEFORE aggregating or combining rows.
    const score=compareV4PhaseArms(shard.baseline,shard.candidate);
    if(shard.candidate.mode!=='selection-threequarter-v4'|| 
       shard.baseline.mode!=='all-v3'||
       score.champion_promoted!==false)
      throw new Error('Not predeclared v4 threequarter selection against v3');
    arms.baseline.push(shard.baseline);
    arms.candidate.push(shard.candidate);
  }
  const assemble=reports=>{
    const original=reports[0];
    const expectedKeys=['schema','source_sha','engine_sha',
      'aq_base_model_sha256','v4_model_sha256','v3_model_sha256',
      'mode','policy','champion_promoted'];
    for(const item of reports){
      if(expectedKeys.some(k=>item[k]!==original[k]))
        throw new Error('Mixed policy/source/model across independent shards');
    }
    const seen=new Set(),rows=[];
    for(const item of reports)for(const entry of item.rows){
      const id=entry.seed+':'+entry.candidateSeat;
      if(seen.has(id))throw new Error('Reused heldout seed/seat');
      seen.add(id);rows.push(entry);
    }
    rows.sort((a,b)=>a.seed-b.seed||a.candidateSeat-b.candidateSeat);
    if(rows.length!==2*seedCount||
       rows.some((x,i)=>x.seed!==seedStart+Math.floor(i/2)||
         x.candidateSeat!==i%2))
      throw new Error('Missing independent seed, seat or wrong block');
    const wins=rows.filter(x=>x.won).length;
    return {...original,seed:seedStart,pairs:seedCount,
      matches:2*seedCount,wins,win_rate:wins/(2*seedCount),rows};
  };
  const baseline=assemble(arms.baseline),candidate=assemble(arms.candidate);
  const checked=compareV4PhaseArms(baseline,candidate);
  if(checked.independent_seeds!==seedCount)
    throw new Error('Invalid independently paired summary');
  return {...checked,schema:'rnad-i5-original-aq-v4-phase-confirmation-v1',
    experiment_type:'single_locked_selection_alpha_confirmation',
    baseline_mode:'all-v3',candidate_mode:'selection-threequarter-v4',
    selection_weights:{v4:.75,v3:.25},
    no_adaptive_early_stopping:true,
    seed_start:seedStart,seed_count:seedCount,
    game_strength_proven:checked.improved_with_95pct_confidence,
    aq_beaten:false,champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2);
  if(args.length<5||args[0]!=='--seed-start'||args[2]!=='--seed-count'||
     args[4]!=='--out'||!args[5]||
     args.length<8||(args.length-6)%2!==0)
    throw new Error('Usage: --seed-start N --seed-count N --out RESULT BASE.json CAND.json ...');
  const shards=[];
  for(let i=6;i<args.length;i+=2){
    shards.push({baseline:JSON.parse(readFileSync(args[i],'utf8')),
      candidate:JSON.parse(readFileSync(args[i+1],'utf8'))});
  }
  const report=aggregateV4PairedConfirmation(shards,{
    seedStart:Number(args[1]),seedCount:Number(args[3])});
  writeFileSync(args[5],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(report));
}
