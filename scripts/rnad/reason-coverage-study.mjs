// Read-only authoritative 4-win Match probe for the already frozen R-NaD
// policy. It must not train, rewrite checkpoints, or infer private opponent sets.
// The real Match runner supplies post-acceptance receipts and seat-local views.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {PersistentRNADInference,RNADMixedPolicy} from './infer.mjs';
import {RnadSwitchLoopGuard} from './switch-loop-guard.mjs';
import {summarizeAcceptedSwitchOpportunities}
  from './switch-opportunity.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
const parse=argv=>{
  const permitted=new Set(['--model','--model-file','--manifest',
    '--seed','--matches','--out']);
  if(argv.length!==12)throw new Error('Expected 6 exact argument pairs');
  const args=new Map();
  for(let i=0;i<argv.length;i+=2){
    if(!permitted.has(argv[i])||args.has(argv[i])||!argv[i+1])
      throw new Error('Duplicate or unsupported coverage option');
    args.set(argv[i],argv[i+1]);
  }
  if(args.size!==6)throw new Error('Missing coverage option');
  return Object.fromEntries([...args].map(([k,v])=>[k.slice(2),v]));
};

function makeGuard(modelFile){
  const actor=new PersistentRNADInference(modelFile);
  const magnet=new PersistentRNADInference(modelFile);
  const guard=new RnadSwitchLoopGuard(new RNADMixedPolicy({
    model:actor,magnet,epsilon:0,selectionMode:'greedy'
  }));
  const policy=ctx=>guard.choose(ctx);
  policy.accepted=ctx=>guard.accepted(ctx);
  return policy;
}
function addCounts(dst,src){
  for(const seat of [0,1]){
    const a=dst[seat],b=src[seat];
    a.voluntary_switches+=b.voluntary_switches;
    a.shaped_switches+=b.shaped_switches;
    a.cost+=b.cost;
    for(const [k,v] of Object.entries(b.reason_counts))
      a.reason_counts[k]=(a.reason_counts[k]??0)+v;
  }
}
export function probe({model,modelFile,manifestFile,seed,matches,out}){
  if(!['base128','base256','c005'].includes(model)||
     !Number.isSafeInteger(seed)||seed<1||
     !Number.isSafeInteger(matches)||matches<1||matches>16||
     seed+matches-1>0xffffffff)
    throw new TypeError('Invalid bounded frozen coverage probe');
  const manifest=JSON.parse(readFileSync(manifestFile,'utf8'));
  const fingerprint=sha(modelFile);
  if(manifest.schema!=='rnad-i5-queued-evaluation-models-v1'||
     manifest.models?.[model]!==fingerprint||
     manifest.engine_sha!==provenance.sourceCommit||
     manifest.is_champion!==false)
    throw new Error('Frozen model/engine/provenance SHA mismatch');
  const source=verifiedRunningSourceSha();
  const bySeat=[0,1].map(seat=>({seat,voluntary_switches:0,
    shaped_switches:0,cost:0,reason_counts:{}}));
  const records=[];
  for(let n=0;n<matches;n++){
    const matchSeed=seed+n;
    // A fresh independent two-seat actor/magnet/guard per Match: no
    // cross-seed GRU or switch-guard memory may leak into the next Match.
    const policies=[makeGuard(modelFile),makeGuard(modelFile)];
    const result=runRnadAcceptanceSmoke({
      seed:matchSeed,matchId:'rnad-coverage-'+model+'-'+matchSeed,
      experimentId:'RNAD-I5-switch-reason-probe-v1',
      sourceSha:source,engineSnapshotSha:provenance.sourceCommit,
      phasePolicies:policies,policy:ctx=>policies[ctx.seat](ctx),
      policyVersion:'RNAD-003-v2-frozen-greedy-anti-loop-v1',
      opponentPolicyVersion:'RNAD-003-v2-frozen-greedy-anti-loop-v1'
    });
    if(!result.completed||!result.trajectories||
       !result.jointTimeline||!result.jointTimeline.length||
       result.score[result.winner]!==4)
      throw new Error('Only completed authoritative four-win Matches count');
    const rows=result.jointTimeline.map(({seat,index})=>{
      if(![0,1].includes(seat)||
         !Number.isSafeInteger(index)||index<0)
        throw new Error('Invalid accepted receipt seat/index');
      const row=result.trajectories[seat].decisions[index];
      if(!row||row.player_seat!==seat)
        throw new Error('Untrusted accepted action in timeline');
      return row;
    });
    const tally=summarizeAcceptedSwitchOpportunities(rows);
    addCounts(bySeat,tally.by_seat);
    records.push({seed:matchSeed,winner:result.winner,score:result.score,
      rounds:result.rounds,accepted_decisions:rows.length,
      voluntary_switches:tally.voluntary_switches,
      shaped_switches:tally.shaped_switches});
    console.log(JSON.stringify({model,seed:matchSeed,score:result.score,
      decisions:rows.length,voluntary:tally.voluntary_switches,
      shaped:tally.shaped_switches}));
  }
  const report={schema:'rnad-i5-frozen-greedy-switch-reason-study-v1',
    source_sha:source,engine_sha:provenance.sourceCommit,
    model_origin_source_sha:manifest.source_sha,
    model_id:model,model_sha256:fingerprint,
    policy:'anti-loop-v1',inference_mode:'greedy',
    reward_objective_measured:'opportunity-v1-counterfactual-read-only',
    seed_start:seed,completed_matches:matches,
    both_seats:true,all_complete_four_win:true,
    by_seat:bySeat,records,
    voluntary_switches:bySeat.reduce((n,s)=>n+s.voluntary_switches,0),
    shaped_switches:bySeat.reduce((n,s)=>n+s.shaped_switches,0),
    model_improvement_confirmed:false,is_champion:false};
  if(report.voluntary_switches!==records.reduce(
        (n,r)=>n+r.voluntary_switches,0)||
     report.shaped_switches!==records.reduce(
        (n,r)=>n+r.shaped_switches,0)||
     bySeat.some(s=>Object.values(s.reason_counts)
       .reduce((n,v)=>n+v,0)!==s.voluntary_switches))
    throw new Error('Incomplete switch reason accounting');
  writeFileSync(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({model,completed_matches:matches,
    voluntary_switches:report.voluntary_switches,
    shaped_switches:report.shaped_switches,
    by_seat:bySeat}));
  return report;
}
if(process.argv[1]?.endsWith('/reason-coverage-study.mjs')){
  const args=parse(process.argv.slice(2));
  probe({model:args.model,modelFile:args['model-file'],
    manifestFile:args.manifest,seed:Number(args.seed),
    matches:Number(args.matches),out:args.out});
}
