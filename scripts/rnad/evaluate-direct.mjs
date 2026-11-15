// Direct neural-vs-neural evaluation with identical Match seeds,
// both seat assignments and an authoritative 4-win Match per result.
// Independent of Stage0 baseline; demonstrates actual relative strength.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {PersistentRNADInference,RNADMixedPolicy} from './infer.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};
import {pairedHoeffding95} from './aggregate-eval.mjs';

const sha256=file=>createHash('sha256').update(readFileSync(file)).digest('hex');
const createPolicy=path=>new RNADMixedPolicy({
  model:new PersistentRNADInference(path),
  magnet:new PersistentRNADInference(path),epsilon:0
});

export function directPairedNeuralEval({
  teacherPath,rnadPath,seed=988501,pairs=64}={}){
  if(typeof teacherPath!=='string'||!teacherPath||
    typeof rnadPath!=='string'||!rnadPath||
    !Number.isSafeInteger(seed)||seed<1||
    !Number.isSafeInteger(pairs)||pairs<1||pairs>100||
    seed+pairs-1>0xffffffff)
    throw new TypeError('Invalid bounded neural-versus-neural evaluation');
  const teacherSha=sha256(teacherPath),rnadSha=sha256(rnadPath);
  if(teacherSha===rnadSha)
    throw new Error('Expected different frozen model files');
  const rows=[];
  let wins=0;
  for(let offset=0;offset<pairs;offset++){
    const sampleSeed=seed+offset;
    for(const rnadSeat of [0,1]){
      // Do not reuse GRU memory between matches or models.
      const teacher=createPolicy(teacherPath);
      const rnad=createPolicy(rnadPath);
      const version=['RNAD-001-teacher','RNAD-001-selfplay'];
      const result=runRnadAcceptanceSmoke({
        seed:sampleSeed,matchId:'direct-neural-'+sampleSeed+'-'+rnadSeat,
        experimentId:'RNAD-001-direct-heldout',
        sourceSha:verifiedRunningSourceSha(),
        engineSnapshotSha:provenance.sourceCommit,
        policyVersions:[0,1].map(seat=>
          seat===rnadSeat?version[1]:version[0]),
        policy:ctx=>ctx.seat===rnadSeat?
          rnad.choose(ctx):teacher.choose(ctx)
      });
      if(!result.completed||!Array.isArray(result.score)||
        result.score[result.winner]!==4||
        !result.trajectories?.every(t=>t.completed))
        throw new Error('Incomplete authoritative neural head-to-head Match');
      const rnadWon=result.winner===rnadSeat;
      if(rnadWon)wins++;
      rows.push({seed:sampleSeed,rnadSeat,won:rnadWon,
        score:result.score,rounds:result.rounds});
    }
  }
  const matches=2*pairs;
  const ci95=pairedHoeffding95(wins,pairs);
  return {schema:'rnad-direct-neural-paired-match-v1',
    source_sha:verifiedRunningSourceSha(),engine_sha:provenance.sourceCommit,
    teacher_model_sha256:teacherSha,rnad_model_sha256:rnadSha,
    seed,pairs,matches,wins,win_rate:wins/matches,
    paired_seed_hoeffding_95pct:ci95,
    improvement_statistically_established:ci95[0]>.5,
    is_champion:false,rows};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const argv=process.argv.slice(2);
  if(argv.length!==10||argv.some((v,i)=>i%2===0&&!v.startsWith('--')))
    throw new Error('Usage: --teacher FILE --rnad FILE --seed N --pairs N --out FILE');
  const params={};
  for(let i=0;i<argv.length;i+=2){
    const key=argv[i].slice(2);
    if(!['teacher','rnad','seed','pairs','out'].includes(key)||
      Object.hasOwn(params,key))throw new Error('Unknown duplicate CLI argument');
    params[key]=argv[i+1];
  }
  if(!Object.hasOwn(params,'out')||!Object.hasOwn(params,'teacher')||
    !Object.hasOwn(params,'rnad'))throw new Error('Missing required input');
  const result=directPairedNeuralEval({
    teacherPath:params.teacher,rnadPath:params.rnad,
    seed:Number(params.seed),pairs:Number(params.pairs)});
  writeFileSync(params.out,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({matches:result.matches,wins:result.wins,
    win_rate:result.win_rate,ci95:result.paired_seed_hoeffding_95pct,
    significant:result.improvement_statistically_established}));
}
