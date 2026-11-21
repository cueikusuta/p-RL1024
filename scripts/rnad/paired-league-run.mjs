// RNAD I5: future frozen, real Match generator for prospective paired studies.
// Keep ongoing league training/old evaluations immutable. Each shard is a
// contiguous subset of one precommitted seed cohort, both seats on every seed.
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createIntegratedAQSession} from './evaluate-shared-g012-vs-aq.mjs';
import {verifyPlan,PAIRED_LEAGUE_EVIDENCE_SCHEMA} from './paired-league-eval.mjs';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256 as AQ_BASE_SHA} from '../../models/aq-base.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const sha=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
const fail=s=>{throw new Error('RNAD prospective paired runner: '+s);};
const check=(condition,s)=>{if(!condition)fail(s);};

function frozenBundle(directory,id,plan){
  const manifest=JSON.parse(readFileSync(join(directory,'manifest.json'),'utf8'));
  check(manifest.schema==='rnad-integrated-policy-actor-bundle-v1'&&
    Number.isInteger(manifest.generation)&&manifest.generation>0&&
    typeof manifest.policy_id==='string'&&manifest.policy_id.length>0&&
    manifest.source_sha===verifiedRunningSourceSha()&&
    manifest.source_sha===plan.source_sha&&
    manifest.engine_sha===provenance.sourceCommit&&
    manifest.engine_sha===plan.engine_sha&&
    Object.keys(manifest.files??{}).sort().join(',')===
      'battle,draft,selection,values',
    'invalid frozen integrated policy/engine/source');
  for(const [head,digest] of Object.entries(manifest.files))
    check(sha(join(directory,head+'.json'))===digest,
      'tampered frozen '+head+' model head');
  const digest=sha(join(directory,'manifest.json'));
  check(digest===plan.model_sha256_by_id[id],
    'unregistered model manifest SHA '+id);
  return {directory,id,manifest,digest};
}
export function evaluateFrozenPairedShard({
  plan,modelDir,modelId,baselineDir=null,part=0,parts=8,
}={}){
  verifyPlan(plan);
  check(Number.isInteger(part)&&Number.isInteger(parts)&&
    parts>=1&&parts<=1024&&part>=0&&part<parts&&
    plan.seed_blocks%parts===0,'invalid precommitted shard partition');
  check(plan.comparisons.some(c=>c.candidate_model_id===modelId||
    (plan.evaluation_type==='shared-opponent'&&c.baseline_model_id===modelId)),
  'unlisted candidate/baseline generation');
  if(plan.evaluation_type==='direct-head-to-head')
    check(typeof baselineDir==='string'&&baselineDir.length>0,
      'missing frozen Champion bundle');
  else check(baselineDir===null&&plan.opponent_id==='AQ-Base-001',
    'shared-opponent runner supports exact frozen AQ-Base only');
  const challenger=frozenBundle(modelDir,modelId,plan);
  const incumbent=baselineDir?frozenBundle(
    baselineDir,plan.opponent_id,plan):null;
  if(incumbent)check(challenger.digest!==incumbent.digest,
    'identical Challenger/Champion checkpoint');
  const aq=createOpponent();
  check(aq.modelSha256===AQ_BASE_SHA,'AQ model changed');
  const pairs=plan.seed_blocks/parts,start=plan.seed_start+part*pairs;
  const rows=[];
  for(let i=0;i<pairs;i++){
    const seed=start+i;
    for(const candidateSeat of [0,1]){
      const actor=createIntegratedAQSession({
        modelDir:challenger.directory,seat:candidateSeat});
      const opponentSeat=1-candidateSeat;
      const baseline=incumbent?createIntegratedAQSession({
        modelDir:incumbent.directory,seat:opponentSeat}):null;
      const sessions=[0,1].map(s=>s===candidateSeat?actor:baseline);
      const policies=[0,1].map(s=>s===candidateSeat||incumbent
        ?'pure-rl-linear':'aq-base');
      const result=runStrategyMatch({seed,policies,pureRlSessions:sessions,
        pureRlAllPhases:true,strategyMode:'integrated',
        v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq});
      const candidateStats=result.pureRlStats?.[candidateSeat];
      const baselineStats=incumbent?result.pureRlStats?.[opponentSeat]:null;
      check(result.completed===true&&[0,1].includes(result.winner)&&
        Array.isArray(result.score)&&result.score.length===2&&
        result.score[result.winner]===4&&
        result.score[1-result.winner]<4&&
        Number.isInteger(result.rounds)&&result.rounds>=4&&
        result.rounds<=64&&
        candidateStats?.updates===0&&candidateStats?.pending===false&&
        candidateStats?.accepted_decisions===
          result.actionCounts[candidateSeat]+2*result.rounds&&
        (!incumbent||(baselineStats?.updates===0&&
          baselineStats?.pending===false&&
          baselineStats?.accepted_decisions===
            result.actionCounts[opponentSeat]+2*result.rounds)),
        'invalid authority-accepted frozen four-win Match');
      rows.push({seed,candidateSeat,
        won:result.winner===candidateSeat,completed:true,
        rounds:result.rounds,score:result.score,
        accepted_decisions:candidateStats.accepted_decisions});
    }
  }
  return {
    schema:PAIRED_LEAGUE_EVIDENCE_SCHEMA,
    model_id:modelId,model_sha256:challenger.digest,
    source_sha:plan.source_sha,engine_sha:plan.engine_sha,
    opponent_id:plan.opponent_id,
    evaluation_policy:'frozen-greedy-no-training',
    training_during_evaluation:false,champion_promoted:false,
    historical_reanalysis:false,
    seed_start:start,seed_blocks:pairs,
    matches:rows.length,wins:rows.filter(x=>x.won).length,
    rows,
  };
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const a=process.argv.slice(2),opt={};
  for(let i=0;i<a.length;i+=2){
    check(['--plan','--model-dir','--model-id','--baseline-dir',
      '--part','--parts','--out'].includes(a[i])&&!Object.hasOwn(opt,a[i])&&
      a[i+1]!==undefined,'invalid frozen evaluation CLI');
    opt[a[i]]=a[i+1];
  }
  check(['--plan','--model-dir','--model-id','--part','--parts','--out']
    .every(k=>typeof opt[k]==='string'),
  'missing paired Match evaluation arguments');
  const plan=JSON.parse(readFileSync(opt['--plan'],'utf8'));
  const r=evaluateFrozenPairedShard({
    plan,modelDir:opt['--model-dir'],modelId:opt['--model-id'],
    baselineDir:opt['--baseline-dir']??null,
    part:Number(opt['--part']),parts:Number(opt['--parts']),
  });
  writeFileSync(opt['--out'],JSON.stringify(r)+'\n',{flag:'wx'});
  console.log(JSON.stringify({model_id:r.model_id,part:Number(opt['--part']),
    matches:r.matches,seed_blocks:r.seed_blocks,
    wins:r.wins,champion_promoted:false}));
}
