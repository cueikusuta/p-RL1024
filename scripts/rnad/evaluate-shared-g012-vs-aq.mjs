// RNAD I5: frozen integrated G0/G1/G2 exact four-win AQ-Base evaluation.
// One trained policy, one sealed actor directory; never calls a trainer.
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256 as AQ_BASE_SHA} from '../../models/aq-base.mjs';
import {projectRnadObservation,rnadLegalActions} from '../../src/ml/legal-action-adapter.js';
import {session} from './shared_league_actor.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

export const SCHEMA='rnad-i5-shared-g012-real-aq-heldout-v1';
export const FIRST_SEED=9500001;
export const SEEDS_PER_PART=512;
export const PARTS=8;
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
const assert=(yes,message)=>{if(!yes)throw Error(message);};

export function createIntegratedAQSession({modelDir,seat}){
  assert([0,1].includes(seat),'Invalid learner seat');
  const frozen=session({modelDir,magnetDir:modelDir,epsilon:.02,greedy:true});
  let round=null,pending=null,accepted=0;
  const byPhase={draft:0,selection:0,battle:0};
  const history={lastKind:'none',consecutiveSwitches:0,
    lastSwitchTarget:-1,lastActive:-1};
  function select(view){
    assert(!pending,'Uncommitted integrated action');
    const obs=projectRnadObservation(view,{seat});
    if(obs.phase==='draft'){
      assert(Number.isSafeInteger(obs.round)&&obs.round>=1&&obs.round<=64&&
        (round===null||obs.round===round+1),'Out-of-order four-win round');
      round=obs.round;
    }
    assert(Number.isSafeInteger(round)&&round>0,'Missing round context');
    obs.acceptedHistory={...history};
    const legalActions=rnadLegalActions(view);
    const choice=frozen.func({observation:obs,legalActions,seat,round});
    assert(choice&&legalActions.some(a=>same(a,choice.action)),
      'Integrated model chose non-authoritative action');
    pending={action:structuredClone(choice.action),phase:obs.phase,
      requestType:obs.requestType??null,active:obs.self?.active??-1,round};
    return choice.action;
  }
  function commit(){
    assert(pending,'Missing integrated accepted proposal');
    const {action,phase,requestType,active,round:r}=pending;
    // runStrategyMatch calls commit() only AFTER authoritative acceptance.
    frozen.func.accepted({seat,round:r,phase,action});
    if(phase==='battle'&&requestType==='turn'){
      history.lastActive=Number.isSafeInteger(active)?active:-1;
      if(action?.kind==='switch'){
        history.lastKind='switch';
        history.consecutiveSwitches=Math.min(3000,history.consecutiveSwitches+1);
        history.lastSwitchTarget=Number.isSafeInteger(action.index)?action.index:-1;
      }else{
        history.lastKind='move';
        history.consecutiveSwitches=0;
        history.lastSwitchTarget=-1;
      }
    }else{
      history.lastKind=phase==='draft'?'draft':
        phase==='selection'?'selection':'none';
      history.consecutiveSwitches=0;
      history.lastSwitchTarget=-1;
      history.lastActive=-1;
    }
    byPhase[phase]++;
    accepted++;
    pending=null;
  }
  function stats(){
    return {schema:'rnad-shared-g012-aq-frozen-seat-session-v1',
      accepted_decisions:accepted,accepted_by_phase:{...byPhase},
      pending:pending!==null,updates:0,train:false,
      phase_mode:'integrated-greedy',switch_guard:'anti-loop-v1'};
  }
  function finish(reward){
    assert(!pending&&[-1,1].includes(reward)&&
      Object.values(byPhase).every(n=>n>0),'Incomplete four-win actor session');
    return {...stats(),terminal_reward:reward};
  }
  return {select,commit,stats,finish};
}

export function evaluatePart({modelDir,generation,part}){
  assert([0,1,2].includes(generation)&&
    Number.isSafeInteger(part)&&part>=0&&part<PARTS,
    'Invalid pre-registered G012 evaluation shard');
  const manifest=JSON.parse(readFileSync(join(modelDir,'manifest.json'),'utf8'));
  assert(manifest.schema==='rnad-integrated-policy-actor-bundle-v1'&&
    manifest.generation===generation+1&&
    manifest.source_sha===verifiedRunningSourceSha()&&
    manifest.engine_sha===provenance.sourceCommit&&
    typeof manifest.policy_id==='string'&&
    Object.keys(manifest.files||{}).sort().join(',')===
      'battle,draft,selection,values',
    'Wrong frozen integrated policy or code provenance');
  for(const [name,digest] of Object.entries(manifest.files))
    assert(sha(join(modelDir,name+'.json'))===digest,
      'Modified or corrupt G012 head '+name);
  const modelManifestSHA=sha(join(modelDir,'manifest.json'));
  const aq=createOpponent();
  assert(aq.modelSha256===AQ_BASE_SHA,'Benchmark AQ-Base source mismatch');
  let wins=0;
  const rows=[];
  const start=FIRST_SEED+part*SEEDS_PER_PART;
  for(let i=0;i<SEEDS_PER_PART;i++){
    const seed=start+i;
    for(const candidateSeat of [0,1]){
      const actor=createIntegratedAQSession({modelDir,seat:candidateSeat});
      const result=runStrategyMatch({
        seed,policies:[0,1].map(s=>s===candidateSeat?'pure-rl-linear':'aq-base'),
        pureRlSessions:[0,1].map(s=>s===candidateSeat?actor:null),
        pureRlAllPhases:true,strategyMode:'integrated',
        v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq});
      assert(result.completed===true&&[0,1].includes(result.winner)&&
        Array.isArray(result.score)&&result.score.length===2&&
        result.score[result.winner]===4&&result.score[1-result.winner]<4&&
        result.pureRlStats?.[candidateSeat]?.updates===0&&
        result.pureRlStats?.[candidateSeat]?.pending===false&&
        result.pureRlStats?.[candidateSeat]?.accepted_decisions===
          result.actionCounts[candidateSeat]+2*result.rounds,
        'Unaccepted, incomplete, or trained full four-win Match');
      const won=result.winner===candidateSeat;
      wins+=Number(won);
      rows.push({seed,candidateSeat,won,
        score:result.score,rounds:result.rounds,
        accepted_decisions:result.pureRlStats[candidateSeat].accepted_decisions});
    }
  }
  assert(rows.length===2*SEEDS_PER_PART,'Incomplete eval shard');
  return {schema:SCHEMA,generation,checkpoint_generation:manifest.generation,part,
    evaluation_policy:'frozen-greedy-no-training',
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    aq_base_model_sha256:AQ_BASE_SHA,
    research_policy_id:manifest.policy_id,
    head_sha256:manifest.files,
    sealed_model_manifest_sha256:modelManifestSHA,
    source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,
    seed_start:start,seed_clusters:SEEDS_PER_PART,matches:rows.length,
    wins,champion_promoted:false,rows};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const argv=process.argv.slice(2),opt={};
  for(let i=0;i<argv.length;i+=2){
    assert(argv[i]?.startsWith('--')&&argv[i+1]!==undefined,'Invalid eval CLI');
    const key=argv[i].slice(2);
    assert(!Object.hasOwn(opt,key),'Duplicate eval argument');
    opt[key]=argv[i+1];
  }
  assert(Object.keys(opt).sort().join(',')==='dir,generation,out,part',
    'Expected --dir --generation --part --out');
  const result=evaluatePart({modelDir:opt.dir,
    generation:Number(opt.generation),part:Number(opt.part)});
  writeFileSync(opt.out,JSON.stringify(result)+'\n',{flag:'wx'});
  console.log(JSON.stringify({generation:result.generation,part:result.part,
    wins:result.wins,matches:result.matches,seed_start:result.seed_start}));
}
