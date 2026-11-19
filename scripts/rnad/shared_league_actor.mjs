// RNAD-I5 exact accepted four-win Match trajectory from one integrated Main.
// A frozen same-version Main/Magnet drives each actor; NO updates in Node.
// The authoritative phase Match owns the one-use acceptance receipts.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,join} from 'node:path';
import {gzipSync} from 'node:zlib';
import {DeepSetsDraftPolicy,encodeOwnDraftPool}
  from './draft-deepsets-infer.mjs';
import {loadVerifiedASelection,scoreFrozenASelection,
        selectionValueFeatures,checkASelectionParity}
  from './selection-a-public-infer.mjs';
import {encodeRichSelectionActor} from './export-selection-rich.mjs';
import {PersistentRNADInference,softmaxRnad}
  from './infer.mjs';
import {RnadSwitchLoopGuard} from './switch-loop-guard.mjs';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {encodeRnadThreePhaseDecisionV5} from
  '../../src/ml/rnad-phase-features-v5.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const ACTOR_SCHEMA='rnad-shared-league-actor-v1';
const MATCH_SCHEMA='rnad-integrated-three-phase-accepted-match-v1';
const BUNDLE_SCHEMA='rnad-integrated-policy-actor-bundle-v1';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const sha=data=>createHash('sha256').update(data).digest('hex');
const fileSha=p=>sha(readFileSync(p));
const assert=(yes,message)=>{if(!yes)throw Error(message);};
const softmax=softmaxRnad;

function linear(vec,weight,bias,n){
  const out=[];
  for(let i=0;i<n;i++){
    let v=bias[i], k=i*vec.length;
    for(let j=0;j<vec.length;j++)v+=weight[k+j]*vec[j];
    out.push(v);
  }
  return out;
}
const relu=a=>a.map(x=>Math.max(0,x));
function valueFromLayers(features,tensors,prefix,hidden){
  const keys=['0.weight','0.bias','2.weight','2.bias'].map(x=>prefix+'.'+x);
  const spec=[[hidden,features.length],[hidden],[1,hidden],[1]];
  const values=keys.map((key,i)=>{
    const t=tensors[key];
    if(!t||!same(t.shape,spec[i])||
      !Array.isArray(t.values)||t.values.length!==
      spec[i].reduce((a,b)=>a*b,1)||t.values.some(x=>!finite(x)))
      throw Error('Bad critic tensor '+key);
    return t.values;
  });
  const h=relu(linear(features,values[0],values[1],hidden));
  const y=Math.tanh(linear(h,values[2],values[3],1)[0]);
  assert(finite(y),'Invalid value prediction');
  return y;
}
function phiOne(deep,f){
  const p=deep.layers;
  return relu(linear(
    relu(linear(f,p['phi.0.weight'],p['phi.0.bias'],48)),
    p['phi.2.weight'],p['phi.2.bias'],48));
}
function draftFeatures(deep,encoded){
  const h=encoded.features.map(row=>phiOne(deep,row));
  assert(h.length>0,'Empty own roster');
  const mean=Array(48).fill(0);
  for(const row of h)for(let i=0;i<48;i++)mean[i]+=row[i]/h.length;
  const extras=encoded.codes.map(code=>{
    const spent=code.reduce((n,i)=>n+encoded.prices[i],0);
    return [spent/100,(encoded.credits-spent)/100,
      code.reduce((n,i)=>n+encoded.carries[i],0)/6,
      code.reduce((n,i)=>n+encoded.megas[i],0)/6];
  });
  return [...mean,...Array.from({length:4},(_,k)=>
    extras.reduce((n,row)=>n+row[k],0)/extras.length)];
}
const verifiedBundleCache=new Map();
const verifiedLegacyCache=new Map();
function readBundle(directory){
  const root=resolve(directory);
  const cached=verifiedBundleCache.get(root);
  // Reuse SHA-verified immutable head data, but always allocate a NEW GRU
  // inference object: recurrent state must never leak between Matches/seats.
  if(cached)return {...cached,
    battle:new PersistentRNADInference(cached.battlePayload)};
  const p=JSON.parse(readFileSync(join(root,'manifest.json')));
  assert(p.schema===BUNDLE_SCHEMA&&
    /^[0-9a-f]{40}$/.test(p.source_sha)&&
    p.source_sha===verifiedRunningSourceSha()&&
    p.engine_sha===provenance.sourceCommit&&
    typeof p.policy_id==='string'&&p.policy_id.length>0&&
    Number.isSafeInteger(p.generation),'Untrusted integrated Actor manifest');
  const required=['draft','selection','battle','values'];
  assert(same(Object.keys(p.files).sort(),required.slice().sort()),
    'Wrong integrated Actor files');
  const verified={};
  for(const name of required){
    const filename=name+'.json', path=join(root,filename), digest=p.files[name];
    assert(/^[0-9a-f]{64}$/.test(digest)&&fileSha(path)===digest,
      'Corrupt frozen Actor Head '+name);
    verified[name]=path;
  }
  const draft=new DeepSetsDraftPolicy(verified.draft,{sha256:p.files.draft});
  const selection=loadVerifiedASelection(
    readFileSync(verified.selection),p.files.selection);
  checkASelectionParity(selection);
  const battlePayload=JSON.parse(readFileSync(verified.battle));
  const battle=new PersistentRNADInference(battlePayload);
  assert(battle.featureSchema==='rnad-three-phase-features-v5',
    'Only v5 battle expert is supported');
  const values=JSON.parse(readFileSync(verified.values));
  assert(values.schema==='rnad-integrated-value-heads-v1'&&
    values.policy_id===p.policy_id&&values.generation===p.generation,
    'Incorrect value/head generation');
  const cache={p,draft,selection,values,battlePayload};
  verifiedBundleCache.set(root,cache);
  return {...cache,battle};
}
function readLegacy(path,expectedSha){
  const key=resolve(path),cached=verifiedLegacyCache.get(key);
  if(cached){
    assert(cached.sha===expectedSha,'Frozen legacy opponent SHA drift');
    return new PersistentRNADInference(cached.payload);
  }
  assert(expectedSha===fileSha(path),'Frozen legacy opponent SHA mismatch');
  const payload=JSON.parse(readFileSync(path));
  const nn=new PersistentRNADInference(payload);
  assert(nn.featureSchema==='rnad-three-phase-features-v3',
    'Only frozen v3 historical opponent in legacy adapter');
  verifiedLegacyCache.set(key,{sha:expectedSha,payload});
  return nn;
}
function sample(p,rng){
  assert(rng&&typeof rng.integer==='function','Missing deterministic authority RNG');
  const x=rng.integer(0x1000000)/0x1000000;
  let sum=0;
  for(let i=0;i<p.length;i++){sum+=p[i];if(x<sum)return i;}
  return p.length-1;
}

function session({modelDir,magnetDir,legacyPath=null,legacySha=null,epsilon=.02}){
  assert(finite(epsilon)&&epsilon>0&&epsilon<=.1,'Invalid behavior epsilon');
  const legacy=legacyPath!==null;
  const main=legacy?readLegacy(legacyPath,legacySha):readBundle(modelDir);
  const magnet=legacy?readLegacy(legacyPath,legacySha):readBundle(magnetDir);
  if(!legacy)assert(main.p.policy_id===magnet.p.policy_id&&
    main.p.generation===magnet.p.generation,'Main/Magnet generation mismatch');
  const saved=[];
  let pending=null;
  const core={
    choose({observation,legalActions,seat,round,rng}){
      assert(pending===null,'Unaccepted policy proposal');
      assert([0,1].includes(seat)&&['draft','selection','battle'].includes(observation.phase),
        'Illegal seat/phase');
      let logits,ref,value,phaseInput;
      if(legacy){
        const x=main.inferProjected(observation,legalActions,seat);
        const m=magnet.inferProjected(observation,legalActions,seat);
        logits=x.logits;ref=m.logits;value=x.value;
        // Stored MAIN-visible accepted tensor, even for frozen v3 opponents.
        phaseInput=encodeForLearner(observation,legalActions);
      }else if(observation.phase==='draft'){
        const a=encodeOwnDraftPool(observation,legalActions);
        logits=main.draft.scoreEncoded(a);
        ref=magnet.draft.scoreEncoded(a);
        value=valueFromLayers(draftFeatures(main.draft,a),
          main.values.tensors,'draft_value',64);
        phaseInput=encodeForLearner(observation,legalActions,a);
      }else if(observation.phase==='selection'){
        const a=encodeRichSelectionActor(observation,legalActions);
        logits=scoreFrozenASelection(main.selection,a);
        ref=scoreFrozenASelection(magnet.selection,a);
        value=valueFromLayers(
          selectionValueFeatures(main.selection,a),
          main.values.tensors,'selection_value',64);
        phaseInput=encodeForLearner(observation,legalActions,a);
      }else{
        const a=main.battle.inferProjected(observation,legalActions,seat);
        const m=magnet.battle.inferProjected(observation,legalActions,seat);
        logits=a.logits;ref=m.logits;value=a.value;
        phaseInput=encodeForLearner(observation,legalActions);
      }
      assert(logits.length===legalActions.length&&
        ref.length===legalActions.length&&finite(value),'Bad model/legal candidates');
      const pi=softmax(logits),mp=softmax(ref);
      const probs=pi.map(x=>(1-epsilon)*x+epsilon/pi.length);
      const ix=sample(probs,rng);
      pending={seat,round,phase:observation.phase,
        action:legalActions[ix],index:ix,phase_input:phaseInput,
        policy_pi:pi,magnet_pi:mp,
        predicted_value:value,behavior_prob:probs[ix]};
      return {action:legalActions[ix],behaviorLogProbability:Math.log(probs[ix]),
        referenceLogProbability:Math.log(mp[ix]),predictedValue:value};
    },
    accepted({seat,round,phase,action}){
      const p=pending;
      assert(p&&p.seat===seat&&p.round===round&&p.phase===phase&&
        same(p.action,action),'No matching accepted behavior');
      saved.push(p);pending=null;
    }
  };
  // The ordinary switch-loop guard filters only authoritative legal support,
  // recording that support through the receipt-backed Match collector.
  const guard=new RnadSwitchLoopGuard(core);
  const func=(args)=>guard.choose(args);
  func.accepted=args=>{guard.accepted(args);core.accepted(args);};
  return {func,saved,get pending(){return pending;}};
}
function encodeForLearner(obs,legal,precomputed){
  if(obs.phase==='draft'){
    const a=precomputed??encodeOwnDraftPool(obs,legal);
    const extras=a.codes.map(code=>{
      const spent=code.reduce((sum,i)=>sum+a.prices[i],0);
      return [spent/100,(a.credits-spent)/100,
        code.reduce((sum,i)=>sum+a.carries[i],0)/6,
        code.reduce((sum,i)=>sum+a.megas[i],0)/6];
    });
    return {pool:a.features,codes:a.codes,extra:extras};
  }
  if(obs.phase==='selection'){
    const a=precomputed??encodeRichSelectionActor(obs,legal);
    return {own_cat:a.own_cats,foe_cat:a.foe_cats,
      own_num:a.own_numeric,foe_num:a.foe_numeric,
      legal:a.legal};
  }
  const a=encodeRnadThreePhaseDecisionV5(obs,legal);
  return {state:a.state,actions:a.actions};
}
export function createIntegratedAcceptedMatch({
  seed,learnerSeat,generation,actorId,policyId,
  mainDir,magnetDir,opponentDir=null,
  legacyOpponent=null,legacyOpponentSha=null,
  role='self',mainSha,magnetSha,opponentSha,
  epsilon=.02
}){
  assert(Number.isSafeInteger(seed)&&seed>0&&seed<=0xffffffff&&
    [0,1].includes(learnerSeat)&&Number.isSafeInteger(generation)&&
    Number.isSafeInteger(actorId)&&
    typeof policyId==='string'&&policyId&&
    ['self','historical','fixed','exploiter'].includes(role),
    'Invalid generation/actor/seat');
  const own=session({modelDir:mainDir,magnetDir,epsilon});
  assert(fileSha(join(mainDir,'manifest.json'))===mainSha&&
    fileSha(join(magnetDir,'manifest.json'))===magnetSha,
    'Actor main/magnet SHA not bound to generation');
  if(role!=='self')assert(
    legacyOpponent?fileSha(legacyOpponent)===opponentSha:
      fileSha(join(opponentDir,'manifest.json'))===opponentSha,
    'Frozen opponent does not match generation registry');
  const other=role==='self'?
    session({modelDir:mainDir,magnetDir,epsilon}):
    legacyOpponent?
    session({legacyPath:legacyOpponent,legacySha:legacyOpponentSha,epsilon}):
    session({modelDir:opponentDir,magnetDir:opponentDir,epsilon});
  const sessions=[null,null];sessions[learnerSeat]=own;
  sessions[1-learnerSeat]=other;
  const id=`g${generation}-a${actorId}-s${seed}-seat${learnerSeat}`;
  const result=runRnadAcceptanceSmoke({
    seed,matchId:id,experimentId:'RNAD-I5-IntegratedPolicyV1-Research',
    sourceSha:verifiedRunningSourceSha(),
    engineSnapshotSha:provenance.sourceCommit,
    phasePolicies:sessions.map(x=>x.func),
    policyVersions:sessions.map((x,i)=>i===learnerSeat?
      policyId:role+'-frozen-'+opponentSha.slice(0,12)),
    includeAcceptedHistory:true,
  });
  assert(result.completed===true&&result.score[result.winner]===4&&
    result.jointTimeline?.length>0&&result.trajectories?.length===2&&
    sessions.every(x=>!x.pending),
    'Missing authoritative accepted completed Match');
  const indices=[0,0],events=[];
  for(let i=0;i<result.jointTimeline.length;i++){
    const {seat,index}=result.jointTimeline[i];
    assert(index===indices[seat]++,'Crossed accepted seat timeline');
    const accepted=result.trajectories[seat].decisions[index];
    const p=sessions[seat].saved[index];
    assert(p&&p.seat===seat&&p.phase===accepted.phase&&
      same(p.action,accepted.accepted_action)&&
      Math.abs(Math.log(p.behavior_prob)-accepted.behavior_log_probability)<1e-8,
      'Untrusted model proposal/authority acceptance mismatch');
    events.push({actor:seat,phase:accepted.phase,
      phase_input:p.phase_input,action_index:p.index,
      behavior_prob:p.behavior_prob,magnet_log_prob:
        accepted.reference_log_probability,policy_pi:p.policy_pi,
      magnet_pi:p.magnet_pi,predicted_value:p.predicted_value,
      reward:[0,0]});
  }
  assert(events.length===result.actionCounts[0]+result.actionCounts[1]+
    4*result.rounds,'Missing accepted draft/selection/battle phases');
  const payoff=result.winner===0?1:-1;
  events.at(-1).reward=[payoff,-payoff];
  return {schema:MATCH_SCHEMA,match_id:id,policy_id:policyId,
    main_sha256:mainSha,magnet_sha256:magnetSha,
    opponent_sha256:opponentSha,source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,seed,learner_seat:learnerSeat,
    completed:true,winner:result.winner,score:result.score,rounds:result.rounds,
    action_counts:result.actionCounts,events};
}
// This CLI writes each accepted Match in a compressed one-use artifact.
// The Actor is read-only and cannot invoke any optimizer or change a checkpoint.
if(process.argv[1]?.endsWith('/shared_league_actor.mjs')){
  const argv=process.argv.slice(2),a={};
  for(let i=0;i<argv.length;i+=2){const k=argv[i],v=argv[i+1];
    if(!k?.startsWith('--')||v===undefined)throw Error('Malformed CLI');
    a[k.slice(2)]=v;
  }
  const meta=JSON.parse(readFileSync(a.meta));
  const runOne=(seed,seat,out)=>{
    assert(Number.isSafeInteger(seed)&&[0,1].includes(seat)&&
      typeof out==='string'&&out.endsWith('.json.gz'),
      'Invalid independent seed/seat/output in actor batch');
    const match=createIntegratedAcceptedMatch({
      seed,learnerSeat:seat,
      generation:meta.generation,actorId:meta.actor_id,
      policyId:meta.policy_id,mainDir:a.main,magnetDir:a.magnet,
      opponentDir:a.opponent??null,legacyOpponent:a.legacy??null,
      legacyOpponentSha:meta.legacy_sha256??null,role:meta.role,
      mainSha:meta.main_sha256,magnetSha:meta.magnet_sha256,
      opponentSha:meta.opponent_sha256});
    writeFileSync(out,gzipSync(JSON.stringify(match)),{flag:'wx'});
    process.stderr.write(JSON.stringify({
      seed:match.seed,seat:match.learner_seat,rounds:match.rounds,
      accepted:match.events.length,winner:match.winner})+'\n');
  };
  if(a['batch-jobs']){
    assert(a.seed===undefined&&a.seat===undefined&&a.out===undefined,
      'Batch and single Match CLI inputs cannot be combined');
    const manifest=JSON.parse(readFileSync(a['batch-jobs']));
    assert(manifest?.schema==='rnad-shared-actor-batch-jobs-v1'&&
      Array.isArray(manifest.jobs)&&manifest.jobs.length>=1&&
      manifest.jobs.length<=64,'Invalid bounded actor batch');
    const unique=new Set();
    for(const job of manifest.jobs){
      assert(job&&Object.keys(job).sort().join(',')==='out,seat,seed'&&
        Number.isSafeInteger(job.seed)&&
        Number.isSafeInteger(job.seat),'Invalid batch job');
      const id=job.seed+':'+job.seat;
      assert(!unique.has(id),'Duplicate actor batch seed/seat');
      unique.add(id);
      runOne(job.seed,job.seat,job.out);
    }
  }else{
    runOne(Number(a.seed),Number(a.seat),a.out);
  }
}
