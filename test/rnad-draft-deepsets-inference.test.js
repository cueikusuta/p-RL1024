import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DeepSetsDraftPolicy,BlendedDeepSetsDraftPolicy,
  CreditAwareDeepSetsDraftPolicy,encodeOwnDraftPool,
  DRAFT_DEEPSETS_FEATURE_WIDTH}
  from '../scripts/rnad/draft-deepsets-infer.mjs';
import {compareDeepSetsDraftAQ,compareDeepSetsDraftVersions,DRAFT_AQ_SCREEN_SCHEMA}
  from '../scripts/rnad/evaluate-deepsets-draft-vs-aq.mjs';
import {comparePublicDraftAgainstV3}
  from '../scripts/rnad/compare-public-draft-aq.mjs';

const dims={
  'phi.0.weight':[48,44],'phi.0.bias':[48],
  'phi.2.weight':[48,48],'phi.2.bias':[48],
  'rho.0.weight':[64,148],'rho.0.bias':[64],
  'rho.2.weight':[32,64],'rho.2.bias':[32],
  'rho.4.weight':[1,32],'rho.4.bias':[1],
};
function withZeroModel(run){
  const dir=mkdtempSync(join(tmpdir(),'rnad-i5-draft-'));
  const file=join(dir,'draft.json');
  const data={schema:'rnad-draft-deepsets-weights-v1',
    architecture:'deepsets-soft',feature_width:DRAFT_DEEPSETS_FEATURE_WIDTH,
    source_sha:'a'.repeat(40),
    weights:Object.fromEntries(Object.entries(dims).map(([name,shape])=>
      [name,{shape,values:Array(shape.reduce((a,b)=>a*b,1)).fill(0)}]))};
  writeFileSync(file,JSON.stringify(data));
  try{run(file,data)}finally{rmSync(dir,{recursive:true,force:true})}
}
function observation(){
  const pool=Array.from({length:7},(_,i)=>({
    setId:'unit-'+i,instanceId:'own-'+i,
    types:i%2===0?['Normal']:['Fire'],
    baseStats:{hp:80,atk:95,def:92,spa:110,spd:86,spe:80+i},
    moves:[],cost:i%2+1,mega:null
  }));
  return {phase:'draft',self:{carry:[pool[0]],fresh:pool.slice(1),credits:10},
    opponent:{secret:{ability:'not-public',team:['secret-mon']}}};
}
const legal=[
  ['own-0','own-1','own-2','own-3','own-4','own-5'],
  ['own-0','own-1','own-2','own-3','own-4','own-6']
];
test('frozen DeepSets inference uses only seat own-set features and legal pool',()=>{
  withZeroModel(path=>{
    const p=new DeepSetsDraftPolicy(path);
    const a=observation();
    const one=encodeOwnDraftPool(a,legal);
    a.opponent.secret.team[0]='other-unrevealed-secret';
    const two=encodeOwnDraftPool(a,legal);
    assert.deepEqual(one,two);
    assert.equal(one.features[0].length,44);
    assert.deepEqual(one.codes[0],[0,1,2,3,4,5]);
    const prediction=p.choose(a,legal);
    assert.equal(prediction.index,0);
    assert.ok(Math.abs(prediction.logProbability+Math.log(2))<1e-10);
    const swapped={...one,codes:one.codes.map(x=>[...x].reverse())};
    assert.deepEqual(p.scoreEncoded(one),p.scoreEncoded(swapped));
    assert.throws(()=>p.choose(a,[legal[0].slice(0,5)]));
    assert.throws(()=>p.choose(a,[['hidden-1',...legal[0].slice(1)]]));
    assert.throws(()=>new DeepSetsDraftPolicy(path,{sha256:'f'.repeat(64)}));
  });
});
test('invalid or tampered model tensor fails before policy deployment',()=>{
  withZeroModel((path,m)=>{
    m.weights['rho.0.bias'].values.pop();
    writeFileSync(path,JSON.stringify(m));
    assert.throws(()=>new DeepSetsDraftPolicy(path),/tensor/);
  });
});
const report=(mode,wins)=>({
  schema:DRAFT_AQ_SCREEN_SCHEMA,
  source_sha:'a'.repeat(40),engine_sha:'b'.repeat(40),
  benchmark:'AQ-Base-001 exact integrated strategy-match',
  aq_base_model_sha256:'c'.repeat(64),
  draft_model_sha256:'d'.repeat(64),
  v4_model_sha256:'e'.repeat(64),v3_model_sha256:'f'.repeat(64),
  mode,seed:2444001,pairs:4,matches:8,wins,win_rate:wins/8,
  strength_established:false,champion_promoted:false,
  rows:Array.from({length:4},(_,i)=>[0,1].map(seat=>{
    const index=i*2+seat,won=index<wins;
    return {seed:2444001+i,seat,won,rounds:5,
      score:won?(seat===0?[4,1]:[1,4]):(seat===0?[1,4]:[4,1]),
      accepted_decisions:45};
  })).flat()
});
test('draft paired A/B needs both seats and does not auto promote',()=>{
  const base=report('all-v3',3);
  const candidate=report('draft-deepsets-soft',5);
  const r=compareDeepSetsDraftAQ(base,candidate);
  assert.equal(r.paired_delta,.25);
  assert.equal(r.independent_seeds,4);
  assert.equal(r.strength_established,false);
  assert.equal(r.champion_promoted,false);
  assert.throws(()=>compareDeepSetsDraftAQ(base,{
    ...candidate,rows:candidate.rows.slice(1)}));
  assert.throws(()=>compareDeepSetsDraftAQ(base,{
    ...candidate,source_sha:'x'.repeat(40)}));
});

test('same-training-source draft 16/32 checkpoints require matched both-seat seeds',()=>{
  const medium=report('draft-deepsets-soft',2);
  const full=report('draft-deepsets-soft',5);
  medium.draft_training_source_sha='a'.repeat(40);
  full.draft_training_source_sha='a'.repeat(40);
  full.draft_model_sha256='b'.repeat(64);
  const outcome=compareDeepSetsDraftVersions(medium,full);
  assert.equal(outcome.paired_delta,.375);
  assert.equal(outcome.strength_established,false);
  assert.throws(()=>compareDeepSetsDraftVersions(medium,
    {...full,draft_training_source_sha:'e'.repeat(40)}));
  assert.throws(()=>compareDeepSetsDraftVersions(medium,
    {...full,seed:2444002}));
});

test('bounded blended draft is finite, strictly legal and requires v3 context',()=>{
  withZeroModel(path=>{
    const deep=new DeepSetsDraftPolicy(path);
    assert.throws(()=>new BlendedDeepSetsDraftPolicy(deep,{deepShare:0.6}));
    const blend=new BlendedDeepSetsDraftPolicy(deep,{deepShare:.5});
    const obs=observation();
    const baseline=[-2,3];
    const chosen=blend.choose(obs,legal,baseline);
    assert.equal(chosen.index,1);
    assert.ok(Number.isFinite(chosen.logProbability));
    assert.equal(chosen.candidateCount,2);
    assert.throws(()=>blend.choose(obs,legal,[1]),/reference logits/);
    assert.throws(()=>blend.choose(obs,legal,[Number.NaN,3]));
    assert.throws(()=>blend.choose(obs,[legal[0].slice(0,5)],baseline));
    assert.deepEqual(blend.choose({...obs,opponent:{invisible:'secret'}},
      legal,baseline),chosen);
  });
});

test('same-seed public AQ draft probe is exploratory and strictly paired',()=>{
  const base=report('all-v3',3);
  const raw=report('all-v3',5);
  const teacher={...raw,
    schema:'rnad-vs-original-aq-base-full-match-paired-v1',
    phase_mode:'aq-public-draft-neural-rest',
    model_sha256:base.v3_model_sha256,
    switch_guard:'anti-loop-v1',evaluation_policy:'greedy-no-training',
    rows:raw.rows.map(({seat,...rest})=>({...rest,candidateSeat:seat}))
  };
  const result=comparePublicDraftAgainstV3(base,teacher);
  assert.equal(result.paired_delta,.25);
  assert.equal(result.strength_established,false);
  assert.throws(()=>comparePublicDraftAgainstV3(base,{
    ...teacher,seed:2444002}));
  assert.throws(()=>comparePublicDraftAgainstV3(base,{
    ...teacher,rows:teacher.rows.slice(1)}));
});

test('opt-in credit calibration uses only own costs and legal support',()=>{
  withZeroModel(path=>{
    const deep=new DeepSetsDraftPolicy(path);
    assert.throws(()=>new CreditAwareDeepSetsDraftPolicy(deep,{costWeight:2}));
    const actor=new CreditAwareDeepSetsDraftPolicy(deep,{costWeight:.5});
    const obs=observation();
    const selected=actor.choose(obs,legal);
    assert.ok([0,1].includes(selected.index));
    assert.ok(Number.isFinite(selected.logProbability));
    assert.equal(actor.modelSha256,deep.modelSha256);
    const secrets={...obs,opponent:{unrevealed:{team:['secret']}}};
    assert.deepEqual(actor.choose(secrets,legal),selected);
    assert.throws(()=>actor.choose(obs,[legal[0].slice(1)]));
  });
});
