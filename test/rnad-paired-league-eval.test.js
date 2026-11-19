import test from 'node:test';
import assert from 'node:assert/strict';
import {studentTCdf,studentTQuantile,pairedClusterCI,pairedSignSensitivity}
  from '../scripts/rnad/paired-league-stats.mjs';
import {evaluatePairedLeague,PAIRED_LEAGUE_EVIDENCE_SCHEMA,
  PAIRED_LEAGUE_PLAN_SCHEMA} from '../scripts/rnad/paired-league-eval.mjs';

const SOURCE='a'.repeat(40),ENGINE='b'.repeat(40),START=7100001,N=128;
const sha=id=>id.repeat(64);
const basePlan=(evaluation_type='shared-opponent')=>({
  schema:PAIRED_LEAGUE_PLAN_SCHEMA,stage:'development',
  evaluation_type,source_sha:SOURCE,engine_sha:ENGINE,
  opponent_id:evaluation_type==='direct-head-to-head'?'CHAMPION':'AQ-Base-001',
  seed_start:START,seed_blocks:N,
  familywise_alpha:.20,confidence_level:.8,
  method:'paired-seed-block-student-t-bonferroni',
  fixed_sample_size:true,model_selection_before_holdout:true,
  automatic_champion_promotion:false,minimum_practical_gain:0,
  model_sha256_by_id:{G10:sha('c'),G11:sha('d'),
    ...(evaluation_type==='direct-head-to-head'?{CHAMPION:sha('e')}:{})},
  excluded_seed_ranges:[{start:6200001,count:128}],
  comparisons:[{candidate_model_id:'G11',
    baseline_model_id:evaluation_type==='direct-head-to-head'?'CHAMPION':'G10'}],
});
function matchReport(model_id,pick,opts={}){
  const {seedStart=START,count=N,opponent_id='AQ-Base-001'}=opts;
  const rows=[];
  for(let i=0;i<count;i++)for(const seat of [0,1]){
    const won=pick(i,seat),winner=won?seat:1-seat;
    rows.push({seed:seedStart+i,candidateSeat:seat,won,
      completed:true,rounds:5,accepted_decisions:120,
      score:winner===0?[4,1]:[1,4]});
  }
  return {schema:PAIRED_LEAGUE_EVIDENCE_SCHEMA,model_id,
    model_sha256:sha(model_id==='G10'?'c':model_id==='G11'?'d':'e'),
    source_sha:SOURCE,engine_sha:ENGINE,opponent_id,
    evaluation_policy:'frozen-greedy-no-training',
    training_during_evaluation:false,champion_promoted:false,
    seed_start:seedStart,seed_blocks:count,matches:count*2,
    wins:rows.filter(x=>x.won).length,rows};
}
const old=()=>matchReport('G10',(i,seat)=>seat===0);
const challenger=()=>matchReport('G11',(i,seat)=>i%4!==0||seat===0);

test('Student-t quantile/CDF match known finite-df reference points',()=>{
  assert.ok(Math.abs(studentTCdf(0,9)-.5)<1e-13);
  assert.ok(Math.abs(studentTQuantile(.975,9)-2.26215716)<1e-6);
  assert.ok(Math.abs(studentTCdf(-2.26215716,9)-.025)<1e-6);
  assert.ok(Math.abs(studentTQuantile(.9,511)-1.28322)<.003);
});
test('paired seed-block t uses actual paired variance and Bonferroni width',()=>{
  const values=Array.from({length:128},(_,i)=>i%4===0?0:.5);
  const once=pairedClusterCI(values,{familywiseAlpha:.20});
  const twice=pairedClusterCI(values,{familywiseAlpha:.20,comparisons:2});
  assert.equal(once.independent_seed_blocks,128);
  assert.ok(once.estimate>.3&&once.estimate<.4);
  assert.ok(once.positive_evidence);
  assert.ok(twice.positive_evidence);
  assert.ok(twice.confidence_interval[0]<once.confidence_interval[0]);
  assert.ok(twice.confidence_interval[1]>once.confidence_interval[1]);
  assert.equal(twice.per_comparison_alpha,.1);
  assert.ok(once.confidence_interval[0]>.30);
  assert.equal(once.finite_sample_distribution_free,false);
  assert.throws(()=>pairedClusterCI([0]),/independent seed/);
});
test('paired head-to-head null is 50%, not zero; no 2N independent inflation',()=>{
  const plan=basePlan('direct-head-to-head');
  const r=challenger();r.opponent_id='CHAMPION';
  const result=evaluatePairedLeague(plan,[r]);
  const comparison=result.comparisons[0];
  assert.equal(result.independent_seed_blocks,N);
  assert.equal(result.complete_four_win_matches,256);
  assert.equal(comparison.null_value,.5);
  assert.equal(comparison.estimate,224/256);
  assert.equal(comparison.matches_per_model,256);
  assert.equal(comparison.statistically_supported_gain,true);
  assert.equal(comparison.estimand,'direct-head-to-head-seat-swapped-win-probability');
  assert.equal(result.diagnostic_only,true);
  assert.equal(result.champion_promoted,false);
});
test('shared AQ benchmark compares per-seed differences, not marginal intervals',()=>{
  const result=evaluatePairedLeague(basePlan(),[old(),challenger()]);
  const x=result.comparisons[0];
  assert.equal(x.estimate,96/256);
  assert.equal(x.candidate_wins,224);
  assert.equal(x.baseline_wins,128);
  assert.equal(x.null_value,0);
  assert.equal(x.independent_seed_blocks,128);
  assert.ok(x.confidence_interval[0]>0);
  assert.equal(x.statistically_supported_gain,true);
  assert.equal(result.statistical_gate_passed,true);
  assert.equal(result.champion_promoted,false);
});
test('two comparisons are familywise corrected and both mandatory',()=>{
  const p=basePlan();
  p.comparisons.push({candidate_model_id:'G12',baseline_model_id:'G10'});
  p.model_sha256_by_id.G12=sha('e');
  const report=matchReport('G12',(i,seat)=>seat===0);
  const result=evaluatePairedLeague(p,[old(),challenger(),report]);
  assert.equal(result.comparisons.length,2);
  assert.equal(result.comparisons[0].per_comparison_alpha,.1);
  assert.equal(result.comparisons[1].statistically_supported_gain,false);
  assert.equal(result.statistical_gate_passed,false);
});
test('independent final holdout must be preregistered and cannot overlap consumed seeds',()=>{
  const p=basePlan();p.stage='final';
  assert.throws(()=>evaluatePairedLeague(p,[old(),challenger()]),/disjoint holdout/);
  p.independent_holdout_precommitted=true;
  assert.throws(()=>evaluatePairedLeague(p,[old(),challenger()]),/disjoint holdout/);
  p.seed_blocks=512;
  const x=evaluatePairedLeague(p,[
    matchReport('G10',(i,seat)=>seat===0,{count:512}),
    matchReport('G11',(i,seat)=>i%4!==0||seat===0,{count:512}),
  ]);
  assert.equal(x.diagnostic_only,false);
  assert.equal(x.requires_independent_chatgpt_review,true);
  p.excluded_seed_ranges=[{start:START+10,count:1}];
  assert.throws(()=>evaluatePairedLeague(p,[
    matchReport('G10',(i,seat)=>seat===0,{count:512}),
    matchReport('G11',(i,seat)=>i%4!==0||seat===0,{count:512}),
  ]),/overlaps evaluation/);
});
test('identical checkpoints, changed weights, duplicate or missing seat fail closed',()=>{
  const good=old(),newer=challenger();
  const corruptions=[
    [good,{...newer,model_sha256:good.model_sha256}],
    [good,{...newer,source_sha:'f'.repeat(40)}],
    [good,{...newer,wins:newer.wins+1}],
    [good,{...newer,rows:newer.rows.slice(1)}],
    [good,{...newer,rows:newer.rows.map((x,i)=>i===0?{
      ...x,candidateSeat:1}:x)}],
    [good,{...newer,rows:newer.rows.map((x,i)=>i===0?{
      ...x,score:[0,0]}:x)}],
    [good,{...newer,rows:newer.rows.map((x,i)=>i===0?{
      ...x,completed:false}:x)}],
    [good,{...newer,training_during_evaluation:true}],
  ];
  for(const x of corruptions)assert.throws(()=>evaluatePairedLeague(basePlan(),x));
  assert.throws(()=>evaluatePairedLeague(basePlan(),[good,newer,newer]),
    /duplicate model\/seed\/seat/);
  const wrongPlan=basePlan();
  wrongPlan.model_sha256_by_id.G11='f'.repeat(64);
  assert.throws(()=>evaluatePairedLeague(wrongPlan,[good,newer]),
    /invalid frozen model/);
});
test('shards may be out of order but must cover every seed and both seats',()=>{
  const oldFull=old(),newFull=challenger();
  const a=JSON.parse(JSON.stringify(oldFull)),b=JSON.parse(JSON.stringify(oldFull));
  a.seed_blocks=64;a.matches=128;a.rows=a.rows.slice(0,128);
  a.wins=a.rows.filter(x=>x.won).length;
  b.seed_start=START+64;b.seed_blocks=64;b.matches=128;b.rows=b.rows.slice(128);
  b.wins=b.rows.filter(x=>x.won).length;
  const result=evaluatePairedLeague(basePlan(),[b,newFull,a]);
  assert.equal(result.independent_seed_blocks,N);
  const duplicate=JSON.parse(JSON.stringify(b));duplicate.seed_start=START;
  assert.throws(()=>evaluatePairedLeague(basePlan(),[a,duplicate,newFull]));
});
test('sensitivity sign test reports zero blocks without calling them losses',()=>{
  const p=pairedSignSensitivity([0,0,.5,.5,.5]);
  assert.deepEqual([p.positive,p.negative,p.zero],[3,0,2]);
  assert.ok(Math.abs(p.two_sided_exact_p-.25)<1e-12);
  assert.equal(pairedSignSensitivity([0,0,0]).two_sided_exact_p,1);
});
