import test from 'node:test';
import assert from 'node:assert/strict';
import {compareV4PhaseArms,compareV4PhaseModels}
  from '../scripts/rnad/compare-v4-phase-arms.mjs';
import {COMPOSITE_SCHEMA,COMPOSITE_MODES}
  from '../scripts/rnad/evaluate-v4-composite-vs-aq.mjs';
import {weightsSha256} from '../models/aq-base.mjs';
import provenance from '../provenance/manifest.json' with {type:'json'};
const report=(mode,winBoth=false)=>({
  schema:COMPOSITE_SCHEMA,
  source_sha:'a'.repeat(40),engine_sha:provenance.sourceCommit,
  aq_base_model_sha256:weightsSha256,
  v4_model_sha256:'b'.repeat(64),v3_model_sha256:'c'.repeat(64),
  seed:401001,pairs:4,matches:8,
  wins:winBoth?8:4,win_rate:winBoth?1:.5,mode,
  policy:'frozen-greedy-v4-v3-anti-loop-v1',
  champion_promoted:false,
  rows:Array.from({length:4},(_,i)=>[0,1].map(seat=>{
    const won=winBoth||seat===0;
    return {seed:401001+i,candidateSeat:seat,won,
      score:won?(seat===0?[4,2]:[1,4]):[4,1],
      rounds:5,accepted_decisions:70};
  })).flat()
});
test('v4 phase comparison uses seed-paired difference without declaring victory',()=>{
  const a=report('all-v3'),b=report('selection-v4',true);
  const r=compareV4PhaseArms(a,b);
  assert.equal(r.independent_seeds,4);
  assert.equal(r.matches_per_arm,8);
  assert.equal(r.delta,.5);
  assert.equal(r.improved_with_95pct_confidence,false);
  assert.equal(r.strength_established,false);
  assert.equal(r.champion_promoted,false);
});
test('v4 phase comparison strictly rejects missing, duplicate and forged evidence',()=>{
  const a=report('all-v3'),b=report('both-v4',true);
  for(const row of [
    {...b,source_sha:'d'.repeat(40)},
    {...b,seed:401002},
    {...b,rows:b.rows.slice(1)},
    {...b,wins:7},
    {...b,mode:'all-v3'},
    {...b,v3_model_sha256:'e'.repeat(64)},
    {...b,rows:b.rows.map((x,i)=>i===0?{...x,candidateSeat:1}:x)},
  ])assert.throws(()=>compareV4PhaseArms(a,row));
});

test('same-phase stronger v4 model is compared on exact seats and frozen v3',()=>{
  const base=report('selection-v4'),candidate=report('selection-v4',true);
  candidate.v4_model_sha256='d'.repeat(64);
  const r=compareV4PhaseModels(base,candidate);
  assert.equal(r.paired_delta,.5);
  assert.equal(r.matches_per_model,8);
  assert.equal(r.gain_established,false);
  assert.equal(r.champion_promoted,false);
  assert.throws(()=>compareV4PhaseModels(base,{...candidate,seed:404001}));
  assert.throws(()=>compareV4PhaseModels(base,{...candidate,mode:'draft-v4'}));
  assert.throws(()=>compareV4PhaseModels(base,{...candidate,
    v3_model_sha256:'e'.repeat(64)}));
});

test('P2 multiple phase comparisons widen familywise CI but never auto-promote',()=>{
  const a=report('all-v3'),b=report('selection-v4',true);
  const raw=compareV4PhaseArms(a,b);
  const multi=compareV4PhaseArms(a,b,{familySize:6});
  assert.equal(multi.familywise_comparisons,6);
  assert.ok(multi.familywise_95pct_delta_bounds[0] <=
            raw.conservative_95pct_delta_bounds[0]);
  assert.ok(multi.familywise_95pct_delta_bounds[1] >=
            raw.conservative_95pct_delta_bounds[1]);
  assert.equal(multi.exploratory_screen_only,true);
  assert.equal(multi.familywise_improvement_established,false);
  assert.throws(()=>compareV4PhaseArms(a,b,{familySize:0}));
  const c=report('selection-v4',true);
  c.v4_model_sha256='d'.repeat(64);
  const model=compareV4PhaseModels(report('selection-v4'),c,{familySize:6});
  assert.equal(model.familywise_comparisons,6);
  assert.equal(model.exploratory_screen_only,true);
});

test('selection ensemble has a monotonically increasing v4 share',()=>{
  const modes=['all-v3','selection-quarter-v4','selection-half-v4',
    'selection-threequarter-v4','selection-v4'];
  assert.deepEqual(modes.map(mode=>COMPOSITE_MODES[mode].selection),
    [1,.75,.5,.25,0]);
  assert.ok(modes.every(mode=>
    COMPOSITE_MODES[mode].draft===1&&COMPOSITE_MODES[mode].battle===1));
});
