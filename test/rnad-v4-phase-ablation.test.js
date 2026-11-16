import test from 'node:test';
import assert from 'node:assert/strict';
import {compareV4PhaseArms}
  from '../scripts/rnad/compare-v4-phase-arms.mjs';
import {COMPOSITE_SCHEMA}
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
