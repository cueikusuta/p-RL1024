import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluateAgainstOriginalAQ,createRnadAQMatchSession,
  FULL_AQ_EVALUATION_SCHEMA}
  from '../scripts/rnad/evaluate-vs-aq-full.mjs';
import {weightsSha256} from '../models/aq-base.mjs';
import {pairedHoeffding95} from '../scripts/rnad/aggregate-eval.mjs';

test('original AQ benchmark is pinned, never replaced by proxy teacher',()=>{
  assert.equal(FULL_AQ_EVALUATION_SCHEMA,
    'rnad-vs-original-aq-base-full-match-paired-v1');
  assert.equal(weightsSha256,
    '5ccc11a3362c7f5a8874847e3afc1f19c539d4ef258bd82704f53d8a1db1946e');
  const interval=pairedHoeffding95(0,2);
  assert.equal(interval[0],0);
  assert.ok(interval[1]>0&&interval[1]<1);
  assert.ok(interval[1]===Math.sqrt(Math.log(40)/4));
});

test('original AQ evaluation rejects unbounded/invalid/missing models before Match',()=>{
  const valid={modelPath:'unavailable-frozen-rnad-v2.json',seed:1970001,
    pairs:8,phaseMode:'neural-all',switchGuard:'anti-loop-v1'};
  for(const args of [
    {...valid,seed:0},{...valid,seed:0xffffffff,pairs:2},
    {...valid,pairs:0},{...valid,pairs:101},{...valid,pairs:1.5},
    {...valid,phaseMode:'aq-proxy-full'},
    {...valid,switchGuard:'unsafe'},
    {...valid,modelPath:''}
  ])assert.throws(()=>evaluateAgainstOriginalAQ(args),
    /Invalid bounded original-AQ paired evaluation/);
  assert.throws(()=>evaluateAgainstOriginalAQ(valid),/ENOENT/);
  assert.throws(()=>createRnadAQMatchSession({
    ...valid,seat:2}),/Invalid bounded full-AQ R-NaD adapter/);
});
