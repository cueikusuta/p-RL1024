import test from 'node:test';
import assert from 'node:assert/strict';
import {directPairedNeuralEval} from '../scripts/rnad/evaluate-direct.mjs';
test('direct neural versus neural evaluation rejects invalid and unbounded requests',()=>{
  const valid={teacherPath:'teacher.json',rnadPath:'rnad.json',
    seed:988501,pairs:64};
  for(const args of [
    {...valid,pairs:0},{...valid,pairs:101},
    {...valid,seed:0},{...valid,seed:0xffffffff,pairs:2},
    {...valid,teacherPath:''},{...valid,rnadPath:null},
    {...valid,pairs:1.5},{...valid,seed:1.5}
  ]){
    assert.throws(()=>directPairedNeuralEval(args),
      /Invalid bounded neural-versus-neural evaluation/);
  }
});
test('direct evaluation fails closed on absent or wrong-path models',()=>{
  assert.throws(()=>directPairedNeuralEval({
    teacherPath:'file-not-present-teacher.json',
    rnadPath:'file-not-present-rnad.json',
    seed:988501,pairs:1}),/ENOENT/);
});
