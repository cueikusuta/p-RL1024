import test from 'node:test';
import assert from 'node:assert/strict';
import {COMPOSITE_MODES,COMPOSITE_SCHEMA,
  evaluateCompositeAgainstOriginalAQ,createCompositeAQSession}
  from '../scripts/rnad/evaluate-composite-vs-aq.mjs';

test('phase mixing is pinned, bounded and not a fabricated AQ win',()=>{
  assert.equal(COMPOSITE_SCHEMA,'rnad-v2-v3-composite-vs-original-aq-paired-v1');
  assert.deepEqual(COMPOSITE_MODES['v3-battle'],
    {draft:0,selection:0,battle:1});
  assert.deepEqual(COMPOSITE_MODES['v2-battle'],
    {draft:1,selection:1,battle:0});
  assert.equal(COMPOSITE_MODES['blend-half'].battle,.5);
  const control=COMPOSITE_MODES['all-v3'];
  assert.deepEqual(control,{draft:1,selection:1,battle:1});
  assert.deepEqual(COMPOSITE_MODES['all-v2'],{draft:0,selection:0,battle:0});
  for(const phase of ['draft','selection','battle']){
    const arm=COMPOSITE_MODES[phase+'-v2'];
    assert.ok(arm,'Missing independently ablated phase: '+phase);
    for(const p of ['draft','selection','battle'])
      assert.equal(arm[p],p===phase?0:1,
        'Only the named phase may switch from v3');
  }
});
test('composite original AQ actor rejects invalid input before full simulation',()=>{
  const valid={v2Path:'absent-v2.json',v3Path:'absent-v3.json',
    seed:1980001,pairs:8,mode:'blend-half'};
  for(const bad of [
    {...valid,v2Path:''},{...valid,v3Path:null},
    {...valid,pairs:0},{...valid,pairs:101},
    {...valid,seed:0},{...valid,seed:0xffffffff,pairs:2},
    {...valid,mode:'bad'}
  ])assert.throws(()=>evaluateCompositeAgainstOriginalAQ(bad),
    /Invalid bounded composite/);
  assert.throws(()=>evaluateCompositeAgainstOriginalAQ(valid),/ENOENT/);
  assert.throws(()=>createCompositeAQSession({
    v2Path:valid.v2Path,v3Path:valid.v3Path,seat:2,mode:valid.mode}),
    /Invalid two-model AQ/);
});
