import test from 'node:test';
import assert from 'node:assert/strict';
import {recoverFocusedPhaseExperiment}
  from '../scripts/rnad/recover-phase-focused-eval.mjs';
import {COMPOSITE_SCHEMA} from '../scripts/rnad/evaluate-v4-composite-vs-aq.mjs';
import {weightsSha256 as AQ} from '../models/aq-base.mjs';
import provenance from '../provenance/manifest.json' with {type:'json'};

function report(mode,v4,winsForSeatOne=false){
  const rows=Array.from({length:4},(_,i)=>[0,1].map(seat=>{
    const won=winsForSeatOne||seat===0;
    return {seed:501001+i,candidateSeat:seat,won,
      score:won?(seat===0?[4,2]:[2,4]):[4,2],
      rounds:5,accepted_decisions:70};
  })).flat();
  return {schema:COMPOSITE_SCHEMA,mode,
    source_sha:'a'.repeat(40),engine_sha:provenance.sourceCommit,
    aq_base_model_sha256:AQ,
    v4_model_sha256:v4,v3_model_sha256:'c'.repeat(64),
    seed:501001,pairs:4,matches:8,
    wins:rows.filter(x=>x.won).length,
    win_rate:rows.filter(x=>x.won).length/8,
    policy:'frozen-greedy-v4-v3-anti-loop-v1',
    champion_promoted:false,rows};
}
const evidence=()=>({
  all:{'all-v3':report('all-v3','b'.repeat(64)),
    'selection-v4':report('selection-v4','b'.repeat(64))},
  'selection-only':{'all-v3':report('all-v3','d'.repeat(64)),
    'selection-v4':report('selection-v4','d'.repeat(64),true)}
});
test('postprocess preserves different source-model SHA and uses prior accepted outcomes',()=>{
  const original=evidence(),x=recoverFocusedPhaseExperiment(original);
  assert.equal(x.evidence_reused_not_retrained,true);
  assert.equal(x.strength_established,false);
  assert.equal(x.champion_promoted,false);
  assert.equal(x.all_phase.candidate_wins,4);
  assert.equal(x.selection_only.candidate_wins,8);
  assert.equal(x.selection_only_vs_all.paired_delta,.5);
  assert.equal(original.all['all-v3'].v4_model_sha256,'b'.repeat(64));
  assert.equal(original['selection-only']['all-v3'].v4_model_sha256,'d'.repeat(64));
});
test('postprocess rejects divergent frozen controls and missing accepted rows',()=>{
  const same=evidence();
  same['selection-only']['all-v3'].rows[0].won=false;
  assert.throws(()=>recoverFocusedPhaseExperiment(same));
  const mismatch=evidence();
  mismatch['selection-only']['selection-v4'].rows.pop();
  assert.throws(()=>recoverFocusedPhaseExperiment(mismatch));
});
