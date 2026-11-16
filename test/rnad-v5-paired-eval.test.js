import test from 'node:test';
import assert from 'node:assert/strict';
import {V5_AQ_SCHEMA} from '../scripts/rnad/evaluate-v5-composite-vs-aq.mjs';
import {compareV5PhaseAgainstV3}
  from '../scripts/rnad/compare-v5-phase-arms.mjs';
import {weightsSha256 as AQ_SHA} from '../models/aq-base.mjs';
import provenance from '../provenance/manifest.json' with {type:'json'};
function fixture(mode='all-v3',wins=[true,false,false,true]){
  const rows=[];
  for(let i=0;i<2;i++)for(const seat of [0,1]){
    const won=wins[2*i+seat];
    const score=seat===0?(won?[4,1]:[1,4]):(won?[1,4]:[4,1]);
    rows.push({seed:2210001+i,candidateSeat:seat,won,score,
      rounds:5,accepted_decisions:100});
  }
  return {schema:V5_AQ_SCHEMA,source_sha:'a'.repeat(40),
    engine_sha:provenance.sourceCommit,
    aq_base_model_sha256:AQ_SHA,
    v3_model_sha256:'b'.repeat(64),
    v5_model_sha256:'c'.repeat(64),
    policy:'frozen-greedy-v5-v3-anti-loop-v1',
    mode,seed:2210001,pairs:2,matches:4,
    wins:wins.filter(Boolean).length,
    win_rate:wins.filter(Boolean).length/4,
    champion_promoted:false,strength_established:false,rows};
}
test('v5 paired AQ screen records exact two seat/seed and no champion',()=>{
  const base=fixture(),testArm=fixture('selection-v5',[true,true,false,true]);
  const result=compareV5PhaseAgainstV3(base,testArm,{familySize:3});
  assert.equal(result.delta,.25);
  assert.equal(result.independent_seed_pairs,2);
  assert.equal(result.strength_established,false);
  assert.equal(result.champion_promoted,false);
});
test('v5 paired AQ screen rejects missing/duplicate seat, SHA swap and wrong scores',()=>{
  const base=fixture();
  for(const mutation of [
    x=>{x.rows[1].seed=x.rows[0].seed;x.rows[1].candidateSeat=x.rows[0].candidateSeat;},
    x=>{x.v5_model_sha256='f'.repeat(64);},
    x=>{x.rows[2].score=[4,0];},
    x=>{x.source_sha='bad';}
  ]){
    const cand=fixture('selection-v5');
    mutation(cand);
    assert.throws(()=>compareV5PhaseAgainstV3(base,cand));
  }
});
