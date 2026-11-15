import test from 'node:test';
import assert from 'node:assert/strict';
import {compareDeploymentModes} from '../scripts/rnad/compare-deployment-modes.mjs';
function example(wins=64,selectionMode='sample',pairs=128){
  const rows=[];
  for(let seed=10;seed<10+pairs;seed++)for(const candidateSeat of [0,1]){
    const won=rows.length<wins;
    rows.push({seed,candidateSeat,won,rounds:5,
      score:won?[4,2].map((n,i)=>candidateSeat===0?n:[4,2][1-i]):
        candidateSeat===0?[2,4]:[4,2]});
  }
  return {schema:'rnad-three-phase-vs-basic-paired-v1',
    model_sha256:'c'.repeat(64),engine_sha:'a'.repeat(40),
    opponent_id:'basic-all-phases',pairs,matches:2*pairs,
    wins,win_rate:wins/(2*pairs),
    inference_selection_mode:selectionMode,rows,champion_promoted:false};
}
test('same paired seed blocks validate improvement conservatively',()=>{
  const sample=example(15,'sample'),greedy=example(130,'greedy');
  const result=compareDeploymentModes(sample,greedy);
  assert.equal(result.independent_seed_blocks,128);
  assert.equal(result.sample_wins,15);
  assert.equal(result.greedy_wins,130);
  assert.equal(result.improvement_statistically_established,true);
});
test('fail closed on mismatched model, opponent, seed, duplicate and forged score',()=>{
  const a=example(15,'sample'),b=example(130,'greedy');
  const changes=[
    b=>b.model_sha256='d'.repeat(64),
    b=>b.engine_sha='b'.repeat(40),
    b=>b.opponent_id='stage0',
    b=>b.rows[1].seed=b.rows[0].seed+10,
    b=>b.rows[3].won=!b.rows[3].won,
    b=>b.rows[4].score=[4,4],
    b=>b.rows[5].rounds=0,
    b=>b.win_rate=.123,
    b=>b.inference_selection_mode='sample',
    b=>b.rows[0].candidateSeat=1
  ];
  for(const fn of changes){
    const x=structuredClone(b);fn(x);
    assert.throws(()=>compareDeploymentModes(a,x));
  }
});
