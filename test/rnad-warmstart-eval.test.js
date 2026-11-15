import test from 'node:test';
import assert from 'node:assert/strict';
import {compareMatchedWarmstart} from '../scripts/rnad/compare_warmstart_eval.mjs';

const SHA='a'.repeat(40),ENGINE='b'.repeat(40);
function fixture({seed,model,wins=[true,false],blockPairs=2}={}){
  const rows=[];
  for(let i=0;i<blockPairs;i++)for(const seat of [0,1]){
    const won=wins[seat];
    rows.push({seed:seed+i,candidateSeat:seat,won,rounds:5,
      score:won?(seat===0?[4,2]:[2,4]):(seat===0?[2,4]:[4,2])});
  }
  return {schema:'rnad-paired-full-match-eval-v1',
    source_sha:SHA,engine_sha:ENGINE,
    model_sha256:model.repeat(64),opponent_id:'stage0-fixed-untrained',
    seed,pairs:blockPairs,matches:2*blockPairs,
    wins:rows.filter(row=>row.won).length,
    win_rate:rows.filter(row=>row.won).length/rows.length,rows};
}
const inputs=()=>[
  fixture({seed:1000,model:'c'}),fixture({seed:1002,model:'c'}),
  fixture({seed:1000,model:'d'}),fixture({seed:1002,model:'d'})
];

test('paired comparison uses seed-level uncertainty and exact matched seats',()=>{
  const result=compareMatchedWarmstart(...inputs(),{blockPairs:2});
  assert.equal(result.matches_per_model,8);
  assert.equal(result.distinct_seed_pairs,4);
  assert.equal(result.paired_win_rate_gain,0);
  assert.equal(result.is_champion,false);
  assert.equal(result.gain_statistically_established,false);
  assert.equal(result.degradation_statistically_established,false);
});

test('tampered victory, invalid score, count and swapped seed fail closed',()=>{
  for(const mutate of [
    r=>{r.rows[0].won=!r.rows[0].won;},
    r=>{r.rows[0].score=[4,4];},
    r=>{r.wins++;},
    r=>{r.rows[0].seed+=10;},
    r=>{r.rows[0].rounds=0;},
    r=>{r.model_sha256='invalid';},
    r=>{r.source_sha='0'.repeat(40);r.engine_sha='0'.repeat(40);r.wins++;},
  ]){
    const args=structuredClone(inputs());
    mutate(args[0]);
    assert.throws(()=>compareMatchedWarmstart(...args,{blockPairs:2}));
  }
});

test('duplicate seed and mismatched evaluation model are rejected',()=>{
  let args=inputs();
  args[1].seed=1001;
  args[1].rows.forEach(row=>row.seed--);
  assert.throws(()=>compareMatchedWarmstart(...args,{blockPairs:2}),
    /Reused|Malformed|Incomplete/);
  args=inputs();args[3].model_sha256='e'.repeat(64);
  assert.throws(()=>compareMatchedWarmstart(...args,{blockPairs:2}),/Mixed/);
  args=inputs();args[2].model_sha256='c'.repeat(64);
  args[3].model_sha256='c'.repeat(64);
  assert.throws(()=>compareMatchedWarmstart(...args,{blockPairs:2}),/identical/);
  args=inputs();args[2].engine_sha='0'.repeat(40);
  assert.throws(()=>compareMatchedWarmstart(...args,{blockPairs:2}),
    /Mismatched|Mixed/);
});

test('known weaker R-NaD is not automatically promoted or misclassified',()=>{
  const t=inputs();
  for(const r of t.slice(2))for(const row of r.rows){
    row.won=false;row.score=row.candidateSeat===0?[1,4]:[4,1];
    r.wins=0;r.win_rate=0;
  }
  const v=compareMatchedWarmstart(...t,{blockPairs:2});
  assert.equal(v.paired_win_rate_gain,-.5);
  assert.equal(v.gain_statistically_established,false);
  assert.equal(v.is_champion,false);
});
