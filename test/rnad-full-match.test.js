import test from 'node:test';
import assert from 'node:assert/strict';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha} from '../scripts/dev/simulator/rnad-match.js';
const args={seed:22002,matchId:'full-match-22002',
  sourceSha:verifiedRunningSourceSha(),
  engineSnapshotSha:'ea81165993110ee9fb3a09dcb97c6708c0d7397b'};
test('full authoritative 4-win Match produces two seat-local accepted trajectories',()=>{
  const result=runRnadAcceptanceSmoke(args);
  assert.equal(result.completed,true);
  assert.equal(result.score[result.winner],4);
  assert.ok(result.rounds>=4&&result.rounds<=7);
  assert.equal(result.trajectories.length,2);
  for(let seat=0;seat<2;seat++){
    const data=result.trajectories[seat];
    assert.equal(data.completed,true);
    assert.equal(data.player_seat,seat);
    assert.equal(data.accepted_decisions,result.actionCounts[seat]);
    assert.ok(data.accepted_decisions>0);
    assert.ok(data.decisions.every(row=>
      row.player_seat===seat&&row.terminal_reward===(result.winner===seat?1:-1)&&
      row.phase==='battle'&&row.observation.seat===seat));
  }
  assert.ok(result.trajectories[0].decisions.every(d=>
    !JSON.stringify(d.observation).includes('internalCheckpoint')));
});
