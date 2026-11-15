import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtempSync,writeFileSync,unlinkSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {mixSeed} from '../models/aq-runner.mjs';
import {verifyTrainingCheckpoints,verifyReportTiming,verifyShardClock} from '../scripts/hosted-proof.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
const sourceSha256='a'.repeat(64),seed=975020;
const options={model:'RL-Deep',opponent:'AQ-Base-001',count:9,seed,sourceSha256};
function fake(){
  const dir=mkdtempSync(join(tmpdir(),'hosted-proof-'));
  const history=Array.from({length:8},(_,i)=>({
    seed:mixSeed(seed,i,0xa975),studentSeat:i%2,winner:0,score:[4,2]
  }));
  const state={schema:'p-rl1024-aq-checkpoint-v1',modelId:'RL-Deep',matches:9,
    seed,sourceSha256,history};
  const cp={...state,checkpointSha256:sha(JSON.stringify(state))};
  const file=join(dir,'training-m8.json');
  writeFileSync(file,JSON.stringify(cp)+'\n');
  return {dir,cp,file};
}
test('full checkpoint sequence validates',()=>{
  const f=fake();try{
    const rows=verifyTrainingCheckpoints(f.dir,options);
    assert.equal(rows.length,1);assert.equal(rows[0].completedMatches,8);
  }finally{rmSync(f.dir,{recursive:true,force:true});}
});
test('missing every checkpoint fails',()=>{
  const f=fake();try{
    unlinkSync(f.file);
    assert.throws(()=>verifyTrainingCheckpoints(f.dir,options),/Missing/);
  }finally{rmSync(f.dir,{recursive:true,force:true});}
});
test('missing checkpoint SHA fails',()=>{
  const f=fake();try{
    delete f.cp.checkpointSha256;writeFileSync(f.file,JSON.stringify(f.cp));
    assert.throws(()=>verifyTrainingCheckpoints(f.dir,options),/Missing checkpoint SHA/);
  }finally{rmSync(f.dir,{recursive:true,force:true});}
});
test('tampered checkpoint fails',()=>{
  const f=fake();try{
    f.cp.history[0].winner=1;writeFileSync(f.file,JSON.stringify(f.cp));
    assert.throws(()=>verifyTrainingCheckpoints(f.dir,options),/SHA mismatch/);
  }finally{rmSync(f.dir,{recursive:true,force:true});}
});
test('resigned checkpoint cannot change expected match count',()=>{
  const f=fake();try{
    f.cp.matches=10;const {checkpointSha256,...payload}=f.cp;
    f.cp.checkpointSha256=sha(JSON.stringify(payload));writeFileSync(f.file,JSON.stringify(f.cp));
    assert.throws(()=>verifyTrainingCheckpoints(f.dir,options),/configuration mismatch/);
  }finally{rmSync(f.dir,{recursive:true,force:true});}
});
test('frozen evaluation timing rejects invalid durations, timestamps and workers',()=>{
  const report={startedAtMs:Date.now(),durationMs:1000,workers:4};
  assert.ok(verifyReportTiming(report)>report.startedAtMs);
  for(const patch of [{durationMs:NaN},{durationMs:-5},{durationMs:0},{durationMs:Infinity},
    {startedAtMs:NaN},{startedAtMs:null},{workers:0},{workers:5}])
    assert.throws(()=>verifyReportTiming({...report,...patch}),/timing/);
});

test('shard clock must use the exact index, seed, run and finite measured wallclock',()=>{
  const x={schema:'p-rl1024-hosted-shard-clock-v1',model:'RL-Deep',
    opponent:'AQ-Base-001',shardIndex:0,offset:0,count:13,seed:975020,
    startedUnix:1794704599,finishedUnix:1794704602,durationSeconds:3,exitCode:0,runId:'123',runAttempt:'2'};
  const expected={model:'RL-Deep',opponent:'AQ-Base-001',index:0,offset:0,count:13,seed:975020,runId:'123',runAttempt:'2'};
  assert.doesNotThrow(()=>verifyShardClock(x,expected));
  for(const patch of [{shardIndex:1},{offset:12},{exitCode:1},{startedUnix:NaN},
    {finishedUnix:1794704500},{durationSeconds:Infinity},{durationSeconds:-1},
    {durationSeconds:0},{runId:'456'},{runAttempt:'1'},{runAttempt:null}])
    assert.throws(()=>verifyShardClock({...x,...patch},expected),/Invalid shard identity/);
});

test('checkpoint history must match the complete final training history prefix',()=>{
  const f=fake();try{
    const finalHistory=[...f.cp.history,{seed:mixSeed(seed,8,0xa975),studentSeat:0,winner:0,score:[4,2]}];
    assert.equal(verifyTrainingCheckpoints(f.dir,{...options,finalHistory}).length,1);
    const altered=structuredClone(finalHistory);
    altered[3].score=[4,3];
    assert.throws(()=>verifyTrainingCheckpoints(f.dir,{...options,finalHistory:altered}),/history diverges/);
    assert.throws(()=>verifyTrainingCheckpoints(f.dir,{...options,finalHistory:[]}),/Invalid final/);
  }finally{rmSync(f.dir,{recursive:true,force:true});}
});
