import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {validateHostedWitness} from '../scripts/merge-hosted-witnesses.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
const digest='a'.repeat(64),modelSha='b'.repeat(64);
const planned={offset:0,count:15,seedStart:975020};
const ctx={index:0,planned,runId:'123456',runAttempt:'2',
  commit:'a'.repeat(40)};
function make(){
  const body={
    schema:'p-rl1024-hosted-shard-witness-v1',
    runId:ctx.runId,runAttempt:ctx.runAttempt,commit:ctx.commit,
    sourceCommit:ctx.commit,index:0,model:'RL-Deep',opponent:'AQ-Base-001',
    seedRoot:975020,shards:18,totalMatches:256,
    sourceFingerprint:digest,
    row:{artifact:'shard-0',offset:0,count:15,seedStart:975020,seedEnd:975034,
      studentWins:1,startedUnix:1794704000,finishedUnix:1794704004,
      durationSeconds:3.5,modelSha256:modelSha,modelFileSha256:digest,
      normalizedEvaluationDigestSha256:digest,
      checkpoints:[{name:'training-m8.json',completedMatches:8,
        fileSha256:digest,stateSha256:digest}]},
    fileDigests:{trainingSha256:digest,comparisonSha256:digest,
      evaluationASha256:digest,evaluationBSha256:digest,clockSha256:digest}
  };
  return {...body,witnessSha256:sha(JSON.stringify(body))};
}
function resign(x){const {witnessSha256,...body}=x;return {...body,witnessSha256:sha(JSON.stringify(body))};}
test('valid witness contains all checkpoints and SHA and matches 1 planned shard',()=>{
  assert.equal(validateHostedWitness(make(),ctx).count,15);
});
test('witness tampering and cross-attempt contamination fail closed',()=>{
  const w=make();
  assert.throws(()=>validateHostedWitness({...w,index:1},ctx),/SHA/);
  assert.throws(()=>validateHostedWitness(resign({...w,runAttempt:'1'}),ctx),/provenance/);
  assert.throws(()=>validateHostedWitness(resign({...w,row:{...w.row,checkpoints:[]}}),ctx),/checkpoint/);
  assert.throws(()=>validateHostedWitness(resign({...w,row:{...w.row,seedEnd:975035}}),ctx),/range/);
  assert.throws(()=>validateHostedWitness(resign({...w,fileDigests:{...w.fileDigests,trainingSha256:modelSha}}),ctx),/SHA evidence/);
});
