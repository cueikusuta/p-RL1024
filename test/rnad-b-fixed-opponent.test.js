import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {approvedFixedSession} from '../scripts/rnad/shared_league_actor.mjs';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha}
  from '../scripts/dev/simulator/rnad-match.js';
import {weightsSha256} from '../models/aq-base.mjs';
import provenance from '../provenance/manifest.json' with {type:'json'};

const hash=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
const coreSha=hash(new URL('../src/ai/draft-core-trio.js',import.meta.url));
const config=kind=>({
  schema:'rnad-i5-authorized-fixed-opponent-v1',
  kind,aq_model_sha256:weightsSha256,core_trio_source_sha256:coreSha,
  epsilon:.02,source_git_sha:verifiedRunningSourceSha(),
  research_only:true
});
test('fixed AQ/Core-Trio sessions reject unknown model or source identity',()=>{
  for(const mode of ['aqbase001','coretrioaq001']){
    const a=approvedFixedSession(config(mode));
    assert.equal(a.func.requiresSeatPublicView,true);
    assert.equal(a.saved.length,0);
    assert.equal(a.pending,null);
  }
  assert.throws(()=>approvedFixedSession({...config('aqbase001'),
    aq_model_sha256:'0'.repeat(64)}),/Unapproved/);
  assert.throws(()=>approvedFixedSession({...config('aqbase001'),
    core_trio_source_sha256:'0'.repeat(64)}),/Unapproved/);
  assert.throws(()=>approvedFixedSession(config('invented')),/Unapproved/);
  assert.throws(()=>approvedFixedSession({...config('aqbase001'),
    source_git_sha:'0'.repeat(40)}),/Unapproved/);
});
test('real authority view accepts exact AQ and Core-Trio draft proposals',()=>{
  // A one-round compatibility smoke only; not a victory-rate estimate.
  const aq=approvedFixedSession(config('aqbase001'));
  const core=approvedFixedSession(config('coretrioaq001'));
  const result=runRnadAcceptanceSmoke({
    seed:20311021,matchId:'i5-b-authority-smoke',
    sourceSha:verifiedRunningSourceSha(),
    engineSnapshotSha:provenance.sourceCommit,
    phasePolicies:[aq.func,core.func],maxRounds:1,
    includeAcceptedHistory:true,
  });
  assert.ok(result.rounds===1);
  for(const p of [aq,core]){
    assert.ok(p.saved.some(x=>x.phase==='draft'));
    assert.ok(p.saved.some(x=>x.phase==='selection'));
    assert.equal(p.pending,null);
    for(const event of p.saved){
      assert.equal(event.policy_pi.length,event.magnet_pi.length);
      assert.ok(event.behavior_prob>0&&event.behavior_prob<=1);
      assert.ok(Math.abs(event.policy_pi.reduce((a,b)=>a+b,0)-1)<1e-10);
      assert.ok(event.policy_pi.every(x=>x>0));
    }
  }
});
