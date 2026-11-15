import test from 'node:test';
import assert from 'node:assert/strict';
import {runPhaseMatch,consumePhaseAuthorityReceipt}
  from '../scripts/dev/simulator/phase-match.js';
import {createBuiltinPhaseAgents}
  from '../scripts/dev/simulator/phase-agents.js';
import {rnadLegalActions,projectRnadObservation}
  from '../src/ml/legal-action-adapter.js';
import {stage0FixedPolicy}
  from '../scripts/dev/simulator/rnad-match.js';

test('authority receipts are one-use, unforgeable and issued only post-Match acceptance',()=>{
  const counts=[0,0];
  const agents=[0,1].map(seat=>createBuiltinPhaseAgents({
    draft:'basic',selection:'basic',battle:{
      id:'rnad-receipt-contract',
      select({view}){
        const proposed=stage0FixedPolicy({
          observation:projectRnadObservation(view,{seat}),
          legalActions:rnadLegalActions(view)});
        assert.throws(()=>consumePhaseAuthorityReceipt(Object.freeze({}),{
          kind:'battle',seat,round:1,action:proposed.action
        }),/Missing authoritative Match acceptance receipt/);
        return proposed.action;
      },
      accepted({action,authorityReceipt,round}){
        assert.ok(authorityReceipt&&typeof authorityReceipt==='object');
        assert.throws(()=>consumePhaseAuthorityReceipt(
          structuredClone(authorityReceipt),{kind:'battle',seat,round,action}),
        /Missing authoritative Match acceptance receipt/);
        assert.throws(()=>consumePhaseAuthorityReceipt(authorityReceipt,{
          kind:'battle',seat:1-seat,round,action
        }),/Missing authoritative Match acceptance receipt/);
        assert.equal(consumePhaseAuthorityReceipt(authorityReceipt,{
          kind:'battle',seat,round,action}),true);
        assert.throws(()=>consumePhaseAuthorityReceipt(authorityReceipt,{
          kind:'battle',seat,round,action
        }),/Missing authoritative Match acceptance receipt/);
        counts[seat]++;
      }
    }
  }));
  const result=runPhaseMatch({seed:22005,agents,maxRounds:1});
  assert.equal(result.completed,false);
  assert.ok(counts.every(n=>n>0));
  assert.deepEqual(counts,result.actionCounts);
});
