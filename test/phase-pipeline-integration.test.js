import test from 'node:test';
import assert from 'node:assert/strict';
import {runPhaseMatch} from '../scripts/dev/simulator/phase-match.js';
import {createBuiltinPhaseAgents,sessionPhaseAgent,BUILTIN_PHASE_AGENTS}
  from '../scripts/dev/simulator/phase-agents.js';
import {createModel,createSession} from '../models/deep.mjs';

test('phase registry exposes bounded independent choices',()=>{
  assert.deepEqual(BUILTIN_PHASE_AGENTS.draft,['basic','strategy']);
  assert.ok(BUILTIN_PHASE_AGENTS.selection.includes('tactical'));
  assert.ok(BUILTIN_PHASE_AGENTS.battle.includes('intermediate'));
  assert.throws(()=>createBuiltinPhaseAgents({draft:'unknown'}),/Unsupported/);
});
test('mixed AIs finish authoritative four-win Match deterministically',()=>{
  const sample=()=>{
    const left=createBuiltinPhaseAgents({draft:'strategy',selection:'tactical',battle:'basic'});
    const right=createBuiltinPhaseAgents({draft:'basic',selection:'strategy',battle:'intermediate'});
    const result=runPhaseMatch({seed:22002,agents:[left,right],maxRounds:64});
    assert.equal(result.completed,true);
    assert.equal(result.score[result.winner],4);
    assert.deepEqual(result.agentIds,[{draft:'strategy',selection:'tactical',battle:'basic'},
      {draft:'basic',selection:'strategy',battle:'intermediate'}]);
    return {score:result.score,winner:result.winner,rounds:result.rounds,
      actionCounts:result.actionCounts,roundWinners:result.roundWinners};
  };
  assert.deepEqual(sample(),sample());
});
test('frozen RL-Deep session can independently replace only battle phase',()=>{
  const session=createSession(createModel(22002),{train:false});
  const left=createBuiltinPhaseAgents({draft:'basic',selection:'tactical',
    battle:sessionPhaseAgent('rl-deep',session)});
  const result=runPhaseMatch({seed:22003,maxRounds:1,
    agents:[left,createBuiltinPhaseAgents({draft:'strategy',selection:'basic',battle:'basic'})]});
  assert.equal(result.rounds,1);
  assert.equal(result.agentIds[0].battle,'rl-deep');
  assert.equal(result.agentStats[0].battle.decisions,result.actionCounts[0]);
  assert.equal(result.agentStats[0].battle.updates,0);
});
test('custom distinct agents see earlier accepted decisions and own seat only',()=>{
  const base=createBuiltinPhaseAgents({draft:'basic',selection:'basic',battle:'basic'});
  const observed={selection:false,battle:false};
  const d=base.draft,s=base.selection,b=base.battle;
  base.draft={id:'draft-custom',select:args=>({action:d.select(args),handoff:{draftPlan:'save-lead'}})};
  base.selection={id:'selection-custom',select:args=>{
    assert.equal(args.context.handoff.draft.draftPlan,'save-lead');
    assert.equal(args.context.current.draft.roster.length,6);
    assert.equal(args.context.seat,0);
    observed.selection=true;
    return {action:s.select(args),handoff:{selectionPlan:'aggressive'}};
  }};
  base.battle={id:'battle-custom',select:args=>{
    assert.equal(args.context.handoff.selection.selectionPlan,'aggressive');
    assert.equal(args.context.current.selection.ids.length,3);
    assert.equal(args.context.seat,0);
    observed.battle=true;
    return b.select(args);
  }};
  const result=runPhaseMatch({seed:22004,maxRounds:1,
    agents:[base,createBuiltinPhaseAgents()]});
  assert.equal(result.rounds,1);
  assert.equal(observed.selection,true);
  assert.equal(observed.battle,true);
});
