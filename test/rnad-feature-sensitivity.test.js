import test from 'node:test';
import assert from 'node:assert/strict';
import {projectRnadObservation}
  from '../src/ml/legal-action-adapter.js';
import {encodeRnadThreePhaseDecisionV3}
  from '../src/ml/rnad-phase-features-v3.js';

// Preserve the distinction between an actor's own move priority, an
// opponent's REVEALED move, and an opponent's still-hidden moves.
// This tests current v3 information flow; it does not assert that v3 has a
// calibrated probability of an unseen priority move. That requires a new,
// separately versioned semantic feature and an actual learning comparison.
function battleView() {
  return {
    phase:'turn',requestType:'turn',turn:3,
    self:{active:0,canSwitch:true,canMega:false,needsSwitch:false,
      team:[
        {instanceId:'mine0',speciesId:'pikachu',types:['Electric'],
          moves:['quickattack','tackle'],
          stats:{atk:80,def:70,spa:80,spd:70,spe:100},
          hp:100,maxHP:100},
        {instanceId:'mine1',speciesId:'eevee',types:['Normal'],
          moves:['tackle'],hp:100,maxHP:100}
      ]},
    opponent:{speciesId:'dragonite',types:['Dragon','Flying'],
      hpPercent:100,remaining:3,status:null,
      publicObservation:{usedMoves:['tackle']},
      privateMoves:['earthquake'],hiddenItemId:'secret'},
    field:{weather:null,terrain:null,trickRoomTurns:0,sideEffects:[[],[]]}
  };
}
const legal=[{kind:'move',index:0},{kind:'move',index:1},
  {kind:'switch',index:1}];
const encoded=view=>encodeRnadThreePhaseDecisionV3(
  projectRnadObservation(view,{seat:0}),legal);

test('own known priority is represented by the inherited action feature',()=>{
  const out=encoded(battleView());
  // v3 preserves v1/v2 action coordinates, where index 12 is own move priority.
  assert.equal(out.actions.length,3);
  assert.ok(out.actions[0][12]>out.actions[1][12],
    'priority +1 move must differ from priority 0 move');
});

test('publicly revealed opponent move identity reaches actor features',()=>{
  const before=encoded(battleView());
  const changed=battleView();
  changed.opponent.publicObservation.usedMoves=['quickattack'];
  const after=encoded(changed);
  // v1 state[29] hashes the list of revealed moves. This is identity
  // sensitivity, NOT an explicit semantic estimate of priority risk.
  assert.notEqual(before.state[29],after.state[29]);
});

test('changing only opponent hidden moves and item never changes features',()=>{
  const before=battleView(),after=battleView();
  after.opponent.privateMoves=['quickattack','extremespeed'];
  after.opponent.hiddenItemId='different-secret';
  after.opponent.privateSet={moves:['extremespeed'],item:'private'};
  assert.deepEqual(encoded(before),encoded(after));
});
