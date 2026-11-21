import test from 'node:test';
import assert from 'node:assert/strict';
import {legalPureRlSelections} from '../src/ai/pure-rl-phases.js';
import {mapPublicAQSelectionScores}
  from '../scripts/rnad/selection-oracle.mjs';

const legal=legalPureRlSelections({phase:'selection',self:{
  team:Array.from({length:6},(_,i)=>({instanceId:'set-'+i}))}});
const scoreFor=(a)=>100-7*Number(a[0].slice(4));
const ranking=legal.flatMap(a=>[
  {selection:[...a],score:scoreFor(a)},
  {selection:[a[0],a[2],a[1]],score:scoreFor(a)}
]).sort((a,b)=>b.score-a.score);

test('AQ public 120 candidates map to exactly 60 authority lead selections',()=>{
  const out=mapPublicAQSelectionScores(ranking,legal);
  assert.equal(out.scores.length,60);
  assert.deepEqual(out.scores,legal.map(scoreFor));
  assert.deepEqual(out.bestAction,legal[0]);
  assert.equal(out.scoreMax,100);
});
test('oracle rejects missing, repeated, illegal, inconsistent and invalid rows',()=>{
  assert.throws(()=>mapPublicAQSelectionScores(ranking.slice(1),legal));
  const badDuplicate=ranking.map(x=>({...x,selection:[...x.selection]}));
  badDuplicate[0].selection=[...badDuplicate[1].selection];
  assert.throws(()=>mapPublicAQSelectionScores(badDuplicate,legal));
  const badValue=ranking.map(x=>({...x}));
  badValue[0].score=NaN;
  assert.throws(()=>mapPublicAQSelectionScores(badValue,legal));
  const badEquivalent=ranking.map(x=>({...x}));
  const key=JSON.stringify([badEquivalent[0].selection[0],
    badEquivalent[0].selection.slice(1).sort()]);
  const counterpart=badEquivalent.findIndex((x,i)=>i>0&&
    JSON.stringify([x.selection[0],x.selection.slice(1).sort()])===key);
  badEquivalent[counterpart].score+=1;
  assert.throws(()=>mapPublicAQSelectionScores(badEquivalent,legal));
  const wrongLegal=legal.map(a=>[...a]);
  wrongLegal[0]=[wrongLegal[0][1],wrongLegal[0][0],wrongLegal[0][2]];
  assert.throws(()=>mapPublicAQSelectionScores(ranking,wrongLegal));
});
