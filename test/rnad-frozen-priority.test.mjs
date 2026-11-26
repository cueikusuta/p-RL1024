import test from 'node:test';
import assert from 'node:assert/strict';
import {firstThreatGate} from '../scripts/rnad/frozen-priority/g421-threat-gate.mjs';
const ordinary={id:'ordinary',basePower:80,accuracy:100,category:'Physical',priority:0};
const threat={...ordinary,id:'threat'};
const quick={...ordinary,id:'quick',basePower:40,priority:1};
const p={speciesId:'our',hp:25,maxHP:100,stats:{spe:90},types:['Normal'],moves:[{id:'ordinary'},{id:'quick'}]};
const z={speciesId:'bench',hp:100,maxHP:100,stats:{spe:50},types:['Normal'],moves:[{id:'ordinary'}]};
const o={self:{team:[p,z],active:0},opponent:{speciesId:'their',hpPercent:65,types:['Normal'],publicObservation:{usedMoves:['threat']}},field:{sideEffects:[[],[]]}};
const a=[{kind:'move',index:0},{kind:'move',index:1},{kind:'switch',index:1}];
const d={data:{moves:{ordinary,threat,quick},species:{their:{baseStats:{}}}},calculateChampionsStats:()=>({spe:120,hp:100}),
  estimateAttackPercent:(_src,dst,m)=>m===threat?(dst.speciesId==='bench'?90:35):m===quick?30:null};
const call=(fn,x=o,dep=d)=>fn(x,a,[6,5,4],0,dep);
test('lethal-only gate requires visible threat and estimated priority KO',()=>{
 assert.equal(call(firstThreatGate).priority,null);
 const lethal={...o,opponent:{...o.opponent,hpPercent:20}};
 assert.deepEqual(call(firstThreatGate,lethal).priority.action,a[1]);
 assert.equal(call(firstThreatGate,{...lethal,opponent:{...lethal.opponent,publicObservation:{usedMoves:[]}}}),null);
 assert.equal(call(firstThreatGate,{...lethal,field:{...lethal.field,trickRoomTurns:2}}),null);
});
import {createFrozenPriorityProfile} from '../scripts/rnad/frozen-priority/frozen-priority-profile.mjs';
test('frozen mode is off by default; explicit on consumes only accepted recommendation; shadow preserves action',()=>{
 const seen={...o,phase:'battle',requestType:'turn',turn:4,opponent:{...o.opponent,hpPercent:20}};
 const args={observation:seen,legalActions:a,logits:[6,5,4],index:0,publicContext:{publicFrame:{publicObservation:seen.opponent.publicObservation}}};
 assert.equal(createFrozenPriorityProfile({deps:d}).decide(args),0);
 const log=[];const v=createFrozenPriorityProfile({mode:'public-lethal-priority-v1',deps:d,onAccepted:x=>log.push(x)});
 assert.equal(v.decide({...args,publicContext:null}),0);
 assert.equal(v.decide(args),1);v.reject();
 assert.equal(v.decide(args),1);v.accept(a[1]);
 assert.equal(v.decide(args),0);assert.equal(log.length,1);
 const shadow=createFrozenPriorityProfile({mode:'shadow',deps:d,onAccepted:x=>log.push(x)});
 assert.equal(shadow.decide(args),0);shadow.accept(a[0]);assert.equal(log.length,2);
 assert.deepEqual(log[1].proposed,a[1]);
});
