import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SWITCH_OPPORTUNITY_OBJECTIVE,
  MAX_MATCH_SWITCH_COST_PER_SEAT,
  acceptedVoluntarySwitchOpportunity,
  shapeSwitchOpportunity,
  summarizeAcceptedSwitchOpportunities
} from '../scripts/rnad/switch-opportunity.mjs';

const poke=(speciesId,types,moves,hp=100)=>({
  speciesId,types,moves: moves.map(id=>({id,pp:20,maxPP:20})),
  hp,maxHP:100
});
const ownTeam=[
  poke('electric','Electric'.split(','),['thunderbolt','tackle']),
  poke('normal',['Normal'],['tackle']),
  poke('rock',['Rock'],['tackle'])
];
const action={kind:'switch',index:1};
const base={
  player_seat:0,phase:'battle',round:1,
  accepted_action:action,
  legal_actions:[{kind:'move',index:0},{kind:'switch',index:1},
    {kind:'switch',index:2}],
  observation:{phase:'battle',requestType:'turn',turn:2,
    self:{team:ownTeam,active:0},
    opponent:{types:['Water'],hpPercent:100}}
};
const clone=x=>structuredClone(x);
const event=(actor,phase='battle')=>({
  actor,phase,reward:[0,0]
});

test('opportunity-v1 sees a legal effective attack forgone for a voluntary switch',()=>{
  assert.equal(SWITCH_OPPORTUNITY_OBJECTIVE,'opportunity-v1');
  const x=acceptedVoluntarySwitchOpportunity(base);
  assert.ok(x.cost>0&&x.cost<=.02);
  assert.ok(x.pressure>.3);
  assert.ok(x.target_pressure<x.pressure);
  assert.equal(x.reason,'lost-available-attack');
});

test('mandatory replacement, pivot, KO-danger and no legal attacks carry no penalty',()=>{
  for(const requestType of ['pivot','replacement']){
    const r=clone(base);r.observation.requestType=requestType;
    assert.equal(acceptedVoluntarySwitchOpportunity(r).cost,0);
  }
  const low=clone(base);low.observation.self.team[0].hp=30;
  assert.equal(acceptedVoluntarySwitchOpportunity(low).cost,0);
  const threatened=clone(base);
  threatened.observation.self.team[0].types=['Fire'];
  assert.equal(acceptedVoluntarySwitchOpportunity(threatened).cost,0);
  const noMove=clone(base);noMove.legal_actions=[action];
  assert.equal(acceptedVoluntarySwitchOpportunity(noMove).cost,0);
  const move=clone(base);move.accepted_action={kind:'move',index:0};
  assert.equal(acceptedVoluntarySwitchOpportunity(move).cost,0);
  const unknown=clone(base);unknown.observation.opponent.types=[];
  assert.equal(acceptedVoluntarySwitchOpportunity(unknown).cost,0);
});

test('only trusted accepted timeline can produce zero-sum bounded shaped rewards',()=>{
  const rows=Array.from({length:40},(_,i)=>({
    ...clone(base),player_seat:i%2,round:1
  }));
  const events=rows.map(r=>event(r.player_seat));
  const report=shapeSwitchOpportunity(events,rows);
  assert.equal(report.objective,'opportunity-v1');
  assert.ok(report.cost_per_seat[0]>0&&report.cost_per_seat[1]>0);
  assert.ok(report.cost_per_seat.every(x=>x<=MAX_MATCH_SWITCH_COST_PER_SEAT));
  assert.ok(events.every(x=>Math.abs(x.reward[0]+x.reward[1])<1e-12));
  for(let i=0;i<events.length;i++){
    const row=events[i];
    assert.equal(row.action_kind,'switch');
    assert.equal(row.request_type,'turn');
    assert.ok(row.switch_opportunity_cost>=0);
    assert.ok(row.switch_opportunity_cost<=.02+1e-12);
    assert.equal(row.reward[row.actor],-row.switch_opportunity_cost);
  }
  assert.throws(()=>shapeSwitchOpportunity(events.slice(0,1),rows),/aligned/);
  assert.throws(()=>shapeSwitchOpportunity(events,rows.map((row,i)=>
    i===0?{...row,player_seat:1}:row)),/Mismatched accepted/);
});

test('no shaping on draft, selection, status moves, or non-switch accepted actions',()=>{
  for(const [phase,accepted] of [
    ['draft',['p1','p2','p3','p4','p5','p6']],
    ['selection',['p1','p2','p3']],
    ['battle',{kind:'move',index:0}]
  ]){
    const row={...clone(base),phase,accepted_action:accepted};
    const items=[event(0,phase)];
    const meta=shapeSwitchOpportunity(items,[row]);
    assert.equal(meta.cost_per_seat[0],0);
    assert.deepEqual(items[0].reward,[0,0]);
  }
});

test('reason codes distinguish voluntary skips without changing opportunity cost',()=>{
  const cases=[
    ['not-voluntary-switch',r=>{r.accepted_action={kind:'move',index:0};}],
    ['non-turn-switch',r=>{r.observation.requestType='replacement';}],
    ['missing-known-team',r=>{r.observation.self.team.splice(1,1);}],
    ['opponent-type-unrevealed',r=>{r.observation.opponent.types=[];}],
    ['invalid-own-hp',r=>{r.observation.self.team[0].maxHP=0;}],
    ['own-hp-below-65-percent',r=>{r.observation.self.team[0].hp=30;}],
    ['own-status',r=>{r.observation.self.team[0].status='PAR';}],
    ['public-type-threat',r=>{r.observation.self.team[0].types=['Fire'];}],
    ['no-legal-attack',r=>{r.legal_actions=[action];}],
    ['low-attack-pressure',r=>{r.observation.self.team[0].moves=[
      {id:'tackle',pp:20,maxPP:20}
    ];}]
  ];
  for(const [reason,change] of cases){
    const row=clone(base);change(row);
    const found=acceptedVoluntarySwitchOpportunity(row);
    assert.equal(found.cost,0,reason);
    assert.equal(found.reason,reason);
  }
  assert.equal(acceptedVoluntarySwitchOpportunity(base).reason,
    'lost-available-attack');
});

test('read-only accepted action diagnostic accounts for both seats and cap',()=>{
  const rows=Array.from({length:80},(_,i)=>({
    ...clone(base),player_seat:i%2
  }));
  const before=clone(rows);
  const result=summarizeAcceptedSwitchOpportunities(rows);
  assert.deepEqual(rows,before);
  assert.equal(result.schema,'rnad-switch-opportunity-reason-diagnostic-v1');
  assert.equal(result.voluntary_switches,80);
  assert.equal(result.shaped_switches,
    result.by_seat.reduce((n,s)=>n+s.shaped_switches,0));
  for(const seat of result.by_seat){
    assert.equal(seat.voluntary_switches,40);
    assert.ok(seat.shaped_switches>0&&seat.shaped_switches<40);
    assert.ok(seat.reason_counts['match-budget-exhausted']>0);
    assert.equal(Object.values(seat.reason_counts)
      .reduce((n,x)=>n+x,0),seat.voluntary_switches);
    assert.ok(seat.cost<=MAX_MATCH_SWITCH_COST_PER_SEAT+1e-12);
  }
  assert.throws(()=>summarizeAcceptedSwitchOpportunities([]),/trusted/);
  assert.throws(()=>summarizeAcceptedSwitchOpportunities([
    {...clone(base),player_seat:3}
  ]),/trusted/);
});

test('diagnostic excludes mandatory switches and retains zero-sum seat accounting',()=>{
  const rows=[clone(base),clone(base),clone(base),clone(base)];
  rows[0].player_seat=0;
  rows[1].player_seat=0;rows[1].observation.requestType='pivot';
  rows[2].player_seat=1;rows[2].observation.opponent.types=[];
  rows[3].player_seat=1;rows[3].observation.self.team[0].hp=20;
  const result=summarizeAcceptedSwitchOpportunities(rows);
  assert.equal(result.voluntary_switches,3);
  assert.equal(result.shaped_switches,1);
  assert.deepEqual(result.by_seat[0].reason_counts,
    {'lost-available-attack':1});
  assert.deepEqual(result.by_seat[1].reason_counts,
    {'opponent-type-unrevealed':1,'own-hp-below-65-percent':1});
  assert.equal(result.cost_per_seat[1],0);
  assert.ok(result.cost_per_seat[0]>0);
});
