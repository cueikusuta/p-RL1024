import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {verifyI5StudySeedLedger}
  from '../scripts/rnad/verify-i5-study-seeds.mjs';
const ledger=JSON.parse(readFileSync(
  new URL('../docs/research/rnad/I5_STUDY_SEED_LEDGER_2026-11-17.json',import.meta.url)));
const manifest=JSON.parse(readFileSync(
  new URL('../docs/research/rnad/SELECTION_BLEND_CONFIRM_MANIFEST.json',import.meta.url)));

test('known I5 model search and independent selected seed blocks are disjoint',()=>{
  const result=verifyI5StudySeedLedger(ledger,manifest);
  assert.equal(result.selected_seed_start,2090001);
  assert.equal(result.selected_seed_end,2090064);
  assert.equal(result.disjoint_within_registered_i5_blocks,true);
  assert.equal(result.exhaustive_historical_repository_audit,false);
  assert.equal(result.champion_promoted,false);
});
test('seed ledger fails closed on overlaps, duplicate IDs and modified confirmation',()=>{
  const copy=()=>structuredClone(ledger);
  const overlap=copy();
  overlap.entries[1].start=overlap.entries[0].start;
  assert.throws(()=>verifyI5StudySeedLedger(overlap,manifest));
  const duplicate=copy();
  duplicate.entries[1].id=duplicate.entries[0].id;
  assert.throws(()=>verifyI5StudySeedLedger(duplicate,manifest));
  assert.throws(()=>verifyI5StudySeedLedger(copy(),{...manifest,seed:2090064}));
  assert.throws(()=>verifyI5StudySeedLedger(copy(),{...manifest,selected_mode:'all-v4'}));
});
