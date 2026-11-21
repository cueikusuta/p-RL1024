// Conservative study-seed ledger verification. Only establishes non-overlap
// within the explicitly registered I5 ranges, NOT all historical repository runs.
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';

export function verifyI5StudySeedLedger(ledger,confirmation){
  if(ledger?.schema!=='rnad-i5-study-seed-ledger-v1'||
     ledger.selected_screen_run!==43962423945||
     !Array.isArray(ledger.entries)||ledger.entries.length<10||
     confirmation?.schema!=='rnad-i5-selection-blend-independent-confirm-v1'||
     confirmation.version!==1||confirmation.selected_screen_run!==43962423945||
     confirmation.selected_mode!=='selection-threequarter-v4'||
     confirmation.v4_source_run!==43957735875)
    throw new TypeError('Invalid versioned source-bound I5 seed audit');
  const ids=new Set(),ranges=[];
  for(const {id,start,count} of ledger.entries){
    if(typeof id!=='string'||!id||ids.has(id)||
       !Number.isSafeInteger(start)||start<1||
       !Number.isSafeInteger(count)||count<1||count>2048||
       start+count-1>0xffffffff)
      throw new Error('Malformed registered I5 research seed block');
    ids.add(id);ranges.push({id,start,end:start+count-1});
  }
  ranges.sort((a,b)=>a.start-b.start||a.end-b.end);
  for(let i=1;i<ranges.length;i++){
    if(ranges[i].start<=ranges[i-1].end)
      throw new Error('Overlapping known I5 seed blocks');
  }
  const selected=ranges.find(x=>x.id==='selected-blend-confirm');
  if(!selected||selected.start!==confirmation.seed||
     selected.end!==confirmation.seed+confirmation.pairs-1||
     !Number.isSafeInteger(confirmation.pairs)||
     confirmation.pairs<32||confirmation.pairs>100)
    throw new Error('Unregistered or modified preselected confirmation seed range');
  // This is a bounded provenance check, not a strength claim or a proof
  // about unregistered older historical experiments.
  return {schema:'rnad-i5-registered-seed-disjointness-audit-v1',
    selected_seed_start:selected.start,
    selected_seed_end:selected.end,
    selected_screen_run:ledger.selected_screen_run,
    registered_blocks:ranges.length,
    disjoint_within_registered_i5_blocks:true,
    exhaustive_historical_repository_audit:false,
    statistically_beats_aq:false,
    champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const [l,m,...extra]=process.argv.slice(2);
  if(!l||!m||extra.length)throw new Error('Usage: LEDGER.json CONFIRM_MANIFEST.json');
  const a=JSON.parse(readFileSync(l,'utf8'));
  const b=JSON.parse(readFileSync(m,'utf8'));
  console.log(JSON.stringify(verifyI5StudySeedLedger(a,b)));
}
