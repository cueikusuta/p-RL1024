// Aggregate V8 CPU self samples for real complete accepted R-NaD Matches.
// Attribution is observational, not proof of an end-to-end speedup.
import {readFileSync} from 'node:fs';
import {basename} from 'node:path';
const filename=process.argv[2];
if(!filename)throw Error('Provide V8 cpuprofile filename');
const profile=JSON.parse(readFileSync(filename,'utf8'));
if(!Array.isArray(profile.nodes)||profile.nodes.length<1)
  throw Error('Missing CPU profile samples');
const groups=new Map();
let total=0;
for(const node of profile.nodes){
  const count=node.hitCount??0;
  if(count<1)continue;
  total+=count;
  const cf=node.callFrame??{};
  const loc=cf.url?.startsWith('file:')?
    basename(new URL(cf.url).pathname):cf.url??'';
  const key=(cf.functionName??'(unknown)')+' @ '+loc+':'+((cf.lineNumber??-1)+1);
  groups.set(key,(groups.get(key)??0)+count);
}
const hottest=[...groups].sort((a,b)=>b[1]-a[1]).slice(0,35)
  .map(([function_name,samples])=>({
    function_name,samples,percent:Math.round(10000*samples/total)/100
  }));
const report={
  schema:'rnad-i5-authoritative-actor-v8-cpu-profile-v1',
  total_samples:total,period_ms:(profile.endTime-profile.startTime)/1000,
  self_samples_top_functions:hottest,
  claim_of_total_generation_speedup:false,
};
process.stdout.write(JSON.stringify(report,null,2)+'\n');
