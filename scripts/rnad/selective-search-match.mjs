import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {createIntegratedAQSession} from './integrated-selective-session.mjs';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-selective-match.js';
import {createOpponent,weightsSha256} from '../../models/aq-base.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import {pairedClusterCI} from './paired-league-stats.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};

import {createSelectiveReranker} from './selective-belief-search.mjs';

const sha=b=>createHash('sha256').update(b).digest('hex');
const read=f=>JSON.parse(readFileSync(f,'utf8'));
const put=(f,data)=>writeFileSync(f,JSON.stringify(data)+'\n',{flag:'wx'});
const [op,planFile,root,arg,out]=process.argv.slice(2);
const p=read(planFile);
const sha40=/^[0-9a-f]{40}$/, sha64=/^[0-9a-f]{64}$/;
assert.equal(p.schema,'rnad-i5-selective-search-v1');
assert.equal(p.engine,provenance.sourceCommit);
assert.equal(p.source,verifiedRunningSourceSha());
assert.equal(p.aq,weightsSha256);
assert.equal(p.seedStart,430330001);
assert.equal(p.smokeSeed,430339001);
assert.equal(p.blocks,1024);
assert.equal(p.parts,16);
assert.equal(p.alpha,.05);
assert.deepEqual(Object.keys(p.models),['G421']);
assert.deepEqual(p.matchups,[['G421','AQ'],['Prior','AQ'],['History','AQ'],
  ['G421','Strategy'],['History','Strategy'],['History','G421']]);
assert.deepEqual(p.swaps,{});
assert.equal(p.model.worlds,4);
assert.equal(p.model.minimumRealWorlds,2);
assert.equal(p.model.maxLogitGap,.75);
assert.equal(p.model.minimumMeanUtilityGain,.08);
assert.equal(p.model.maxWorstWorldLoss,.05);
const external={AQ:'aq-base',Strategy:'strategy'};
const neural=id=>!Object.hasOwn(external,id);
const modelIds=[...Object.keys(p.models),...Object.keys(p.swaps)];
for(const [id,x] of Object.entries(p.models)){
  assert.equal(x.generation,Number(id.slice(1))+1);
  assert(sha64.test(x.zip)&&sha64.test(x.actor)&&sha40.test(x.source));
}
const planSha=sha(readFileSync(planFile));
const studyHead=process.env.GITHUB_SHA??null;
assert(studyHead===null||sha40.test(studyHead));
const models=join(root,'models');
const part=Number(arg);
function evalPart(start,count){
  const aq=createOpponent();
  assert.equal(aq.modelSha256,p.aq);
  const manifestSha={};
  for(const id of modelIds){
    const dir=join(models,id);
    const raw=readFileSync(join(dir,'manifest.json'));
    const m=JSON.parse(raw);
    assert.equal(m.source_sha,p.source);
    assert.equal(m.engine_sha,p.engine);
    assert.equal(m.generation,p.models[id].generation);
    assert.equal(m.schema,'rnad-integrated-policy-actor-bundle-v1');
    assert.deepEqual(Object.keys(m.files).sort(),['battle','draft','selection','values']);
    for(const [name,digest] of Object.entries(m.files)) assert.equal(sha(readFileSync(join(dir,name+'.json'))),digest);
    assert.equal(read(join(dir,'battle.json')).feature_schema,'rnad-three-phase-features-v6');
    assert.equal(sha(raw),p.models[id].actor);
    manifestSha[id]=sha(raw);
  }
  const rows=[];
  for(const [left,right] of p.matchups) for(let seed=start;seed<start+count;seed++) for(const seat of [0,1]){
    const ids=seat===0?[left,right]:[right,left];
    const trackers=ids.map(id=>['Prior','History'].includes(id)?createSelectiveReranker({mode:id,model:p.model}):null);
    const sessions=ids.map((id,s)=>!neural(id)?null:createIntegratedAQSession({
      modelDir:join(models,'G421'),seat:s,researchRerank:trackers[s]?.rerank??null}));
    const result=runStrategyMatch({seed,policies:ids.map(id=>neural(id)?'pure-rl-linear':external[id]),
      pureRlSessions:sessions,pureRlAllPhases:true,strategyMode:'integrated',
      ...(right==='AQ'?{v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq}:{})});
    const {winner,score,rounds}=result;
    assert(result.completed===true && [0,1].includes(winner) && Array.isArray(score) && score.length===2 &&
      score[winner]===4 && Number.isSafeInteger(score[1-winner]) && score[1-winner]>=0 && score[1-winner]<4 &&
      Number.isSafeInteger(rounds) && rounds>=4 && rounds<=64,`incomplete ${left}/${right} ${seed} seat ${seat}`);
    const accepted=ids.map((id,s)=>{
      if(!neural(id))return null;
      const st=result.pureRlStats?.[s], count=result.actionCounts?.[s]+2*rounds;
      assert(st?.updates===0 && st.pending===false && Number.isSafeInteger(count) && count>2*rounds &&
        st.accepted_decisions===count,`bad accepted decisions ${id} ${seed} seat ${s}`);
      return count;
    });
    rows.push({left,right,seed,seat,winner,score,rounds,accepted,search:trackers.map(x=>x?.stats??null)});
  }
  return {planSha,studyHead,manifestSha,part:op==='smoke'?null:part,start,count,noTraining:true,rows};
}
function summarize(){
  const groups=new Map(p.matchups.map(([l,r])=>[l+'/'+r,Array.from({length:p.blocks},()=>[null,null])]));
  const count=p.blocks/p.parts;
  let pinned;
  const keys=['turns','eligible','twoWorlds','publicReveals','changed','worlds'];
  const search={Prior:{},History:{}};
  for(const dest of Object.values(search)){
    for(const key of keys)dest[key]=0;
    dest.histogram=Array(10).fill(0);
  }
  for(let i=0;i<p.parts;i++){
    const x=read(join(root,'i5-selective-part-'+i,'shard.json'));
    assert.equal(x.planSha,planSha);
    assert.equal(x.studyHead??null,studyHead);
    assert.equal(x.noTraining,true);
    assert.equal(x.part,i);
    assert.equal(x.start,p.seedStart+i*count);
    assert.equal(x.count,count);
    assert.equal(x.rows.length,count*2*p.matchups.length);
    assert.equal(Object.values(x.manifestSha).filter(s=>sha64.test(s)).length,modelIds.length);
    assert.deepEqual(Object.keys(x.manifestSha).sort(),modelIds.slice().sort());
    if(pinned)assert.deepEqual(pinned,x.manifestSha);else pinned=x.manifestSha;
    for(const r of x.rows){
      const g=groups.get(r.left+'/'+r.right), k=r.seed-p.seedStart;
      assert(Number.isSafeInteger(r.seed) && r.seed>=x.start && r.seed<x.start+count &&
        g && [0,1].includes(r.seat) && g[k][r.seat]===null && [0,1].includes(r.winner) &&
        Array.isArray(r.score) && r.score.length===2 && r.score[r.winner]===4 &&
        Number.isSafeInteger(r.score[1-r.winner]) && r.score[1-r.winner]>=0 && r.score[1-r.winner]<4 &&
        Number.isSafeInteger(r.rounds) && r.rounds>=4 && r.rounds<=64);
      assert(Array.isArray(r.accepted) && r.accepted.length===2);
      for(const [s,id] of (r.seat===0?[r.left,r.right]:[r.right,r.left]).entries())
        assert(!neural(id)?r.accepted[s]===null:Number.isSafeInteger(r.accepted[s])&&r.accepted[s]>2*r.rounds);
      assert(Array.isArray(r.search)&&r.search.length===2);
      const ids=r.seat===0?[r.left,r.right]:[r.right,r.left];
      for(const [seat,id] of ids.entries()){
        const st=r.search[seat];
        if(!Object.hasOwn(search,id)){assert.equal(st,null);continue;}
        for(const key of keys){
          assert(Number.isSafeInteger(st[key])&&st[key]>=0);
          search[id][key]+=st[key];
        }
        assert(st.turns<=r.accepted[seat]&&st.changed<=st.twoWorlds&&st.twoWorlds<=st.eligible&&
          st.publicReveals<=st.eligible&&st.worlds>=2*st.twoWorlds&&st.worlds<=4*st.twoWorlds&&
          st.histogram?.length===10&&st.histogram.every(n=>Number.isSafeInteger(n)&&n>=0)&&
          st.histogram.reduce((a,b)=>a+b,0)===st.eligible);
        st.histogram.forEach((n,i)=>{search[id].histogram[i]+=n;});
      }
      g[k][r.seat]=Number(r.winner===r.seat);
    }
  }
  const block=key=>groups.get(key).map(a=>{assert(a.every(v=>v!==null));return (a[0]+a[1])/2;});
  const rates=Object.fromEntries([...groups.keys()].map(k=>{
    const b=block(k);
    return [k,{wins:b.reduce((a,v)=>a+2*v,0),matches:2*p.blocks,
      twoWins:b.filter(v=>v===1).length,splits:b.filter(v=>v===.5).length,twoLosses:b.filter(v=>v===0).length,
      ...pairedClusterCI(b,{familywiseAlpha:p.alpha,comparisons:k==='History/G421'?3:6,range:[0,1],nullValue:.5})}];
  }));
  const contrast=(a,b,comparisons)=>{
    const x=block(a), y=block(b);
    const diffs=x.map((v,i)=>v-y[i]);
    return {higher:diffs.filter(x=>x>0).length,same:diffs.filter(x=>x===0).length,lower:diffs.filter(x=>x<0).length,
      ...pairedClusterCI(diffs,{familywiseAlpha:p.alpha,comparisons})};
  };
  const changes={
    historyMinusG421AQ:contrast('History/AQ','G421/AQ',1),
    historyMinusPriorAQ:contrast('History/AQ','Prior/AQ',3),
    historyMinusG421Strategy:contrast('History/Strategy','G421/Strategy',3),
  };
  const decision=changes.historyMinusG421AQ.confidence_interval[0]>0 &&
    changes.historyMinusG421Strategy.confidence_interval[0]>-.02 &&
    rates['History/G421'].confidence_interval[0]>.48 &&
    search.History.changed>0?'consider confirmation':'hold';
  return {schema:'rnad-i5-selective-search-result-v1',planSha,studyHead,pinned,source:p.source,engine:p.engine,
    aqSHA:p.aq,seedStart:p.seedStart,independentSeedBlocks:p.blocks,
    completedMatches:p.blocks*2*groups.size,rates,changes,search,
    latencyHistogramUpperMs:[.05,.1,.2,.5,1,2,5,10,25,'overflow'],decision,
    researchOnly:true,championPromoted:false};
}
if(op==='smoke'){
  const data=evalPart(p.smokeSeed,2);
  assert.equal(data.rows.length,24);
  const counters={};
  for(const arm of ['Prior','History']){
    const selected=data.rows.flatMap(r=>r.search.map((st,seat)=>({st,
      id:(r.seat===0?[r.left,r.right]:[r.right,r.left])[seat]}))).filter(x=>x.id===arm);
    counters[arm]=Object.fromEntries(['turns','eligible','twoWorlds','publicReveals','changed']
      .map(k=>[k,selected.reduce((n,x)=>n+x.st[k],0)]));
  }
  console.log('24 verified, excluded smoke matches '+JSON.stringify(counters));
}else if(op==='run'){
  assert(Number.isInteger(part)&&part>=0&&part<p.parts);
  const n=p.blocks/p.parts;
  put(out,evalPart(p.seedStart+part*n,n));
  console.log('completed shard '+part);
}else if(op==='summary'){
  const report=summarize();
  put(out,report);
  console.log(JSON.stringify(report));
}else throw Error('expected smoke, run or summary');
