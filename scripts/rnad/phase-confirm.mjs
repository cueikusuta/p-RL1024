import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {createIntegratedAQSession} from './evaluate-shared-g012-vs-aq.mjs';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256} from '../../models/aq-base.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import {pairedClusterCI} from './paired-league-stats.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const sha=b=>createHash('sha256').update(b).digest('hex');
const read=f=>JSON.parse(readFileSync(f,'utf8'));
const put=(f,data)=>writeFileSync(f,JSON.stringify(data)+'\n',{flag:'wx'});
const [op,planFile,root,arg,out]=process.argv.slice(2);
const p=read(planFile);
const sha40=/^[0-9a-f]{40}$/, sha64=/^[0-9a-f]{64}$/;
assert.equal(p.schema,'rnad-i5-paired-phase-2026-11-v1');
assert.equal(p.engine,provenance.sourceCommit);
assert.equal(p.source,verifiedRunningSourceSha());
assert.equal(p.aq,weightsSha256);
assert.equal(p.seedStart,410260001);
assert.equal(p.smokeSeed,410269001);
assert.equal(p.blocks,2048);
assert.equal(p.parts,32);
assert.equal(p.alpha,.05);
assert.deepEqual(Object.keys(p.models),['G198','G398','G421']);
assert.deepEqual(p.matchups,[['G398','AQ'],['G421','AQ'],['G198','AQ'],['G398','Strategy'],['G421','Strategy'],['G421','G398'],['SwapDraft','AQ'],['SwapSelection','AQ'],['SwapBattle','AQ']]);
assert.deepEqual(p.swaps,{SwapDraft:'draft',SwapSelection:'selection',SwapBattle:'battle'});
assert.equal(p.holdout.runtimePolicy,'strategy');
const external={AQ:'aq-base',Strategy:'strategy'};
const neural=id=>!Object.hasOwn(external,id);
const modelIds=[...Object.keys(p.models),...Object.keys(p.swaps)];
for(const [id,x] of Object.entries(p.models)){
  assert.equal(x.generation,Number(id.slice(1))+1);
  assert(sha64.test(x.zip)&&sha64.test(x.actor)&&sha40.test(x.source));
}
const planSha=sha(readFileSync(planFile));
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
    assert.equal(m.generation,p.models[id]?.generation??p.models.G198.generation);
    assert.equal(m.schema,'rnad-integrated-policy-actor-bundle-v1');
    assert.deepEqual(Object.keys(m.files).sort(),['battle','draft','selection','values']);
    for(const [name,digest] of Object.entries(m.files)) assert.equal(sha(readFileSync(join(dir,name+'.json'))),digest);
    assert.equal(read(join(dir,'battle.json')).feature_schema,'rnad-three-phase-features-v6');
    if(Object.hasOwn(p.swaps,id)){
      const b=read(join(models,'G198','manifest.json')), d=read(join(models,'G421','manifest.json'));
      const phase=p.swaps[id];
      for(const name of ['draft','selection','battle']) assert.equal(m.files[name],(name===phase?d:b).files[name]);
      const expected={...read(join(models,'G198','values.json')).tensors};
      if(phase!=='battle'){
        const next=read(join(models,'G421','values.json')).tensors;
        for(const key of Object.keys(expected)) if(key.startsWith(phase+'_value.')) expected[key]=next[key];
      }
      assert.deepEqual(read(join(dir,'values.json')).tensors,expected);
    }
    manifestSha[id]=sha(raw);
  }
  const rows=[];
  for(const [left,right] of p.matchups) for(let seed=start;seed<start+count;seed++) for(const seat of [0,1]){
    const ids=seat===0?[left,right]:[right,left];
    const sessions=ids.map((id,s)=>!neural(id)?null:createIntegratedAQSession({modelDir:join(models,id),seat:s}));
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
    rows.push({left,right,seed,seat,winner,score,rounds,accepted});
  }
  return {planSha,manifestSha,part:op==='smoke'?null:part,start,count,noTraining:true,rows};
}
function summarize(){
  const groups=new Map(p.matchups.map(([l,r])=>[l+'/'+r,Array.from({length:p.blocks},()=>[null,null])]));
  const count=p.blocks/p.parts;
  let pinned;
  for(let i=0;i<p.parts;i++){
    const x=read(join(root,'i5-confirm-part-'+i,'shard.json'));
    assert.equal(x.planSha,planSha);
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
      g[k][r.seat]=Number(r.winner===r.seat);
    }
  }
  const block=key=>groups.get(key).map(a=>{assert(a.every(v=>v!==null));return (a[0]+a[1])/2;});
  const rates=Object.fromEntries([...groups.keys()].map(k=>{
    const b=block(k);
    return [k,{wins:b.reduce((a,v)=>a+2*v,0),matches:2*p.blocks,
      twoWins:b.filter(v=>v===1).length,splits:b.filter(v=>v===.5).length,twoLosses:b.filter(v=>v===0).length,
      ...pairedClusterCI(b,{familywiseAlpha:p.alpha,comparisons:k==='G398/AQ'?1:11,range:[0,1],nullValue:.5})}];
  }));
  const contrast=(a,b,comparisons)=>{
    const x=block(a), y=block(b);
    const diffs=x.map((v,i)=>v-y[i]);
    return {higher:diffs.filter(x=>x>0).length,same:diffs.filter(x=>x===0).length,lower:diffs.filter(x=>x<0).length,
      ...pairedClusterCI(diffs,{familywiseAlpha:p.alpha,comparisons})};
  };
  const changes={
    G398MinusG421AQ:contrast('G398/AQ','G421/AQ',11),
    G398MinusG421Strategy:contrast('G398/Strategy','G421/Strategy',11),
    G421MinusG198AQ:contrast('G421/AQ','G198/AQ',11)};
  const phaseEffects=Object.fromEntries(Object.entries(p.swaps).map(([id,phase])=>[phase,contrast(id+'/AQ','G198/AQ',3)]));
  return {schema:'rnad-i5-paired-phase-result-v1',planSha,pinned,source:p.source,engine:p.engine,aqSHA:p.aq,seedStart:p.seedStart,
    independentSeedBlocks:p.blocks,completedMatches:p.blocks*2*groups.size,rates,changes,phaseEffects,researchOnly:true,championPromoted:false};
}
if(op==='smoke'){
  const data=evalPart(p.smokeSeed,2);
  assert.equal(data.rows.length,36);
  console.log('36 smoke matches verified; excluded from the held-out report');
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
