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
assert.equal(p.schema,'rnad-i5-g421-paired-2026-11-v1');
assert.equal(p.engine,provenance.sourceCommit);
assert.equal(p.source,verifiedRunningSourceSha());
assert.equal(p.aq,weightsSha256);
assert.equal(p.seedStart,400100001);
assert.equal(p.smokeSeed,400101001);
assert.equal(p.blocks,512);
assert.equal(p.parts,16);
assert.equal(p.alpha,.05);
assert.deepEqual(Object.keys(p.models),['G198','G398','G421']);
assert.deepEqual(p.matchups,[['G421','AQ'],['G198','AQ'],['G398','AQ'],['G421','G198'],['G421','G398']]);
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
  for(const id of Object.keys(p.models)){
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
    manifestSha[id]=sha(raw);
  }
  const rows=[];
  for(const [left,right] of p.matchups) for(let seed=start;seed<start+count;seed++) for(const seat of [0,1]){
    const ids=seat===0?[left,right]:[right,left];
    const sessions=ids.map((id,s)=>id==='AQ'?null:createIntegratedAQSession({modelDir:join(models,id),seat:s}));
    const result=runStrategyMatch({seed,policies:ids.map(id=>id==='AQ'?'aq-base':'pure-rl-linear'),
      pureRlSessions:sessions,pureRlAllPhases:true,strategyMode:'integrated',
      ...(right==='AQ'?{v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq}:{})});
    const {winner,score,rounds}=result;
    assert(result.completed===true && [0,1].includes(winner) && Array.isArray(score) && score.length===2 &&
      score[winner]===4 && Number.isSafeInteger(score[1-winner]) && score[1-winner]>=0 && score[1-winner]<4 &&
      Number.isSafeInteger(rounds) && rounds>=4 && rounds<=64,`incomplete ${left}/${right} ${seed} seat ${seat}`);
    const accepted=ids.map((id,s)=>{
      if(id==='AQ')return null;
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
    const x=read(join(root,'g421-paired-part-'+i,'shard.json'));
    assert.equal(x.planSha,planSha);
    assert.equal(x.noTraining,true);
    assert.equal(x.part,i);
    assert.equal(x.start,p.seedStart+i*count);
    assert.equal(x.count,count);
    assert.equal(x.rows.length,count*2*p.matchups.length);
    assert.equal(Object.values(x.manifestSha).filter(s=>sha64.test(s)).length,3);
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
        assert(id==='AQ'?r.accepted[s]===null:Number.isSafeInteger(r.accepted[s])&&r.accepted[s]>2*r.rounds);
      g[k][r.seat]=Number(r.winner===r.seat);
    }
  }
  const block=key=>groups.get(key).map(a=>{assert(a.every(v=>v!==null));return (a[0]+a[1])/2;});
  const opt={familywiseAlpha:p.alpha,comparisons:7};
  const rates=Object.fromEntries([...groups.keys()].map(k=>{const b=block(k);return [k,{wins:b.reduce((a,v)=>a+2*v,0),matches:2*p.blocks,
    ...pairedClusterCI(b,{...opt,range:[0,1],nullValue:.5})}];}));
  const changes=Object.fromEntries(['G198','G398'].map(id=>[id,pairedClusterCI(block('G421/AQ').map((x,i)=>x-block(id+'/AQ')[i]),opt)]));
  return {schema:'rnad-i5-g421-paired-result-v1',planSha,pinned,source:p.source,engine:p.engine,aqSHA:p.aq,seedStart:p.seedStart,
    independentSeedBlocks:p.blocks,completedMatches:p.blocks*2*groups.size,rates,changes,researchOnly:true,championPromoted:false};
}
if(op==='smoke'){
  const data=evalPart(p.smokeSeed,2);
  assert.equal(data.rows.length,20);
  console.log('20 smoke matches verified; excluded from the held-out report');
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
