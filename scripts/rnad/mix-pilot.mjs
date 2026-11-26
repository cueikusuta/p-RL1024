import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {createIntegratedAQSession} from './evaluate-shared-g012-vs-aq.mjs';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256} from '../../models/aq-base.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import {studentTQuantile} from './paired-league-stats.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const [cmd,planFile,root,repName,partText,out]=process.argv.slice(2);
const sha=data=>createHash('sha256').update(data).digest('hex');
const p=JSON.parse(readFileSync(planFile));
const planSha=sha(readFileSync(planFile));
const head=verifiedRunningSourceSha();
const aq=createOpponent();
assert.equal(p.schema,'rnad-i5-opponent-mix-pilot-v1');
assert.equal(p.engine,provenance.sourceCommit);
assert.equal(p.aq,weightsSha256);
assert.equal(p.aq,aq.modelSha256);
assert.equal(p.generation,422);
assert.equal(p.screen.blocks,128);
assert.equal(p.evaluation.blocksPerRep,512);
assert.equal(p.evaluation.partsPerRep,8);
assert.deepEqual(p.evaluation.arms,['control','weakness']);
const groups=['control/AQ','weakness/AQ','control/Strategy','weakness/Strategy','weakness/control'];
const repNames=['rep0','rep1'];
const opponents={AQ:'aq-base',Strategy:p.evaluation.heldOutPolicy};
assert.equal(opponents.Strategy,'strategy');
const read=path=>JSON.parse(readFileSync(path));
function sealed(dir,generation){
  const raw=readFileSync(join(dir,'manifest.json')),m=JSON.parse(raw);
  assert.equal(m.generation,generation);
  assert.equal(m.engine_sha,p.engine);
  assert.equal(m.source_sha,head);
  assert.deepEqual(Object.keys(m.files).sort(),['battle','draft','selection','values']);
  for(const [key,digest] of Object.entries(m.files))
    assert.equal(sha(readFileSync(join(dir,key+'.json'))),digest);
  return sha(raw);
}
function play(dirs,left,right,seed,seat){
  const ids=seat===0?[left,right]:[right,left];
  const result=runStrategyMatch({
    seed,policies:ids.map(id=>opponents[id]??'pure-rl-linear'),
    pureRlSessions:ids.map((id,slot)=>opponents[id]?null:
      createIntegratedAQSession({modelDir:dirs[id],seat:slot})),
    pureRlAllPhases:true,strategyMode:'integrated',
    ...(right==='AQ'?{v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq}:{})});
  const {winner,rounds,score}=result;
  assert(result.completed===true&&[0,1].includes(winner)&&Array.isArray(score)&&score.length===2&&
    score[winner]===4&&Number.isSafeInteger(score[1-winner])&&score[1-winner]>=0&&score[1-winner]<4&&
    Number.isSafeInteger(rounds)&&rounds>=4&&rounds<=64,`Incomplete ${left}/${right} seed ${seed}, seat ${seat}`);
  for(const [slot,id] of ids.entries()){
    if(opponents[id])continue;
    const s=result.pureRlStats?.[slot];
    assert(s?.updates===0&&s.pending===false&&s.accepted_decisions===result.actionCounts[slot]+2*rounds);
  }
  return Number(winner===seat);
}
function screen(){
  const dirs={G421:join(root,'actor')};
  const pins={G421:sealed(dirs.G421,422)};
  const counts=[];
  for(const id of p.screen.history){
    dirs[id]=join(root,'past',id);
    pins[id]=sealed(dirs[id],Number(id.slice(6)));
    const rows=[];
    for(let i=0;i<p.screen.blocks;i++)rows.push([0,1].map(seat=>play(dirs,'G421',id,p.screen.start+i,seat)));
    counts.push({id,wins:rows.flat().reduce((a,b)=>a+b,0),rows});
  }
  counts.sort((a,b)=>a.wins-b.wins||a.id.localeCompare(b.id));
  return {schema:'rnad-i5-mix-screen-v1',planSha,head,screenStart:p.screen.start,blocks:p.screen.blocks,pins,counts};
}
function evaluate(rep,part,smoke=false){
  assert(repNames.includes(rep)&&Number.isInteger(part)&&part>=0&&part<8);
  const count=smoke?1:p.evaluation.blocksPerRep/p.evaluation.partsPerRep;
  const start=smoke?p.evaluation.smokeStart:p.evaluation[rep]+part*count;
  const dirs={},pins={},training={};
  for(const arm of p.evaluation.arms){
    const base=join(root,'i5-mix-trained-'+rep+'-'+arm);
    dirs[arm]=join(base,'generation423-actor');
    pins[arm]=sealed(dirs[arm],423);
    training[arm]=read(join(base,'training-receipt.json'));
    assert.equal(training[arm].arm,arm);
    assert.equal(training[arm].rep,rep);
    assert.equal(training[arm].head,head);
    assert.equal(training[arm].planSha,planSha);
    assert.equal(training[arm].trainSeed,p.training[rep]);
    assert.equal(training[arm].actorSha,pins[arm]);
    assert.equal(training[arm].acceptedMatches,1280);
    assert.equal(training[arm].updates,40);
  }
  assert.equal(training.control.originalCheckpoint,training.weakness.originalCheckpoint);
  assert.notEqual(pins.control,pins.weakness);
  const groupsOut={};
  for(const key of groups){
    const [l,r]=key.split('/');
    groupsOut[key]=Array.from({length:count},(_,i)=>[0,1].map(seat=>play(dirs,l,r,start+i,seat)));
  }
  return {rep,part,start,count,pins,training,groups:groupsOut,planSha,head};
}
function summarize(){
  const all={};
  for(const rep of repNames){
    const blocks=Object.fromEntries(groups.map(k=>[k,[]]));
    let evidence;
    for(let i=0;i<8;i++){
      const x=read(join(root,'i5-mix-eval-'+rep+'-'+i,'shard.json'));
      assert.equal(x.rep,rep);assert.equal(x.part,i);assert.equal(x.planSha,planSha);assert.equal(x.head,head);
      assert.equal(x.count,64);assert.equal(x.start,p.evaluation[rep]+64*i);
      const item={pins:x.pins,training:x.training};
      if(evidence)assert.deepEqual(item,evidence);else evidence=item;
      assert.deepEqual(Object.keys(x.groups).sort(),groups.slice().sort());
      for(const key of groups) {
        const v=x.groups[key];
        assert(v.length===64&&v.every(y=>Array.isArray(y)&&y.length===2&&y.every(w=>w===0||w===1)));
        blocks[key].push(...v.map(y=>(y[0]+y[1])/2));
      }
    }
    all[rep]={blocks,...evidence};
  }
  const diff=(b,k)=>b['weakness/'+k].map((v,i)=>v-b['control/'+k][i]);
  const pool=fn=>repNames.flatMap(r=>fn(all[r].blocks));
  const rates=b=>Object.fromEntries(groups.map(k=>[k,{wins:b[k].reduce((a,v)=>a+2*v,0),
    matches:2*b[k].length,twoWins:b[k].filter(x=>x===1).length,
    split:b[k].filter(x=>x===.5).length,twoLosses:b[k].filter(x=>x===0).length}]));
  const moments=(x)=>{
    const n=x.length,estimate=x.reduce((a,v)=>a+v,0)/n;
    const variance=x.reduce((a,v)=>a+(v-estimate)**2,0)/(n-1);
    return {n,estimate,variance};
  };
  // Treat the two independently seeded evaluation strata as fixed. Pool their
  // within-stratum variances, not their between-training-run variation.
  function stratified(extract,{comparisons=1,nullValue=0,range=[-1,1]}={}){
    const m=repNames.map(r=>moments(extract(all[r].blocks)));
    const n=m[0].n;assert(m.every(x=>x.n===n));
    const raw=pool(extract);
    // Conservative relative to Welch-Satterthwaite's >= n-1 df.
    const t=studentTQuantile(1-.05/(2*comparisons),n-1);
    const variance=(m[0].variance+m[1].variance)/(4*n);
    const estimate=(m[0].estimate+m[1].estimate)/2;
    const w=t*Math.sqrt(variance);
    return {estimate,nullValue,se:Math.sqrt(variance),comparisons,seedBlocks:raw.length,degreesOfFreedom:n-1,
      ci:[Math.max(range[0],estimate-w),Math.min(range[1],estimate+w)],
      higher:raw.filter(x=>x>nullValue).length,same:raw.filter(x=>x===nullValue).length,
      lower:raw.filter(x=>x<nullValue).length,perRep:m.map(x=>x.estimate)};
  }
  const primary=stratified(b=>diff(b,'Strategy'));
  const aqContrast=stratified(b=>diff(b,'AQ'),{comparisons:2});
  const headToHead=stratified(b=>b['weakness/control'],{comparisons:2,nullValue:.5,range:[0,1]});
  const go=primary.ci[0]>0&&primary.perRep.every(x=>x>0)&&aqContrast.ci[0]>-.02;
  return {schema:'rnad-i5-mix-pilot-result-v1',planSha,head,engine:p.engine,baseCheckpoint:p.bootstrap.checkpoint,
    matches:10240,seedBlocks:1024,evidence:Object.fromEntries(repNames.map(rep=>[rep,{
      training:all[rep].training,rates:rates(all[rep].blocks)}])),
    primary,aqContrast,headToHead,considerLongerConfirmation:go,researchOnly:true,promoted:false};
}
let result;
if(cmd==='screen')result=screen();
else if(cmd==='run')result=evaluate(repName,Number(partText));
else if(cmd==='smoke'){for(const rep of repNames)evaluate(rep,0,true);console.log('Both replicates, both seats, five groups, excluded smoke');process.exit(0);}
else if(cmd==='summary')result=summarize();
else throw Error('Unknown command');
writeFileSync(out,JSON.stringify(result)+'\n',{flag:'wx'});
if(cmd==='summary')console.log(JSON.stringify(result));
else if(cmd==='screen')console.log(JSON.stringify(result.counts.map(({id,wins})=>({id,wins}))));
