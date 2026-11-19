// Offline comparative CPU study; never invoked from authoritative policy.choose.
import {readFileSync,readdirSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {createRequire} from 'node:module';
import {performance} from 'node:perf_hooks';
import {PersistentRNADInference} from './infer.mjs';

const args=Object.fromEntries(process.argv.slice(2).reduce((acc,x,i,a)=>{
  if(i%2===0)acc.push([x.slice(2),a[i+1]]);
  return acc;
},[]));
const assert=(flag,message)=>{if(!flag)throw Error(message)};
const require=createRequire(resolve(args.deps,"package.json"));
const ort=require("onnxruntime-node");
const onnx=resolve(args.onnx),battle=resolve(args.battle);
const manifest=JSON.parse(readFileSync(resolve(args.manifest)));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
assert(sha(readFileSync(onnx))===manifest.onnx_sha256,
       'Offline ONNX candidate bytes modified');
assert(sha(readFileSync(battle))===manifest.source_actor_battle_sha256,
       'Frozen Actor battle JSON modified');
const readAccepted=dir=>readdirSync(dir).filter(x=>x.endsWith('.json.gz'))
  .sort().map(name=>JSON.parse(gunzipSync(readFileSync(join(dir,name)))));
const matches=[
  ...readAccepted(join(args.smoke,'self','accepted')).map(row=>({row,self:true})),
  ...readAccepted(join(args.smoke,'frozen','accepted')).map(row=>({row,self:false})),
];
assert(matches.length===4&&matches.every(x=>x.row.completed&&
  x.row.score[x.row.winner]===4),'Not four authoritative complete Matches');
const trials=matches.map(({row,self})=>row.events.filter(e=>
  e.phase==='battle'&&(self||e.actor===row.learner_seat)).map(e=>({
    seat:e.actor,state:e.phase_input.state,actions:e.phase_input.actions,
})));
assert(trials.every(x=>x.length>0),'Missing own-seat Battle events');
const ortSession=await ort.InferenceSession.create(onnx,{
  executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1,
});
const makeInputs=(e,memory)=>({
  state:new ort.Tensor('float32',Float32Array.from(e.state),[1,96]),
  actions:new ort.Tensor('float32',
    Float32Array.from(e.actions.flat()),[1,e.actions.length,48]),
  memory:new ort.Tensor('float32',memory,[1,128]),
});
const sameResults=async()=>{
  let maxAbs=0,decisions=0;
  for(const trial of trials){
    const jsNet=new PersistentRNADInference(battle);
    const memories=[new Float32Array(128),new Float32Array(128)];
    for(const e of trial){
      const native=await ortSession.run(makeInputs(e,memories[e.seat]));
      const baseline=jsNet.infer(e.state,e.actions,e.seat);
      for(let i=0;i<e.actions.length;i++){
        maxAbs=Math.max(maxAbs,Math.abs(native.logits.data[i]-baseline.logits[i]));
      }
      maxAbs=Math.max(maxAbs,Math.abs(native.value.data[0]-baseline.value));
      memories[e.seat]=new Float32Array(native.next_memory.data);
      decisions++;
    }
  }
  assert(maxAbs<=2e-4,'Frozen Actor JS/ONNX numerical parity exceeded 2e-4: '+maxAbs);
  return {maxAbs,decisions};
};
const jsPass=()=>{
  for(const trial of trials){
    const nn=new PersistentRNADInference(battle);
    for(const e of trial)nn.infer(e.state,e.actions,e.seat);
  }
};
const onnxPass=async()=>{
  for(const trial of trials){
    let mem=[new Float32Array(128),new Float32Array(128)];
    for(const e of trial){
      const x=await ortSession.run(makeInputs(e,mem[e.seat]));
      mem[e.seat]=new Float32Array(x.next_memory.data);
    }
  }
};
const parity=await sameResults();
jsPass();await onnxPass();
const ms=time=>Math.round(time*1000)/1000;
const jsTimes=[],ortTimes=[];
for(let i=0;i<3;i++){
  let start=performance.now();
  jsPass();
  jsTimes.push(ms(performance.now()-start));
  start=performance.now();
  await onnxPass();
  ortTimes.push(ms(performance.now()-start));
}
const avg=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
process.stdout.write(JSON.stringify({
  schema:'rnad-i5-offline-native-battle-cpu-study-v1',
  accepted_complete_matches:matches.length,
  battle_decisions_per_trial:parity.decisions,
  frozen_js_ort_max_abs:parity.maxAbs,
  js_trial_ms:jsTimes,onnx_trial_ms:ortTimes,
  mean_js_ms:ms(avg(jsTimes)),mean_onnx_ms:ms(avg(ortTimes)),
  ratio_js_over_onnx_not_generation_speed:ms(avg(jsTimes)/avg(ortTimes)),
  synchronous_authority_not_modified:true,
  production_onnx_actor_enabled:false,
  champion_promoted:false
})+'\n');
