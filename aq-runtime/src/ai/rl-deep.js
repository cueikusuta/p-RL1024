// Research-only RL-Deep. No teacher scores, hidden state or tactical overrides.
import {rankPureRlAllPhases} from './pure-rl-phases.js';
const D=256,H=16,N=D*H+H+H+1+H+1;
const B=D*H,P=B+H,PB=P+H,V=PB+1,VB=V+H;
export const DEEP_SCHEMA='i976-hashed-mlp-ppo-v1';
export function random(seed){let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};}
export function createDeep(seed=976){
  const rng=random(seed),w=Array(N).fill(0);
  for(let i=0;i<B;i++)w[i]=(rng()*2-1)*Math.sqrt(3/D);
  for(let j=0;j<H;j++){w[P+j]=(rng()*2-1)*.05;w[V+j]=(rng()*2-1)*.05;}
  return {schema:DEEP_SCHEMA,input:D,hidden:H,w,m:Array(N).fill(0),v:Array(N).fill(0),step:0};
}
export function validateDeep(m){
  if(m?.schema!==DEEP_SCHEMA||m.input!==D||m.hidden!==H||
    !Number.isSafeInteger(m.step)||m.step<0||
    !['w','m','v'].every(k=>Array.isArray(m[k])&&m[k].length===N&&m[k].every(Number.isFinite))||
    m.w.some(x=>Math.abs(x)>100)||m.v.some(x=>x<0))throw Error('Invalid RL-Deep checkpoint');
  return m;
}
export function encodeDeep(features){
  const buckets=new Map();
  for(const [key,value] of Object.entries(features)){
    if(!Number.isFinite(value)||Math.abs(value)>10)throw Error('Invalid public feature');
    let hash=2166136261;for(let i=0;i<key.length;i++)hash=Math.imul(hash^key.charCodeAt(i),16777619);
    const index=(hash>>>0)%D,sign=(hash&0x10000)?1:-1;
    buckets.set(index,(buckets.get(index)??0)+sign*value);
  }
  const norm=Math.sqrt([...buckets.values()].reduce((s,x)=>s+x*x,0))||1;
  return [...buckets].sort((a,b)=>a[0]-b[0]).map(([i,x])=>[i,x/norm]);
}
// A shared nonlinear candidate encoder; the value head pools ALL legal
// candidates, so its baseline is independent of the sampled action.
export function forwardDeep(model,xs){
  const w=model.w;
  const h=xs.map(x=>Array.from({length:H},(_,j)=>{
    let z=w[B+j];for(const [i,a] of x)z+=w[j*D+i]*a;return Math.tanh(z);
  }));
  const logits=h.map(row=>row.reduce((s,x,j)=>s+x*w[P+j],w[PB]));
  const max=Math.max(...logits),exp=logits.map(z=>Math.exp(z-max)),sum=exp.reduce((s,x)=>s+x,0);
  const probs=exp.map(x=>x/sum),mean=Array(H).fill(0);
  for(const row of h)for(let j=0;j<H;j++)mean[j]+=row[j]/h.length;
  const value=Math.tanh(mean.reduce((s,x,j)=>s+x*w[V+j],w[VB]));
  return {h,probs,mean,value};
}
export function deepLossGradient(model,row,{clip=.2,entropy=.01}={}){
  const f=forwardDeep(model,row.xs),{probs,h,value,mean}=f;
  const a=row.selected,adv=row.reward-row.oldValue,ratio=probs[a]/row.oldProb;
  const clipped=Math.max(1-clip,Math.min(1+clip,ratio));
  const useGradient=!(adv>=0&&ratio>1+clip)&&!(adv<0&&ratio<1-clip);
  const scale=useGradient?-adv*ratio:0;
  const ent=-probs.reduce((s,p)=>s+p*Math.log(Math.max(p,1e-300)),0);
  const loss=-Math.min(ratio*adv,clipped*adv)+.5*(value-row.reward)**2-entropy*ent;
  const g=new Float64Array(N),dv=(value-row.reward)*(1-value*value);
  g[VB]=dv;for(let j=0;j<H;j++)g[V+j]=dv*mean[j];
  for(let k=0;k<probs.length;k++){
    const dz=scale*((k===a?1:0)-probs[k])+entropy*probs[k]*(Math.log(Math.max(probs[k],1e-300))+ent);
    g[PB]+=dz;
    for(let j=0;j<H;j++){
      g[P+j]+=dz*h[k][j];
      const dh=(dz*model.w[P+j]+dv*model.w[V+j]/h.length)*(1-h[k][j]**2);
      g[B+j]+=dh;for(const [i,x] of row.xs[k])g[j*D+i]+=dh*x;
    }
  }
  return {loss,g,entropy:ent,valueLoss:(value-row.reward)**2};
}
export function updateDeep(model,rows,{epochs=3,rate=.001,entropy=.01}={}){
  validateDeep(model);
  if(!rows.length||!Number.isInteger(epochs)||epochs<1||epochs>8||!Number.isFinite(rate)||rate<=0||rate>.01||!Number.isFinite(entropy)||entropy<0||entropy>.1)throw Error('Invalid PPO update');
  let loss=0,valueLoss=0,ent=0;
  for(let epoch=0;epoch<epochs;epoch++){
    const grad=new Float64Array(N);loss=0;valueLoss=0;ent=0;
    for(const row of rows){const r=deepLossGradient(model,row,{entropy});loss+=r.loss/rows.length;valueLoss+=r.valueLoss/rows.length;ent+=r.entropy/rows.length;for(let i=0;i<N;i++)grad[i]+=r.g[i]/rows.length;}
    const norm=Math.sqrt(grad.reduce((s,x)=>s+x*x,0)),scale=Math.min(1,.5/(norm||1));
    model.step++;
    for(let i=0;i<N;i++){
      const g=grad[i]*scale;model.m[i]=.9*model.m[i]+.1*g;model.v[i]=.999*model.v[i]+.001*g*g;
      model.w[i]-=rate*(model.m[i]/(1-.9**model.step))/(Math.sqrt(model.v[i]/(1-.999**model.step))+1e-8);
    }
    validateDeep(model);
  }
  return {loss,valueLoss,entropy:ent,decisions:rows.length,updates:epochs};
}
const neutral=Object.freeze({schema:'i975-pure-linear-sarsa-v1',weights:{}});
export function createDeepSession(model,{seat=0,seed=1,train=false,greedy=false}={}){
  validateDeep(model);
  if(![0,1].includes(seat)||!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff)throw Error('Invalid session');
  const rng=random(seed),trajectory=[],times=[],phaseTimes={draft:[],selection:[],battle:[]};
  let pending=null,finished=false,streak=0,decisions=0;
  return {
    trajectory,times,phaseTimes,
    select(view){
      if(pending||finished)throw Error('Session not ready');
      const start=performance.now();
      const candidates=rankPureRlAllPhases(neutral,view,{seat,consecutiveOwnSwitches:streak});
      if(!candidates.length)throw Error('No legal candidates');
      const xs=candidates.map(c=>encodeDeep(c.features)),f=forwardDeep(model,xs);
      let selected=0;
      if(greedy){for(let i=1;i<f.probs.length;i++)if(f.probs[i]>f.probs[selected])selected=i;}
      else{const u=rng();let c=0;for(let i=0;i<f.probs.length;i++){c+=f.probs[i];selected=i;if(u<c)break;}}
      const action=candidates[selected].action;
      pending={xs,selected,oldProb:f.probs[selected],oldValue:f.value,voluntarySwitch:action.kind==='switch'&&view.requestType==='turn'};
      const elapsed=performance.now()-start;times.push(elapsed);
      phaseTimes[['draft','selection'].includes(view.phase)?view.phase:'battle'].push(elapsed);
      return structuredClone(action);
    },
    commit(){
      if(!pending||finished)throw Error('No accepted action');
      streak=pending.voluntarySwitch?Math.min(4,streak+1):0;
      if(train){const {voluntarySwitch,...row}=pending;trajectory.push(row);}
      pending=null;decisions++;
    },
    finish(reward){
      if(pending||finished||![-1,1].includes(reward))throw Error('Invalid terminal outcome');
      for(const row of trajectory)row.reward=reward;
      finished=true;return {decisions,updates:0,train};
    },
    stats(){return {decisions,updates:0,train,finished};}
  };
}
