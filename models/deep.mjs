// Independent battle-only actor-critic. Inputs are exactly I975's public sparse features.
import {createHash} from 'node:crypto';
import {legalPureRlActions} from '../src/ai/pure-rl-linear.js';

export const featureVersion='pure-rl-sparse-public';
const H=16, SCHEMA='p-rl1024-deep-actor-critic-v1';
const digest=s=>createHash('sha256').update(s).digest();
function initial(seed,key,j) { return (digest(`${seed}\0${key}\0${j}`).readUInt32BE(0)/0xffffffff-0.5)*0.08; }
const initialCache=new Map();
function initialVector(seed,key) {
  const id=`${seed}\0${key}`;
  if(!initialCache.has(id))initialCache.set(id,Array.from({length:H},(_,j)=>initial(seed,key,j)));
  return initialCache.get(id);
}
const clone=x=>structuredClone(x);
export function createModel(seed=975020) {
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff)throw new Error('Invalid model seed');
  return {schema:SCHEMA,featureVersion,seed,hidden:H,weights:{},actor:Array.from({length:H},(_,j)=>initial(seed,'actor',j)),critic:Array.from({length:H},(_,j)=>initial(seed,'critic',j)),actorBias:0,criticBias:0};
}
export function validateModel(model) {
  if(!model||model.schema!==SCHEMA||model.featureVersion!==featureVersion||model.hidden!==H||!Number.isSafeInteger(model.seed)||model.seed<1||model.seed>0xffffffff||!model.weights||typeof model.weights!=='object'||Array.isArray(model.weights)||Object.keys(model.weights).length>500000)throw new Error('Invalid deep model');
  const validVector=v=>Array.isArray(v)&&v.length===H&&v.every(x=>Number.isFinite(x)&&Math.abs(x)<1e6);
  if(!validVector(model.actor)||!validVector(model.critic)||!Number.isFinite(model.actorBias)||!Number.isFinite(model.criticBias)||!Object.entries(model.weights).every(([k,v])=>/^[a-z0-9:._|=\/-]{1,128}$/i.test(k)&&validVector(v)))throw new Error('Invalid deep model weights');
  return model;
}
export const loadModel=text=>validateModel(JSON.parse(text));
export const saveModel=model=>JSON.stringify(validateModel(model));
function hidden(model,features) {
  const out=Array(H).fill(0);
  for(const [key,x] of Object.entries(features)) {
    const w=model.weights[key]??initialVector(model.seed,key);
    for(let j=0;j<H;j++)out[j]+=x*w[j];
  }
  return out.map(Math.tanh);
}
const dot=(a,b)=>a.reduce((s,x,j)=>s+x*b[j],0);
// Trusted projection of the exact I975 sparse features. Its caller obtains
// the action from legalPureRlActions(view) first. Keeping the original feature
// insertion sequence preserves the floating-point accumulation order.
const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));
const validName=x=>typeof x==='string'&&/^[a-z0-9:._|=/-]{1,128}$/i.test(x);
const asName=x=>typeof x==='string'&&validName(x)?x.toLowerCase():'unknown';
const health=p=>clamp(p.hp/Math.max(1,p.maxHP),0,1);
const band=(ratio,offset=0)=>Math.min(4,Math.floor((ratio*100+offset)/25));
export function trustedFeatures(view,action){
  const self=view.self,active=self.team[self.active],foe=view.opponent;
  if(!foe||typeof foe.hpPercent!=='number'||!Number.isFinite(foe.hpPercent)||
    typeof active.hp!=='number'||!Number.isFinite(active.hp)||
    typeof active.maxHP!=='number'||!Number.isFinite(active.maxHP)||
    active.maxHP<=0)throw new TypeError('Invalid public feature inputs');
  const type=action.kind==='switch'?'switch':
    action.index==='struggle'?'struggle':
      asName(active.moves[action.index]?.id);
  const own=health(active),enemy=clamp(foe.hpPercent/100,0,1);
  const keyAction='act:'+type;
  const x=new Map();
  const add=(k,v=1)=>{if(v)x.set(k,(x.get(k)??0)+v);};
  add('bias');
  add(keyAction);
  add('request:'+asName(view.requestType));
  add('turn:'+Math.min(5,Math.floor(Math.max(0,view.turn??1)/5)));
  add('our-status:'+asName(active.status||'none'));
  add('foe-status:'+asName(foe.status||'none'));
  add('our-alive:'+self.team.filter(p=>p.hp>0).length);
  add('foe-remaining:'+clamp(foe.remaining??1,0,3));
  for(const offset of [0,12]){
    add('our-hp:'+offset+':'+band(own,offset));
    add('foe-hp:'+offset+':'+band(enemy,offset));
    add(keyAction+'|our-hp:'+offset+':'+band(own,offset));
    add(keyAction+'|foe-hp:'+offset+':'+band(enemy,offset));
  }
  add(keyAction+'|foe:'+asName(foe.speciesId));
  for(const t of (Array.isArray(foe.types)?foe.types:[]))
    add(keyAction+'|foe-type:'+asName(t));
  for(const t of (Array.isArray(active.types)?active.types:[]))
    add(keyAction+'|own-type:'+asName(t));
  if(action.kind==='switch'){
    const target=self.team[action.index];
    add('switch-hp:'+band(health(target)));
    add('switch-species:'+asName(target.speciesId));
    add('switch-status:'+asName(target.status||'none'));
  }else{
    add('move-slot:'+String(action.index));
    if(action.targetIndex!==undefined)add('revival-target-hp:0');
    if(action.mega)add('mega:yes');
  }
  if(view.field?.trickRoomTurns>0)add('field:trickroom');
  for(const field of ['weather','terrain'])
    add('field:'+field+':'+asName(view.field?.[field]||'none'));
  return Object.fromEntries(x);
}
function candidates(model,view,legal,{validated=false,alreadyLegal=false}={}) {
  if(!validated)validateModel(model);
  const actions=alreadyLegal?legal:(()=>{
    const allowed=new Set(legal.map(a=>JSON.stringify(a)));
    return legalPureRlActions(view).filter(a=>allowed.has(JSON.stringify(a)));
  })();
  const rows=actions.map(action=>{
    const features=trustedFeatures(view,action),h=hidden(model,features);
    return {action,features,h,logit:dot(model.actor,h)+model.actorBias};
  });
  if(!rows.length&&legal.length)throw new Error('No permitted action');
  const max=Math.max(...rows.map(r=>r.logit));
  const exp=rows.map(r=>Math.exp(r.logit-max)),sum=exp.reduce((a,b)=>a+b,0);
  for(let i=0;i<rows.length;i++)rows[i].probability=exp[i]/sum;
  const mean=Array.from({length:H},(_,j)=>rows.reduce((s,r)=>s+r.h[j],0)/rows.length);
  return {rows,mean,value:Math.tanh(dot(model.critic,mean)+model.criticBias)};
}
export function rankActions(model,view,legal=legalPureRlActions(view)) {
  return candidates(model,view,legal).rows.map(({action,logit,probability})=>({action,q:logit,probability})).sort((a,b)=>b.q-a.q);
}
function sample(rows,rng) {
  const v=rng.integer(0x1000000)/0x1000000;let p=0;
  for(let i=0;i<rows.length;i++){p+=rows[i].probability;if(v<p)return i;}
  return rows.length-1;
}
// Collect only proposals accepted by Match. Update after complete Match terminal reward.
export function createSession(model,{train=false,rng=null,alpha=0.0005}={}) {
  validateModel(model);if(train&&(!rng||typeof rng.integer!=='function'))throw new Error('Training needs seeded RNG');
  let proposed=null,finished=false,decisions=0,updates=0;const trajectory=[];
  return {
    select(view) {
      if(finished||proposed)throw new Error('Session is not ready');
      const legal=legalPureRlActions(view);
      if(!legal.length)throw new Error('No legal action');
      // Frozen inference needs only the winning logit. The full softmax,
      // critic, mean value and per-action trajectories serve training only.
      // This preserves candidate order and the original 1e-12 tie rule.
      if(!train){
        let index=0;
        let bestLogit=dot(model.actor,hidden(model,trustedFeatures(view,legal[0])))+model.actorBias;
        for(let i=1;i<legal.length;i++){
          const logit=dot(model.actor,hidden(model,trustedFeatures(view,legal[i])))+model.actorBias;
          if(logit>bestLogit+1e-12){index=i;bestLogit=logit;}
        }
        proposed={frozen:true};
        return clone(legal[index]);
      }
      const c=candidates(model,view,legal,{validated:true,alreadyLegal:true});
      const index=sample(c.rows,rng);
      proposed={...c,index};return clone(c.rows[index].action);
    },
    commit() { if(!proposed||finished)throw new Error('No accepted action');if(train)trajectory.push(proposed);proposed=null;decisions++; },
    finish(reward) {
      if(finished||proposed||![-1,1].includes(reward))throw new Error('Invalid terminal reward');
      if(train&&trajectory.length) {
        const actor=Array(H).fill(0),critic=Array(H).fill(0),wgrad={},scale=1/trajectory.length;let cb=0;
        for(const t of trajectory) {
          const advantage=reward-t.value,dv=advantage*(1-t.value*t.value);cb+=dv*scale;
          for(let j=0;j<H;j++)critic[j]+=dv*t.mean[j]*scale;
          for(let i=0;i<t.rows.length;i++) {
            const row=t.rows[i],dlogit=advantage*((i===t.index?1:0)-row.probability)*scale;
            for(let j=0;j<H;j++)actor[j]+=dlogit*row.h[j];
            for(const [key,x] of Object.entries(row.features)) {
              const g=wgrad[key]??=Array(H).fill(0);
              for(let j=0;j<H;j++)g[j]+=(dlogit*model.actor[j]+dv*scale*model.critic[j]/t.rows.length)*(1-row.h[j]*row.h[j])*x;
            }
          }
        }
        const clip=x=>Math.max(-5,Math.min(5,x));
        for(let j=0;j<H;j++){model.actor[j]+=alpha*clip(actor[j]);model.critic[j]+=alpha*clip(critic[j]);}
        model.criticBias+=alpha*clip(cb);
        for(const [key,grad] of Object.entries(wgrad)) {const w=model.weights[key]??=Array.from({length:H},(_,j)=>initial(model.seed,key,j));for(let j=0;j<H;j++)w[j]+=alpha*clip(grad[j]);}
        validateModel(model);updates=1;
      }
      finished=true;return {decisions,updates,train};
    },
    stats(){return {decisions,updates,train,finished};}
  };
}
