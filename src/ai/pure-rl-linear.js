// I975: battle-only, research-only pure reinforcement learning.
// Rules/legality come from the authoritative Battle; neither policy nor features
// import v2-E, rankings, threat scores, model labels, or opponent-private data.
export const PURE_RL_SCHEMA = 'i975-pure-linear-sarsa-v1';
const finite = n => typeof n === 'number' && Number.isFinite(n);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const validName = x => typeof x === 'string' && /^[a-z0-9:._|=/-]{1,128}$/i.test(x);
const health = p => clamp(p.hp / Math.max(1, p.maxHP), 0, 1);
const band = (ratio, offset=0) => Math.min(4, Math.floor((ratio*100+offset)/25));
const asName = value => typeof value === 'string' && validName(value)
  ? value.toLowerCase() : 'unknown';

export function legalPureRlActions(view) {
  if (!view || !view.self || !Array.isArray(view.self.team) ||
      !Number.isInteger(view.self.active) ||
      !view.self.team[view.self.active] ||
      !['turn','replacement','pivot','wait','finished'].includes(view.requestType))
    throw new TypeError('Invalid public battle decision view');
  const self=view.self, active=self.team[self.active];
  if (self.confirmed || ['wait','finished'].includes(view.requestType) ||
      (view.requestType==='replacement' && active.hp>0 && !self.needsSwitch))
    return [];
  const forced=view.requestType==='pivot' || view.requestType==='replacement';
  const switches=[];
  if (self.canSwitch) for(let i=0;i<self.team.length;i++)
    if(i!==self.active && self.team[i]?.hp>0)
      switches.push({kind:'switch',index:i});
  if (forced || active.hp<=0) {
    if(!switches.length)throw new Error('No legal replacement available');
    return switches;
  }
  if(view.requestType!=='turn')throw new TypeError('Unknown battle turn');
  const moves=[];
  if(!Array.isArray(active.moves))throw new TypeError('Missing own moves');
  const variants=Boolean(self.canMega)?[false,true]:[false];
  for(let index=0;index<active.moves.length;index++) {
    const slot=active.moves[index];
    if(!slot || typeof slot.id!=='string' || slot.disabled!==false)continue;
    const targets=slot.id==='revivalblessing'
      ?self.team.flatMap((p,i)=>i!==self.active && p.hp<=0?[i]:[])
      :[null];
    for(const target of targets)for(const mega of variants) {
      const move={kind:'move',index};
      if(target!==null)move.targetIndex=target;
      if(mega)move.mega=true;
      moves.push(move);
    }
  }
  if(!moves.length)for(const mega of variants)
    moves.push({kind:'move',index:'struggle',...(mega?{mega:true}:{})});
  return [...moves,...switches];
}

// Only whitelisted own/public values from decisionViewForAI are observed.
// Numeric scalars are tile-coded, while categorical features are sparse.
// This function deliberately never consumes v2-E rankings/scores or labels.
export function pureRlFeatures(view,action) {
  const legal=legalPureRlActions(view);
  if(!legal.some(a=>JSON.stringify(a)===JSON.stringify(action)))
    throw new TypeError('Action is not in the legal public candidate set');
  const self=view.self,active=self.team[self.active],foe=view.opponent;
  if(!foe || !finite(foe.hpPercent) || !finite(active.hp) ||
      !finite(active.maxHP) || active.maxHP<=0)
    throw new TypeError('Invalid public feature inputs');
  const type=action.kind==='switch'?'switch':
    action.index==='struggle'?'struggle':
      asName(active.moves[action.index]?.id);
  const own=health(active),enemy=clamp(foe.hpPercent/100,0,1);
  const keyAction='act:'+type;
  const x=new Map();
  const add=(k,v=1)=>{if(v) x.set(k,(x.get(k)??0)+v);};
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

export function createPureRlModel() {
  return {schema:PURE_RL_SCHEMA,weights:{}};
}
export function validatePureRlModel(model) {
  if(!model || model.schema!==PURE_RL_SCHEMA ||
     !model.weights || typeof model.weights!=='object' ||
     Array.isArray(model.weights) ||
     Object.keys(model.weights).length>500000 ||
     Object.entries(model.weights).some(([k,w])=>
       !validName(k) || !finite(w) || Math.abs(w)>100))
    throw new TypeError('Invalid pure RL linear model');
  return model;
}
export function pureRlValue(model,features) {
  validatePureRlModel(model);
  if(!features || typeof features!=='object' || Array.isArray(features))
    throw new TypeError('Expected sparse public features');
  let value=0;
  for(const [key,x] of Object.entries(features)) {
    if(!validName(key)||!finite(x)||Math.abs(x)>10)
      throw new TypeError('Invalid sparse feature');
    value+=(model.weights[key]??0)*x;
  }
  return value;
}
function dot(trace,features){
  let sum=0;
  for(const [key,x] of Object.entries(features))
    sum+=(trace[key]??0)*x;
  return sum;
}
function randomInteger(rng,n) {
  if(!rng || typeof rng.integer!=='function')
    throw new TypeError('Exploration requires a seeded RNG');
  const result=rng.integer(n);
  if(!Number.isInteger(result)||result<0||result>=n)
    throw new TypeError('Invalid seeded RNG result');
  return result;
}
export function rankPureRlActions(model,view) {
  validatePureRlModel(model);
  const actions=legalPureRlActions(view);
  return actions.map(action=>{
    const features=pureRlFeatures(view,action);
    const q=Object.entries(features).reduce((total,[key,value])=>
      total+(model.weights[key]??0)*value,0);
    return {action,features,q};
  });
}

// A true-online linear SARSA(lambda) session for exactly one seat and one
// complete four-win Match. select() is side-effect-free until commit() is called
// AFTER Match.battleAction accepts the command. The next observation is always
// the next decision of this same seat, across rounds and simultaneous turns.
export function createPureRlSession(model,{alpha=0.025,gamma=1,lambda=0.8,
  epsilon=0,rng=null,train=false,ranker=rankPureRlActions}={}) {
  validatePureRlModel(model);
  for(const [name,value,lo,hi] of [
    ['alpha',alpha,0,1],['gamma',gamma,0,1],
    ['lambda',lambda,0,1],['epsilon',epsilon,0,1]
  ])if(!finite(value)||value<lo||value>hi||(name==='alpha'&&train&&value===0))
    throw new TypeError('Invalid SARSA parameter '+name);
  if(typeof train!=='boolean'||typeof ranker!=='function')
    throw new TypeError('Invalid training flag or independent ranker');
  if(epsilon>0&&(!rng||typeof rng.integer!=='function'))
    throw new TypeError('Exploration requires a seeded RNG');
  const traces=Object.create(null);
  let previous=null,proposed=null,qOld=0,decisions=0,updates=0,finished=false;
  function update(current,reward,nextValue,terminal) {
    if(!train || !previous)return;
    const q=pureRlValue(model,previous);
    const delta=reward+(terminal?0:gamma*nextValue)-q;
    const correction=q-qOld;
    const traceDot=dot(traces,previous);
    for(const key of Object.keys(traces)) {
      traces[key]*=gamma*lambda;
      if(Math.abs(traces[key])<1e-14)delete traces[key];
    }
    const dutch=1-alpha*gamma*lambda*traceDot;
    for(const [key,value] of Object.entries(previous))
      traces[key]=(traces[key]??0)+dutch*value;
    for(const [key,e] of Object.entries(traces))
      model.weights[key]=(model.weights[key]??0)+
        alpha*(delta+correction)*e;
    for(const [key,value] of Object.entries(previous))
      model.weights[key]=(model.weights[key]??0)-alpha*correction*value;
    // qOld = Q(s', a') BEFORE the above weight update (per true-online SARSA).
    qOld=terminal?0:nextValue;
    validatePureRlModel(model);
    updates++;
  }
  return {
    select(view){
      if(finished||proposed)throw new Error('Pure RL session not ready to select');
      const candidates=ranker(model,view);
      if(!candidates.length)throw new Error('No legal action for pure RL seat');
      let best=0;
      for(let i=1;i<candidates.length;i++)
        if(candidates[i].q>candidates[best].q+1e-12)best=i;
      const exploratory=epsilon>0&&randomInteger(rng,1000000)<epsilon*1000000;
      const index=exploratory?randomInteger(rng,candidates.length):best;
      proposed=candidates[index];
      return Array.isArray(proposed.action)?[...proposed.action]:{...proposed.action};
    },
    commit(){
      if(!proposed||finished)throw new Error('No accepted pure RL action');
      // Q(next) is evaluated against the CURRENT weights before updating.
      update(proposed.features,0,proposed.q,false);
      previous=proposed.features;
      proposed=null;decisions++;
    },
    finish(reward){
      if(finished||proposed||![-1,1].includes(reward))
        throw new Error('Match must finish after an accepted action');
      update(null,reward,0,true);
      finished=true;
      return {decisions,updates,train};
    },
    stats(){return {decisions,updates,train,finished};}
  };
}
