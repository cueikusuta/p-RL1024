// Opt-in, seat-scoped AI composition for an authoritative four-win Match.
// The handoff contains accepted own decisions and opponent PUBLIC observations;
// it never receives an opponent Match checkpoint, hidden roster, or true set.
export const PHASE_PIPELINE_SCHEMA='p-rl1024-phase-pipeline-v1';
const STAGES=['draft','selection','battle'];
const hasOwn=(object,key)=>Object.prototype.hasOwnProperty.call(object,key);
const isObject=value=>value!==null&&typeof value==='object'&&!Array.isArray(value);
const clone=value=>structuredClone(value);
function immutable(value) {
  if(value&&typeof value==='object'){
    Object.freeze(value);
    for(const child of Object.values(value))immutable(child);
  }
  return value;
}
function validateHandoff(value) {
  if(value===undefined)return null;
  if(!isObject(value))throw new TypeError('Handoff must be a JSON object');
  let serialized;
  try{serialized=JSON.stringify(value);}catch{throw new TypeError('Invalid handoff JSON');}
  if(typeof serialized!=='string'||serialized.length>4096)
    throw new TypeError('Handoff exceeds 4096 characters');
  const data=JSON.parse(serialized);
  const visit=(v,depth=0)=>{
    if(depth>12)throw new TypeError('Handoff nesting exceeds limit');
    if(v&&typeof v==='object')for(const [key,entry] of Object.entries(v)){
      if(['__proto__','constructor','prototype'].includes(key))
        throw new TypeError('Unsafe handoff key');
      visit(entry,depth+1);
    }
  };
  visit(data);
  return data;
}
function normalize(stage,value){
  const agent=typeof value==='function'?{id:stage+':custom',select:value}:value;
  if(!isObject(agent)||typeof agent.id!=='string'||
    !/^[a-z0-9][a-z0-9._:-]{0,63}$/i.test(agent.id)||
    typeof agent.select!=='function'||
    (agent.accepted!==undefined&&typeof agent.accepted!=='function')||
    (agent.finish!==undefined&&typeof agent.finish!=='function'))
    throw new TypeError('Invalid '+stage+' agent');
  return agent;
}
const ownSet=p=>Object.fromEntries(['instanceId','setId','speciesId',
  'speciesClauseId','itemId','abilityId','moves','types','cost','stats','mega']
  .filter(key=>hasOwn(p,key)).map(key=>[key,clone(p[key])]));
function publicFoe(view){
  const foe=view?.opponent;
  if(!isObject(foe))return null;
  // Do not forward arbitrary future engine fields into cross-AI memory.
  return Object.fromEntries(['speciesId','types','hpPercent','status','remaining',
    'stages','publicObservation']
    .filter(key=>hasOwn(foe,key)).map(key=>[key,clone(foe[key])]));
}
const cleanAction=action=>clone(action);

// Each phase handler receives {view,context,rng,seat,round}, returning either
// a normal action or {action,handoff}. The handoff is published only AFTER the
// authoritative Match accepts that action. Handlers may be different models.
export function createPhasePipeline(agents,{seat}={}){
  if(![0,1].includes(seat)||!isObject(agents)||
    STAGES.some(stage=>!hasOwn(agents,stage)))
    throw new TypeError('Expected three phase agents and own seat');
  const handlers=Object.fromEntries(STAGES.map(stage=>[stage,normalize(stage,agents[stage])]));
  const ids=Object.freeze(Object.fromEntries(STAGES.map(stage=>[stage,handlers[stage].id])));
  const notes={draft:null,selection:null,battle:null};
  const completed=[];
  let current=null,pending=null,finished=false;
  function snapshot(){
    return immutable(clone({schema:PHASE_PIPELINE_SCHEMA,seat,
      round:current?.round??null,agents:ids,
      current,completed,handoff:notes}));
  }
  function select(stage,view,{round,rng=null}={}){
    if(finished||pending||!STAGES.includes(stage)||
      !Number.isInteger(round)||round<1||round>64||!isObject(view))
      throw new TypeError('Invalid phase selection');
    if(stage==='draft'){
      if(view.phase!=='draft'||current||
        (completed.length&&round<=completed.at(-1).round))
        throw new Error('Draft out of order');
      current={round,draft:null,selection:null,
        battle:{decisions:0,lastAction:null,lastPublicOpponent:null},result:null};
    }else if(!current||current.round!==round||!current.draft||
      (stage==='selection'&&(view.phase!=='selection'||current.selection))||
      (stage==='battle'&&(!current.selection||!view.requestType)))
      throw new Error('Phase out of order');
    const raw=handlers[stage].select({view:immutable(clone(view)),
      context:snapshot(),rng,seat,round});
    if(raw&&typeof raw.then==='function')throw new TypeError('Asynchronous AI unsupported');
    const enveloped=isObject(raw)&&hasOwn(raw,'action');
    const action=enveloped?raw.action:raw;
    if(stage==='battle'?(!isObject(action)||typeof action.kind!=='string'):
      (!Array.isArray(action)||!action.every(id=>typeof id==='string')))
      throw new TypeError('Invalid '+stage+' decision');
    const handoff=enveloped?validateHandoff(raw.handoff):null;
    pending={stage,round,action:cleanAction(action),handoff};
    return action;
  }
  function accepted(stage,view,action,authorityReceipt=null){
    if(!pending||pending.stage!==stage||!current||
      current.round!==pending.round||
      JSON.stringify(action)!==JSON.stringify(pending.action))
      throw new Error('No matching accepted proposal');
    const ids=clone(pending.action);
    if(stage==='draft'){
      const pool=[...(view.self?.carry??[]),...(view.self?.fresh??[])];
      const byId=new Map(pool.map(p=>[p.instanceId,p]));
      if(ids.length!==6||ids.some(id=>!byId.has(id)))
        throw new TypeError('Invalid accepted draft roster');
      current.draft={ids,roster:ids.map(id=>ownSet(byId.get(id)))};
    }else if(stage==='selection'){
      if(ids.length!==3||ids.some(id=>!current.draft.ids.includes(id)))
        throw new TypeError('Invalid accepted selection');
      current.selection={ids,opponentPublicSpecies:
        clone(Array.isArray(view.opponent?.species)?view.opponent.species:[])};
    }else{
      current.battle.decisions++;
      current.battle.lastAction=ids;
      current.battle.lastPublicOpponent=publicFoe(view);
    }
    if(pending.handoff!==null)notes[stage]=pending.handoff;
    pending=null;
    handlers[stage].accepted?.({view:immutable(clone(view)),action:ids,
      context:snapshot(),seat,round:current.round,
      authorityReceipt});
  }
  function roundEnd(round,winner){
    if(pending||!current||current.round!==round||
      !current.selection||![0,1].includes(winner))
      throw new Error('Invalid round end');
    current.result={winner};
    completed.push(current);
    current=null;
  }
  function finish(result,authorityReceipt=null){
    if(finished||pending||current||!isObject(result))
      throw new Error('Pipeline not ready to finish');
    finished=true;
    return Object.fromEntries(STAGES.map(stage=>[stage,
      handlers[stage].finish?.({result:immutable(clone(result)),
        context:snapshot(),seat,authorityReceipt})??null]));
  }
  return {ids,select,accepted,roundEnd,finish,snapshot};
}
