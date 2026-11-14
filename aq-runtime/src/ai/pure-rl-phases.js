// I975: Exhaustive legal draft and 3-member selection for pure RL.
// No Basic/v2-E team scoring, set value, type matchups or opponent-private data.
import {speciesKey} from '../draft.js';
import {validatePureRlModel,rankPureRlActions} from './pure-rl-linear.js';

const safe=x=>{
  if(typeof x!=='string'||!x||x.length>128)return 'unknown';
  const label=x.normalize('NFKC').toLowerCase();
  if(/^[a-z0-9:._/-]{1,128}$/i.test(label))return label;
  // Public species names may be Japanese; hash into a stable ASCII label.
  let h=2166136261;
  for(let i=0;i<label.length;i++){
    h^=label.charCodeAt(i);
    h=Math.imul(h,16777619);
  }
  return 'unicode-'+(h>>>0).toString(16).padStart(8,'0');
};
const bounded=(x,min,max)=>Math.min(max,Math.max(min,x));
function ownOptions(view){
  if(!view?.self||!Array.isArray(view.self.fresh)||
    !Array.isArray(view.self.carry)||
    !Number.isSafeInteger(view.self.credits)||
    view.self.credits<0)throw new TypeError('Invalid public draft view');
  const pool=[...view.self.carry,...view.self.fresh];
  if(pool.length<6||pool.length>20||new Set(pool.map(p=>p.instanceId)).size!==pool.length)
    throw new TypeError('Invalid bounded offered roster');
  return pool;
}

export function legalPureRlTeams(view){
  if(view?.phase!=='draft')throw new TypeError('Expected public draft view');
  const pool=ownOptions(view);
  const carried=new Set(view.self.carry.map(p=>p.instanceId));
  const choices=[],selected=[],usedSpecies=new Set(),usedItems=new Set();
  function visit(start,cost){
    if(selected.length===6){
      choices.push(selected.map(i=>pool[i].instanceId));
      if(choices.length>60000)throw new Error('Draft action space exceeds research bound');
      return;
    }
    const missing=6-selected.length;
    if(pool.length-start<missing)return;
    for(let i=start;i<=pool.length-missing;i++){
      const p=pool[i];
      const species=speciesKey(p),item=p.itemId??'';
      const price=carried.has(p.instanceId)?0:p.cost;
      // The research control Match has no Lv4 CPU handicap. Mirror the
      // authoritative validateTeam 1..7 cost bound even for carryovers.
      if(typeof species!=='string'||!species||
        typeof item!=='string'||!Number.isInteger(p.cost)||
        p.cost<1||p.cost>7||!Number.isInteger(price)||price<0||
        usedSpecies.has(species)||(item&&usedItems.has(item))||
        cost+price>view.self.credits)continue;
      usedSpecies.add(species);if(item)usedItems.add(item);
      selected.push(i);visit(i+1,cost+price);selected.pop();
      usedSpecies.delete(species);if(item)usedItems.delete(item);
    }
  }
  visit(0,0);
  if(!choices.length)throw new Error('No legal six-member draft options');
  return choices;
}

export function legalPureRlSelections(view){
  if(view?.phase!=='selection'||!Array.isArray(view.self?.team)||
    view.self.team.length!==6)throw new TypeError('Expected six-member selection view');
  const team=view.self.team;
  if(new Set(team.map(p=>p.instanceId)).size!==6)
    throw new TypeError('Duplicate selection team IDs');
  const options=[];
  for(let i=0;i<6;i++)for(let j=0;j<6;j++)for(let k=j+1;k<6;k++)
    if(i!==j&&i!==k)
      options.push([team[i].instanceId,team[j].instanceId,team[k].instanceId]);
  return options;
}

function phaseContext(view,phase,seat){
  if(![0,1].includes(seat))throw new TypeError('Invalid own seat');
  const x=new Map();
  const add=(k,n=1)=>x.set(k,(x.get(k)??0)+n);
  add('bias');
  add('phase:'+phase);
  add('round:'+bounded(view.round??1,1,8));
  add('our-score:'+bounded(view.score?.[seat]??0,0,4));
  add('credits:'+bounded(Math.floor((view.self.credits??0)/5),0,15));
  return {x,add};
}
export function pureRlPhaseFeatures(view,action,{seat=0}={}){
  const phase=view?.phase;
  if(phase!=='draft'&&phase!=='selection')
    throw new TypeError('Invalid pure RL phase');
  const options=phase==='draft'?legalPureRlTeams(view):legalPureRlSelections(view);
  if(!options.some(ids=>JSON.stringify(ids)===JSON.stringify(action)))
    throw new TypeError('Illegal phase action');
  return encodeVerifiedPhase(view,action,seat);
}
function encodeVerifiedPhase(view,action,seat){
  const phase=view.phase,{x,add}=phaseContext(view,phase,seat);
  const roster=phase==='draft'?[...view.self.carry,...view.self.fresh]:view.self.team;
  const byId=new Map(roster.map(p=>[p.instanceId,p]));
  const selected=action.map(id=>byId.get(id));
  if(selected.some(p=>!p))throw new TypeError('Unknown phase instance ID');
  const types=new Map(),items=new Set(),carryIds=new Set(
    (view.self.carry??[]).map(p=>p.instanceId));
  for(const p of selected){
    add(phase+':species:'+safe(p.speciesId));
    for(const type of (p.types??[])){
      const label=safe(type);
      types.set(label,(types.get(label)??0)+1);
    }
    if(p.itemId)items.add(p.itemId);
  }
  add(phase+':unique-types:'+bounded(types.size,0,12));
  for(const [label,count] of types) {
    add(phase+':type:'+label);
    add(phase+':type-count:'+label+':'+bounded(count,1,6));
  }
  if(phase==='draft'){
    const teamCost=selected.reduce((sum,p)=>
      sum+(carryIds.has(p.instanceId)?0:p.cost),0);
    add('draft:cost:'+bounded(Math.floor(teamCost/3),0,20));
    add('draft:carry:'+selected.filter(p=>carryIds.has(p.instanceId)).length);
    add('draft:mega:'+selected.filter(p=>Boolean(p.mega)).length);
    add('draft:unique-items:'+items.size);
  }else{
    const lead=selected[0];
    add('selection:lead:'+safe(lead.speciesId));
    for(const type of (lead.types??[]))
      add('selection:lead-type:'+safe(type));
    for(const species of (Array.isArray(view.opponent?.species)
      ?view.opponent.species:[]))
      add('selection:opponent-public:'+safe(species));
  }
  return Object.fromEntries(x);
}
export function rankPureRlPhaseActions(model,view,{seat=0}={}){
  validatePureRlModel(model);
  const actions=view.phase==='draft'?legalPureRlTeams(view)
    :legalPureRlSelections(view);
  return actions.map(action=>{
    const features=encodeVerifiedPhase(view,action,seat);
    const q=Object.entries(features).reduce((sum,[k,v])=>
      sum+(model.weights[k]??0)*v,0);
    return {action,features,q};
  });
}

/**
 * One full-phase legal pure-RL action ranker shared by SARSA training and
 * frozen screen evaluation. The same accepted self-switch history must be
 * presented to Q(o,a) in both modes; no external heuristic may override it.
 * MC (whose trainer does not keep switch streak yet) passes an empty context.
 */
export function rankPureRlAllPhases(model,view,{seat=0,consecutiveOwnSwitches=0}={}){
  if(view?.phase==='draft'||view?.phase==='selection')
    return rankPureRlPhaseActions(model,view,{seat});
  return rankPureRlActions(model,view,{consecutiveOwnSwitches});
}
