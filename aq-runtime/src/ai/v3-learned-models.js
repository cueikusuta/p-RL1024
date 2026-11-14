// I877 experimental, opt-in, public-observation learning primitives.
// Shared linear Policy scorer and binary opponent intent classifier.
// NO rule/legality authority, hidden opponent sets, or server checkpoints.
import {effectiveness} from '../battle-mechanics.js';
import {rankBasicSelectionCandidates,publicRosterTypes} from './selection-basic.js';
import {extractV3Observation,V3_FEATURE_SCHEMA} from './v3-observation.js';

export const V3_LEARNED_SCHEMAS=Object.freeze({
  selection:'i877-selection-policy-v1',
  humanSelection:'i877-human-preview-selection-v2',
  battle:'i877-battle-policy-v1',
  opponent:'i877-opponent-switch-v1',
  value:V3_FEATURE_SCHEMA,
});
export const V3_TYPES=Object.freeze([
  'Normal','Fire','Water','Electric','Grass','Ice','Fighting','Poison',
  'Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy',
]);
const FEATURE_CONTRACTS=Object.freeze({
  [V3_LEARNED_SCHEMAS.selection]:'i877-trio-public-types-v1',
  [V3_LEARNED_SCHEMAS.humanSelection]:'i877-human-cross-preview-v2',
  [V3_LEARNED_SCHEMAS.battle]:'i877-legal-action-public-v1',
  [V3_LEARNED_SCHEMAS.opponent]:'i877-public-switch-observer-v1',
});
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const clamp=(n,lo=0,hi=1)=>Math.min(hi,Math.max(lo,n));
const ratio=p=>p&&finite(p.hp)&&finite(p.maxHP)&&p.maxHP>0
  ?clamp(p.hp/p.maxHP):1;
const moveIds=p=>(p.moves??[]).map(row=>row?.id??row);
const normStat=(p,stat)=>clamp((p.stats?.[stat]??80)/250);
const strictVector=x=>Array.isArray(x)&&x.length>=2&&x.length<=256&&
  x.every(v=>finite(v)&&Math.abs(v)<=4);
const sameAction=(a,b)=>a?.kind===b?.kind&&a?.index===b?.index&&
  Boolean(a?.mega)===Boolean(b?.mega)&&a?.targetIndex===b?.targetIndex;
const scalar=(p,stat)=>normStat(p,stat);

function safeEffectiveness(moveType,types,data) {
  if(!types?.length||!data?.typeChart)return null;
  // Type chart is the sole authority. Incomplete test fixture falls back
  // to unknown, never a fabricated neutral effectiveness.
  try {return clamp(effectiveness(moveType,types,data.typeChart)/4);}
  catch{return null;}
}
function memberSelectionVector(p,data){
  const types=Array.isArray(p.types)?p.types:[];
  const moves=moveIds(p).map(id=>data.moves?.[id]).filter(Boolean);
  return [
    scalar(p,'hp'),scalar(p,'atk'),scalar(p,'def'),
    scalar(p,'spa'),scalar(p,'spd'),scalar(p,'spe'),
    clamp(moves.filter(m=>m.category==='Status').length/4),
    moves.length?clamp(Math.max(0,...moves.map(m=>m.basePower??0))/200):0,
    ...V3_TYPES.map(type=>Number(types.includes(type))),
  ];
}
function selectionVector(chosen,foeTypes,data) {
  const members=chosen;
  const features=[1,...members.flatMap(p=>memberSelectionVector(p,data))];
  const visible=foeTypes.filter(types=>types.length);
  features.push(...V3_TYPES.map(type=>visible.length
    ?visible.filter(types=>types.includes(type)).length/visible.length:0));
  const pressure=visible.map(types=>Math.max(0,...members.map(member=>
    Math.max(0,...moveIds(member).map(id=>{
      const move=data.moves?.[id];
      if(!move||move.category==='Status'||!(move.basePower>0))return 0;
      const factor=safeEffectiveness(move.type,types,data);
      return factor===null?0:clamp(move.basePower/200)*factor;
    })))));
  features.push(
    visible.length?Math.min(...pressure):0,
    visible.length?pressure.reduce((s,x)=>s+x,0)/pressure.length:0,
    visible.length?Math.max(...pressure):0,
    Number(visible.length===foeTypes.length&&visible.length>0),
  );
  if(!strictVector(features))throw new TypeError('Invalid selection features');
  return features;
}
// Enumerate all Basic-legal 3-member selections and all legal lead orders.
// Candidate IDs stay ephemeral; trained artifacts contain ONLY numeric weights.
export function buildV3SelectionSample(view,chosen,{data,sets,
  forcedSetId=null,forceLead=false,excludedSpeciesKey=null,
  excludedItemId=null}={}) {
  if(!Array.isArray(chosen)||chosen.length!==3||!data?.moves||
     !Array.isArray(sets)||!Array.isArray(view?.self?.team))
    throw new TypeError('Invalid V3 selection sample');
  const legal=rankBasicSelectionCandidates(view,{data,sets,
    forcedSetId,forceLead,excludedSpeciesKey,excludedItemId});
  const byId=new Map(view.self.team.map(p=>[p.instanceId,p]));
  const foeTypes=publicRosterTypes(view.opponent.species,sets,data);
  const candidates=[],keys=new Set();
  let selectedIndex=-1;
  for(const {selection} of legal)for(let lead=0;lead<(forceLead?1:3);lead++){
    const ids=[selection[lead],...selection.filter((_,i)=>i!==lead)];
    const key=ids.join('|');
    if(keys.has(key))continue;
    keys.add(key);
    const members=ids.map(id=>byId.get(id));
    if(members.some(p=>!p))throw new TypeError('Unknown own selected member');
    if(key===chosen.join('|'))selectedIndex=candidates.length;
    candidates.push(selectionVector(members,foeTypes,data));
  }
  if(selectedIndex<0||!candidates.length)
    throw new Error('Chosen selection is absent from legal candidates');
  return {candidates,selectedIndex};
}
export function buildV3BattleSample(view,action,legalActions,{data,
  roundWins=0,roundLosses=0}={}) {
  if(!data?.moves||!Array.isArray(legalActions)||!legalActions.length)
    throw new TypeError('Invalid V3 battle sample');
  const obs=extractV3Observation(view,{roundWins,roundLosses}).features;
  const active=view.self.team[view.self.active];
  const known=Array.isArray(view.opponent.types)&&
    view.opponent.types.length>0;
  const candidates=legalActions.map(candidate=>{
    const isMove=candidate.kind==='move';
    const isSwitch=candidate.kind==='switch';
    const member=isSwitch?view.self.team[candidate.index]:active;
    const id=isMove&&Number.isInteger(candidate.index)
      ?active.moves?.[candidate.index]?.id:null;
    const move=data.moves?.[id];
    const types=Array.isArray(member?.types)?member.types:[];
    const factor=move&&known?safeEffectiveness(
      move.type,view.opponent.types,data):null;
    const features=[...obs,
      Number(isMove),Number(isSwitch),Number(candidate.mega===true),
      isMove?clamp((move?.basePower??0)/200):0,
      isMove?(move?.accuracy===true?1:clamp((move?.accuracy??100)/100)):0,
      isMove?clamp(((move?.priority??0)+7)/14):0,
      Number(isMove&&move?.category==='Status'),
      Number(isMove&&types.includes(move?.type)),
      factor??0,Number(factor!==null),
      ratio(member),scalar(member??{},'spe'),
      ...V3_TYPES.map(type=>Number(move?.type===type)),
      ...V3_TYPES.map(type=>Number(types.includes(type))),
    ];
    if(!strictVector(features))throw new TypeError('Invalid battle features');
    return features;
  });
  const selectedIndex=legalActions.findIndex(row=>sameAction(row,action));
  if(selectedIndex<0)throw new Error('Battle training action not in legal set');
  return {candidates,selectedIndex};
}
// Observed from a seat's OWN authorized view, before the opponent submits.
// The label of what the opponent chose is attached offline; it is never
// available to the inference feature extractor.
export function v3OpponentSwitchFeatures(observerView) {
  if(!observerView?.self?.team||!observerView?.opponent)
    throw new TypeError('Missing observer public view');
  const self=observerView.self;
  const own=self.team[self.active];
  const foe=observerView.opponent;
  const types=Array.isArray(foe.types)?foe.types:[];
  const hazards=observerView.field?.sideEffects?.[1]??[];
  const features=[1,ratio(own),
    clamp((foe.hpPercent??100)/100),
    clamp((foe.remaining??3)/3),
    self.team.filter(p=>p.hp>0).length/Math.max(1,self.team.length),
    clamp((observerView.turn??0)/100),
    Number((observerView.field?.trickRoomTurns??0)>0),
    Number(hazards.some(e=>e.id==='stealthrock')),
    clamp((hazards.find(e=>e.id==='spikes')?.layers??0)/3),
    Number(types.length>0),
    ...V3_TYPES.map(type=>Number(types.includes(type))),
  ];
  if(!strictVector(features))throw new TypeError('Invalid opponent features');
  return features;
}
const logit=(w,x,b=0)=>w.reduce((s,v,i)=>s+v*x[i],b);
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):
  Math.exp(x)/(1+Math.exp(x));
function validateTraining(samples,type) {
  if(!Array.isArray(samples)||samples.length<2||samples.length>20000)
    throw new TypeError('Insufficient or unbounded '+type+' samples');
  const first=type==='ranker'?samples[0]?.candidates?.[0]:
    samples[0]?.features;
  if(!strictVector(first))throw new TypeError('Invalid training vector');
  const width=first.length;
  for(const sample of samples){
    const rows=type==='ranker'?sample.candidates:[sample.features];
    if(!Array.isArray(rows)||!rows.length||rows.length>64||
       rows.some(x=>!strictVector(x)||x.length!==width))
      throw new TypeError('Inconsistent training vectors');
    if(type==='ranker'&&(!Number.isInteger(sample.selectedIndex)||
       sample.selectedIndex<0||sample.selectedIndex>=rows.length))
      throw new TypeError('Invalid legal selection label');
    if(type==='binary'&&![0,1].includes(sample.label))
      throw new TypeError('Invalid binary label');
  }
  return width;
}
const checkConfig=({epochs,learningRate,l2})=>{
  if(!Number.isInteger(epochs)||epochs<1||epochs>100||
     !finite(learningRate)||learningRate<=0||learningRate>1||
     !finite(l2)||l2<0||l2>0.5)throw new TypeError('Invalid learning config');
};
export function fitV3RankPolicy(samples,{schema,epochs=15,
  learningRate=0.08,l2=0.001}={}) {
  if(![V3_LEARNED_SCHEMAS.selection,V3_LEARNED_SCHEMAS.battle,
    V3_LEARNED_SCHEMAS.humanSelection].includes(schema))throw new TypeError('Unknown Policy schema');
  checkConfig({epochs,learningRate,l2});
  const n=validateTraining(samples,'ranker');
  const weights=Array(n).fill(0);
  // Deterministic SGD: no RNG, no hidden-state dependency or external ML.
  for(let pass=0;pass<epochs;pass++)for(const sample of samples){
    const logits=sample.candidates.map(x=>logit(weights,x));
    const largest=Math.max(...logits);
    const exps=logits.map(v=>Math.exp(v-largest));
    const total=exps.reduce((a,b)=>a+b,0);
    const gradient=Array(n).fill(0);
    for(let j=0;j<sample.candidates.length;j++){
      const error=exps[j]/total-Number(j===sample.selectedIndex);
      for(let i=0;i<n;i++)gradient[i]+=error*sample.candidates[j][i];
    }
    for(let i=0;i<n;i++)
      weights[i]-=learningRate*(gradient[i]+l2*weights[i]);
  }
  return {schema,featureContract:FEATURE_CONTRACTS[schema],featureCount:n,weights};
}
export function rankV3LearnedPolicy(model,candidates) {
  if(!model||![V3_LEARNED_SCHEMAS.selection,V3_LEARNED_SCHEMAS.battle,
    V3_LEARNED_SCHEMAS.humanSelection].includes(model.schema)||
     model.featureContract!==FEATURE_CONTRACTS[model.schema]||!Number.isInteger(model.featureCount)||
     model.featureCount<2||model.featureCount>256||
     !Array.isArray(model.weights)||model.weights.length!==model.featureCount||
     model.weights.some(n=>!finite(n)||Math.abs(n)>100)||
     !Array.isArray(candidates)||!candidates.length||candidates.length>64||
     candidates.some(x=>!strictVector(x)||x.length!==model.featureCount))
    throw new TypeError('Invalid/unsupported Policy model or candidates');
  const scores=candidates.map(x=>logit(model.weights,x));
  const high=Math.max(...scores);
  const e=scores.map(s=>Math.exp(s-high));
  const norm=e.reduce((a,b)=>a+b,0);
  const probabilities=e.map(x=>x/norm);
  let selectedIndex=0;
  for(let i=1;i<scores.length;i++)
    if(scores[i]>scores[selectedIndex])selectedIndex=i;
  return {selectedIndex,probabilities};
}
export function scoreV3RankPolicy(model,samples) {
  let correct=0,loss=0,first=0;
  for(const sample of samples){
    const p=rankV3LearnedPolicy(model,sample.candidates);
    correct+=Number(p.selectedIndex===sample.selectedIndex);
    first+=Number(sample.selectedIndex===0);
    loss-=Math.log(Math.max(1e-10,p.probabilities[sample.selectedIndex]));
  }
  return {examples:samples.length,accuracy:correct/samples.length,
    firstCandidateBaseline:first/samples.length,
    crossEntropy:loss/samples.length};
}
export function fitV3OpponentSwitch(samples,{epochs=30,
  learningRate=0.1,l2=0.002}={}) {
  checkConfig({epochs,learningRate,l2});
  const n=validateTraining(samples,'binary');
  const weights=Array(n).fill(0);
  let bias=0;
  for(let pass=0;pass<epochs;pass++)for(const {features,label} of samples){
    const error=sigmoid(logit(weights,features,bias))-label;
    bias-=learningRate*error;
    for(let i=0;i<n;i++)
      weights[i]-=learningRate*(error*features[i]+l2*weights[i]);
  }
  return {schema:V3_LEARNED_SCHEMAS.opponent,
    featureContract:FEATURE_CONTRACTS[V3_LEARNED_SCHEMAS.opponent],
    featureCount:n,weights,bias};
}
export function predictV3OpponentSwitch(model,features) {
  if(!model||model.schema!==V3_LEARNED_SCHEMAS.opponent||
     model.featureContract!==FEATURE_CONTRACTS[model.schema]||
     !Number.isInteger(model.featureCount)||model.featureCount<2||
     model.featureCount>256||!Array.isArray(model.weights)||
     model.weights.length!==model.featureCount||
     model.weights.some(n=>!finite(n)||Math.abs(n)>100)||
     !finite(model.bias)||Math.abs(model.bias)>100||
     !strictVector(features)||features.length!==model.featureCount)
    throw new TypeError('Invalid opponent switch model/schema/features');
  return clamp(sigmoid(logit(model.weights,features,model.bias)),
    1e-7,1-1e-7);
}
export function scoreV3OpponentSwitch(model,samples) {
  let brier=0,logLoss=0,positives=0;
  for(const {features,label} of samples){
    const p=predictV3OpponentSwitch(model,features);
    brier+=(p-label)**2;
    logLoss-=label*Math.log(p)+(1-label)*Math.log(1-p);
    positives+=label;
  }
  return {examples:samples.length,
    brier:brier/samples.length,logLoss:logLoss/samples.length,
    positiveRate:positives/samples.length};
}
