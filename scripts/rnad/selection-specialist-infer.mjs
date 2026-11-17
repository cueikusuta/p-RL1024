// Versioned research-only selection specialist: exact public 96D + 60x48D
// scores, with the 20 unordered trios x 3 legal leads explicitly separated.
// No teacher utilities, opponent hidden information, or champion switch.
import {readFileSync} from 'node:fs';

const triplets=[];
for(let i=0;i<6;i++)for(let j=0;j<6;j++)for(let k=j+1;k<6;k++)
  if(i!==j&&i!==k)triplets.push([i,j,k]);
const keys=[...new Set(triplets.map(t=>[...t].sort().join(',')))].sort((a,b)=>
  a.split(',').map(Number)[0]-b.split(',').map(Number)[0]||
  a.split(',').map(Number)[1]-b.split(',').map(Number)[1]||
  a.split(',').map(Number)[2]-b.split(',').map(Number)[2]);
const teamIndex=new Map(keys.map((x,i)=>[x,i]));
const trioIds=triplets.map(t=>teamIndex.get([...t].sort().join(',')));
const leadIds=triplets.map(t=>t[0]);
const trioRows=keys.map(k=>triplets.map((t,i)=>
  [...t].sort().join(',')===k?i:-1).filter(i=>i>=0));
const leadRows=Array.from({length:6},(_,lead)=>triplets.map((t,i)=>
  t[0]===lead?i:-1).filter(i=>i>=0));
if(triplets.length!==60||keys.length!==20||
  trioRows.some(r=>r.length!==3)||leadRows.some(r=>r.length!==10))
  throw new Error('Selection action enumeration contract changed');

const shapes=structured=>({
  'state.weight':[96,96],'state.bias':[96],
  'action.weight':[96,48],'action.bias':[96],
  ...(structured?{
    'trio.weight':[96,32],'trio.bias':[96],
    'lead.weight':[96,32],'lead.bias':[96],
    'trio_head.weight':[1,96],'trio_head.bias':[1],
    'lead_head.weight':[1,96],'lead_head.bias':[1],
  }:{}),
  'interaction.0.weight':[96,96],'interaction.0.bias':[96],
  'interaction.2.weight':[1,96],'interaction.2.bias':[1],
  'trio_ids':[60],'lead_ids':[60],'trio_rows':[20,3],
  'lead_rows':[6,10],
});
const toFlattened=a=>a.flat(Infinity);
export function validateSelectionSpecialist(payload){
  if(!payload||payload.schema!=='rnad-selection-specialist-96x48-v1'||
     !['flat-hard','structured-hard','flat-mixed','structured-mixed']
       .includes(payload.arm)||
     payload.structured!==payload.arm.startsWith('structured')||
     typeof payload.source_sha!=='string'||
     !/^[0-9a-f]{40}$/.test(payload.source_sha)||
     payload.not_champion!==true||
     payload.training_accepted_teacher_matches!==256||
     payload.training_epochs!==6||
     !payload.weights||typeof payload.weights!=='object')
    throw new TypeError('Untrusted selection specialist model');
  const expected=shapes(payload.structured);
  if(Object.keys(payload.weights).length!==Object.keys(expected).length||
     Object.keys(expected).some(k=>!Object.hasOwn(payload.weights,k)))
    throw new TypeError('Selection model has missing/unexpected weights');
  for(const [name,shape] of Object.entries(expected)){
    const w=payload.weights[name],size=shape.reduce((a,b)=>a*b,1);
    if(!w||JSON.stringify(w.shape)!==JSON.stringify(shape)||
       !Array.isArray(w.values)||w.values.length!==size||
       w.values.some(v=>typeof v!=='number'||!Number.isFinite(v)))
      throw new TypeError('Invalid selection tensor: '+name);
  }
  for(const [name,expectedArray] of Object.entries({
    'trio_ids':trioIds,'lead_ids':leadIds,
    'trio_rows':toFlattened(trioRows),'lead_rows':toFlattened(leadRows)
  })){
    if(payload.weights[name].values.some((v,i)=>v!==expectedArray[i]))
      throw new TypeError('Selection canonical 60 candidate-order buffer mismatch '+name);
  }
  return payload;
}
function linear(vec,p,w,b){
  const weight=p.weights[w],bias=p.weights[b],rows=weight.shape[0],cols=weight.shape[1];
  if(vec.length!==cols)throw new TypeError('Bad linear input length');
  const out=new Array(rows);
  for(let i=0;i<rows;i++){
    let sum=bias.values[i];
    for(let j=0;j<cols;j++)sum+=weight.values[i*cols+j]*vec[j];
    out[i]=sum;
  }
  return out;
}
const relu=v=>v.map(x=>Math.max(0,x));
const plus=(...xs)=>xs[0].map((_,i)=>xs.reduce((s,x)=>s+x[i],0));
function meanByRows(actions,rows,start,end){
  const inv=1/rows.length;
  const out=Array(end-start).fill(0);
  for(const ix of rows)
    for(let j=0;j<out.length;j++)out[j]+=actions[ix][start+j]*inv;
  return out;
}
function oneScore(p,state,act){
  const hidden=relu(linear(relu(act),p,'interaction.0.weight','interaction.0.bias'));
  return linear(hidden,p,'interaction.2.weight','interaction.2.bias')[0];
}
export function scoreSelectionSpecialist(model, state, candidates){
  const p=validateSelectionSpecialist(model);
  if(!Array.isArray(state)||state.length!==96||
     !Array.isArray(candidates)||candidates.length!==60||
     candidates.some(a=>!Array.isArray(a)||a.length!==48)||
     [...state,...candidates.flat()].some(x=>
       typeof x!=='number'||!Number.isFinite(x)))
    throw new TypeError('Expected public 96D state and exact 60x48D legal selection features');
  const s=relu(linear(state,p,'state.weight','state.bias'));
  const acts=candidates.map(a=>relu(linear(a,p,'action.weight','action.bias')));
  if(!p.structured)
    return acts.map(act=>oneScore(p,s,plus(s,act)));
  const teams=trioRows.map(rows=>relu(linear(
    meanByRows(candidates,rows,0,32),p,'trio.weight','trio.bias')));
  const leads=leadRows.map(rows=>relu(linear(
    meanByRows(candidates,rows,16,48),p,'lead.weight','lead.bias')));
  const logits=new Array(60);
  for(let i=0;i<60;i++){
    const t=teams[trioIds[i]],lead=leads[leadIds[i]];
    const team=linear(relu(plus(s,t)),p,'trio_head.weight','trio_head.bias')[0];
    const head=linear(relu(plus(s,lead)),p,'lead_head.weight','lead_head.bias')[0];
    const pair=oneScore(p,s,plus(s,t,lead,acts[i]));
    logits[i]=team+head+pair;
  }
  if(logits.some(x=>!Number.isFinite(x)))
    throw new Error('Nonfinite selection logits');
  return logits;
}
export function chooseSelectionSpecialist(model,state,legalFeatureVectors){
  const logits=scoreSelectionSpecialist(model,state,legalFeatureVectors);
  let best=0;
  for(let i=1;i<logits.length;i++)if(logits[i]>logits[best])best=i;
  return {index:best,logits};
}
if(process.argv[1]?.endsWith('/selection-specialist-infer.mjs')){
  if(process.argv.length!==6||
     process.argv[2]!=='--model'||process.argv[4]!=='--input')
    throw new Error('Usage: node selection-specialist-infer.mjs --model model.json --input state-actions.json');
  const model=JSON.parse(readFileSync(process.argv[3],'utf8'));
  const inputs=JSON.parse(readFileSync(process.argv[5],'utf8'));
  const logits=scoreSelectionSpecialist(model,inputs.state,inputs.actions);
  process.stdout.write(JSON.stringify({logits})+'\n');
}
