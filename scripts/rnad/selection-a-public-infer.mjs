// Research-only SHA-pinned JS inference for teacher-decomposed A selection.
// Actor inputs: own six complete sets + six PUBLIC opponent species profiles.
// No teacher utility, opponent hidden set, training labels, or champion switch.
import {createHash,timingSafeEqual} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';

export const A_PUBLIC_JS_SCHEMA='rnad-i5-A-public-actor-js-weights-v1';
const verifiedModels=new WeakSet();
const vocabNames=['species','names','abilities','items','moves','types'];
const triplets=[];
for(let i=0;i<6;i++)for(let j=0;j<6;j++)for(let k=j+1;k<6;k++)
  if(i!==j&&i!==k)triplets.push([i,j,k]);
const trioKeys=[...new Set(triplets.map(t=>[...t].sort((a,b)=>a-b).join(',')))].sort(
  (a,b)=>{const x=a.split(',').map(Number),y=b.split(',').map(Number);
         return x[0]-y[0]||x[1]-y[1]||x[2]-y[2];});
const keys=new Map(trioKeys.map((key,i)=>[key,i]));
const trioIndices=triplets.map(t=>keys.get([...t].sort((a,b)=>a-b).join(',')));
const leadIndices=triplets.map(t=>t[0]);
const trioRows=trioKeys.map(k=>k.split(',').map(Number));

function tensorShapes(sizes){
  return {
    'embed.species.weight':[sizes.species,12],
    'embed.names.weight':[sizes.names,12],
    'embed.abilities.weight':[sizes.abilities,12],
    'embed.items.weight':[sizes.items,12],
    'embed.moves.weight':[sizes.moves,12],
    'embed.types.weight':[sizes.types,8],
    'embed.own_proj.0.weight':[64,72],
    'embed.own_proj.0.bias':[64],
    'embed.foe_proj.0.weight':[64,26],
    'embed.foe_proj.0.bias':[64],
    'pair.weight':[64,64], 'pair.bias':[64],
    'opponent.weight':[64,64], 'opponent.bias':[64],
    'core.0.weight':[96,256], 'core.0.bias':[96],
    'core.2.weight':[1,96], 'core.2.bias':[1],
    'lead.0.weight':[96,192], 'lead.0.bias':[96],
    'lead.2.weight':[1,96], 'lead.2.bias':[1],
    'trio_index':[60], 'lead_index':[60], 'trio_rows':[20,3],
  };
}
const isObj=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
const finite=x=>typeof x==='number'&&Number.isFinite(x);
function verifyPayload(p){
  if(!isObj(p)||p.schema!==A_PUBLIC_JS_SCHEMA||
    !/^[0-9a-f]{64}$/.test(p.source_model_sha256||'')||
    !/^[0-9a-f]{40}$/.test(p.source_sha||'')||
    !/^[0-9a-f]{40,64}$/.test(p.engine_sha||'')||
    p.teacher!=='rnad-aq-public-strategy-selection-v1'||
    p.labels_as_inference_inputs!==false||
    p.champion_promoted!==false)
    throw new TypeError('Untrusted frozen A public inference provenance');
  const v=p.public_actor_vocabulary;
  if(!isObj(v)||v.schema!=='rnad-i5-selection-rich-vocabulary-v1'||
    !/^[0-9a-f]{64}$/.test(v.sha256||'')||
    !isObj(v.sizes)||vocabNames.some(name=>
      !Number.isInteger(v.sizes[name])||v.sizes[name]<2||v.sizes[name]>30000))
    throw new TypeError('Untrusted A public categorical vocab');
  const shapes=tensorShapes(v.sizes),raw=p.tensors;
  if(!isObj(raw)||Object.keys(raw).length!==Object.keys(shapes).length||
    Object.keys(shapes).some(k=>!Object.hasOwn(raw,k)))
    throw new TypeError('Incorrect A frozen tensor names');
  for(const [name,shape] of Object.entries(shapes)){
    const t=raw[name];
    if(!isObj(t)||JSON.stringify(t.shape)!==JSON.stringify(shape)||
      !Array.isArray(t.values)||
      t.values.length!==shape.reduce((a,b)=>a*b,1)||
      t.values.some(x=>!finite(x)))
      throw new TypeError('Bad A tensor '+name);
  }
  for(const [name,expected] of [
      ['trio_index',trioIndices],['lead_index',leadIndices],
      ['trio_rows',trioRows.flat()]]){
    if(raw[name].values.some((x,i)=>x!==expected[i]))
      throw new TypeError('A canonical 60-action buffer mismatch '+name);
  }
  if(!isObj(p.parity_fixture)||!isObj(p.parity_fixture.actor)||
    !Array.isArray(p.parity_fixture.logits)||
    p.parity_fixture.logits.length!==60||
    p.parity_fixture.logits.some(x=>!finite(x)))
    throw new TypeError('No 60-score public PyTorch parity fixture');
  return p;
}
export function loadVerifiedASelection(raw,expectedSha){
  if(!(typeof raw==='string'||Buffer.isBuffer(raw))||
    typeof expectedSha!=='string'||!/^[0-9a-f]{64}$/.test(expectedSha))
    throw new TypeError('Need immutable external model JSON SHA256');
  const digest=createHash('sha256').update(raw).digest();
  if(!timingSafeEqual(digest,Buffer.from(expectedSha,'hex')))
    throw new Error('A selection JSON SHA256 mismatch');
  const payload=verifyPayload(JSON.parse(raw.toString()));
  // A verified artifact must be immutable for the lifetime of an evaluation.
  const freezeDeep=x=>{
    if(x&&typeof x==='object'&&!Object.isFrozen(x)){
      for(const child of Object.values(x))freezeDeep(child);
      Object.freeze(x);
    }
    return x;
  };
  const frozen=freezeDeep(payload);
  verifiedModels.add(frozen);
  return frozen;
}
const relu=x=>x.map(v=>Math.max(0,v));
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const mean=vectors=>{
  const n=vectors.length,result=Array(vectors[0].length).fill(0);
  for(const v of vectors)for(let i=0;i<result.length;i++)result[i]+=v[i]/n;
  return result;
};
const maxima=vectors=>vectors[0].map((v,i)=>
  Math.max(...vectors.map(row=>row[i])));
function embedding(p,kind,index){
  const t=p.tensors['embed.'+kind+'.weight'];
  if(!Number.isInteger(index)||index<0||index>=t.shape[0])
    throw new TypeError('A public categorical index outside bounded vocab');
  return t.values.slice(index*t.shape[1],(index+1)*t.shape[1]);
}
function linear(p,stem,x){
  const w=p.tensors[stem+'.weight'],b=p.tensors[stem+'.bias'];
  if(x.length!==w.shape[1])throw new TypeError('A linear width mismatch '+stem);
  const out=[];
  for(let i=0;i<w.shape[0];i++){
    let value=b.values[i],start=i*w.shape[1];
    for(let j=0;j<w.shape[1];j++)value+=w.values[start+j]*x[j];
    out.push(value);
  }
  return out;
}
function validateActor(p,a){
  const v=p.public_actor_vocabulary;
  if(!isObj(a)||a.schema!=='rnad-i5-selection-public-actor-tensors-v1'||
    a.vocab_sha256!==v.sha256||
    JSON.stringify(a.legal)!==JSON.stringify(triplets))
    throw new TypeError('A actor must contain canonical legal public selection');
  for(const [key,rows,width] of [
      ['own_cats',a.own_cats,9],['foe_cats',a.foe_cats,3],
      ['own_numeric',a.own_numeric,16],['foe_numeric',a.foe_numeric,6]]){
    if(!Array.isArray(rows)||rows.length!==6||
      rows.some(row=>!Array.isArray(row)||row.length!==width||
        row.some(n=>key.endsWith('cats')?!Number.isInteger(n):
          !finite(n)||n<0||n>2)))
      throw new TypeError('Invalid public A actor '+key);
  }
  const cats={own_cats:['species','abilities','items','moves','moves',
                        'moves','moves','types','types'],
              foe_cats:['names','types','types']};
  for(const [field,types] of Object.entries(cats))
    for(const row of a[field])for(let j=0;j<row.length;j++)
      if(row[j]<0||row[j]>=v.sizes[types[j]])
        throw new TypeError('Out-of-vocabulary public A token');
}
function assembleOwn(p,cats,nums){
  const parts=[
    embedding(p,'species',cats[0]),embedding(p,'abilities',cats[1]),
    embedding(p,'items',cats[2]),
    mean(cats.slice(3,7).map(id=>embedding(p,'moves',id))),
    mean(cats.slice(7,9).map(id=>embedding(p,'types',id))),nums
  ];
  return relu(linear(p,'embed.own_proj.0',parts.flat()));
}
function assembleFoe(p,cats,nums){
  return relu(linear(p,'embed.foe_proj.0',[
    ...embedding(p,'names',cats[0]),
    ...mean(cats.slice(1).map(id=>embedding(p,'types',id))),
    ...nums
  ]));
}
export function scoreFrozenASelection(p,actor){
  if(!verifiedModels.has(p))
    throw new TypeError('A actor requires externally SHA-verified immutable model');
  validateActor(p,actor);
  const own=actor.own_cats.map((cats,i)=>
    assembleOwn(p,cats,actor.own_numeric[i]));
  const foe=actor.foe_cats.map((cats,i)=>
    assembleFoe(p,cats,actor.foe_numeric[i]));
  const projectedOwn=own.map(x=>linear(p,'pair',x));
  const projectedFoe=foe.map(x=>linear(p,'opponent',x));
  const enriched=own.map((x,i)=>
    add(x,mean(projectedFoe.map(y=>
      relu(add(projectedOwn[i],y))))));
  const teamMean=mean(enriched),opponents=mean(foe);
  const core=trioRows.map(ids=>{
    const members=ids.map(i=>enriched[i]);
    const input=[...mean(members),...maxima(members),
      ...teamMean,...opponents];
    const hidden=relu(linear(p,'core.0',input));
    return linear(p,'core.2',hidden)[0];
  });
  const lead=enriched.map(x=>{
    const hidden=relu(linear(p,'lead.0',[
      ...x,...teamMean,...opponents]));
    return linear(p,'lead.2',hidden)[0];
  });
  const logits=trioIndices.map((team,i)=>core[team]+lead[leadIndices[i]]);
  if(logits.some(x=>!finite(x)))
    throw new Error('Nonfinite frozen A prediction');
  return logits;
}
// Value features for IntegratedPolicyV1: only own/public embeddings,
 // no teacher supervision or hidden opponent set enters this critic.
export function selectionValueFeatures(p, actor) {
  if (!verifiedModels.has(p)) throw new TypeError('Unverified selection critic');
  validateActor(p,actor);
  const own=actor.own_cats.map((cats,i)=>assembleOwn(p,cats,actor.own_numeric[i]));
  const foe=actor.foe_cats.map((cats,i)=>assembleFoe(p,cats,actor.foe_numeric[i]));
  return [...mean(own),...mean(foe)];
}
export function checkASelectionParity(p,{tolerance=0.00015}={}){
  const fixture=p.parity_fixture;
  const actual=scoreFrozenASelection(p,fixture.actor);
  const maxError=Math.max(...actual.map((x,i)=>
    Math.abs(x-fixture.logits[i])));
  if(maxError>tolerance)
    throw new Error('A Python-JS public inference parity mismatch '+maxError);
  return maxError;
}
export function selectFrozenAAction(p,actor){
  const logits=scoreFrozenASelection(p,actor);
  let best=0;
  for(let i=1;i<logits.length;i++)if(logits[i]>logits[best])best=i;
  return {index:best,logits};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2);
  if(args.length!==4||args[0]!=='--model'||args[2]!=='--sha256')
    throw new Error('Use --model FILE --sha256 externally-pinned-sha');
  const p=loadVerifiedASelection(readFileSync(args[1]),args[3]);
  console.log(JSON.stringify({schema:p.schema,
    model:p.source_model_filename,max_parity_abs_error:checkASelectionParity(p)}));
}
