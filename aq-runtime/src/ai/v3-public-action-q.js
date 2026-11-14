// I877: research-only public action-conditioned Monte Carlo advantage.
// Q(s,a) ≈ V_public(s)+A_public(s,a), trained from accepted action and
// *final* four-win Match result, NOT true counterfactual Q or TD bootstrap.
// Caller supplies public legal candidates and public v2-E numeric scores.
const SCHEMA='i877-public-action-advantage-v1';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const clamp=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));
export const PUBLIC_ACTION_Q_SCHEMA=SCHEMA;
export const PUBLIC_ACTION_Q_GAIN=120;
export const PUBLIC_ACTION_Q_MAX_DELTA=24;
const validFeatures=(xs,n)=>Array.isArray(xs)&&xs.length===n&&
  xs.every(x=>finite(x)&&Math.abs(x)<=4);
export function createPublicActionQ(featureCount) {
  if(!Number.isInteger(featureCount)||featureCount<2||featureCount>256)
    throw new TypeError('Invalid action Q feature count');
  return {schema:SCHEMA,featureCount,weights:Array(featureCount).fill(0),
    label:'selected-action Monte Carlo public Match residual; not causal Q'};
}
export function validatePublicActionQ(model) {
  if(!model||model.schema!==SCHEMA||
     !Number.isInteger(model.featureCount)||model.featureCount<2||
     model.featureCount>256||!Array.isArray(model.weights)||
     model.weights.length!==model.featureCount||
     model.weights.some(w=>!finite(w)||Math.abs(w)>25))
    throw new TypeError('Invalid research action Q schema');
  return model;
}
function center(candidates,model) {
  validatePublicActionQ(model);
  if(!Array.isArray(candidates)||candidates.length<1||
     candidates.length>64||
     candidates.some(x=>!validFeatures(x,model.featureCount)))
    throw new TypeError('Invalid public legal action features');
  const avg=Array(model.featureCount).fill(0);
  for(const row of candidates)for(let k=0;k<avg.length;k++)
    avg[k]+=row[k]/candidates.length;
  return candidates.map(row=>row.map((v,k)=>v-avg[k]));
}
const dot=(w,x)=>w.reduce((sum,a,i)=>sum+a*x[i],0);
export function rankPublicActionQ(model,candidates,baselineScores,{
  enabled=true,gain=PUBLIC_ACTION_Q_GAIN,
  maxDelta=PUBLIC_ACTION_Q_MAX_DELTA,
}={}) {
  const features=center(candidates,model);
  if(!Array.isArray(baselineScores)||
     baselineScores.length!==features.length||
     baselineScores.some(x=>!finite(x)||Math.abs(x)>1e6)||
     baselineScores.some((v,i)=>i>0&&v>baselineScores[i-1]+1e-9)||
     !finite(gain)||gain<0||gain>200||
     !finite(maxDelta)||maxDelta<0||maxDelta>100)
    throw new TypeError('Invalid public ranked baseline action scores');
  const advantages=features.map(x=>dot(model.weights,x));
  const anchor=advantages[0];
  const scores=baselineScores.map((base,i)=>base+(
    enabled?clamp(gain*(advantages[i]-anchor),-maxDelta,maxDelta):0));
  let selectedIndex=0;
  for(let i=1;i<scores.length;i++)
    if(scores[i]>scores[selectedIndex])selectedIndex=i;
  return {selectedIndex,scores,
    advantages,changedVsBaseline:selectedIndex!==0,
    counterfactualAvailable:false,
    notCalibratedProbability:true};
}
function checkedGroups(model,groups,maxGroups=128) {
  validatePublicActionQ(model);
  if(!Array.isArray(groups)||groups.length<1||groups.length>maxGroups)
    throw new TypeError('Missing bounded complete Match Q samples');
  for(const group of groups){
    if(!Array.isArray(group)||!group.length||group.length>3000)
      throw new TypeError('Invalid Q Match group');
    for(const row of group){
      if(!Array.isArray(row.candidates)||!Number.isInteger(row.selectedIndex)||
         row.selectedIndex<0||row.selectedIndex>=row.candidates.length||
         ![0,1].includes(row.outcome)||!finite(row.baselineValue)||
         row.baselineValue<=0||row.baselineValue>=1)
        throw new TypeError('Q training must use accepted action with complete outcome');
      center(row.candidates,model);
    }
  }
}
export function updatePublicActionQ(model,groups,{
  epochs=10,learningRate=.15,l2=.001,clip=3,
}={}) {
  checkedGroups(model,groups);
  if(!Number.isInteger(epochs)||epochs<1||epochs>120||
     !finite(learningRate)||learningRate<=0||learningRate>.5||
     !finite(l2)||l2<0||l2>0.1||
     !finite(clip)||clip<=0||clip>10)
    throw new TypeError('Invalid Q research training hyperparameters');
  const w=[...model.weights];
  // One normalized update per completed seat-Match; no action selected
  // in hindsight and no evaluation/holdout labels used for fitting.
  for(let epoch=0;epoch<epochs;epoch++){
    const grad=Array(w.length).fill(0);
    for(const group of groups){
      const scale=1/(groups.length*group.length);
      for(const sample of group){
        const chosen=center(sample.candidates,model)[sample.selectedIndex];
        const target=sample.outcome-sample.baselineValue;
        const error=dot(w,chosen)-target;
        for(let k=0;k<w.length;k++)
          grad[k]+=error*chosen[k]*scale;
      }
    }
    const norm=Math.sqrt(grad.reduce((sum,x)=>sum+x*x,0));
    const factor=Math.min(1,clip/Math.max(1e-12,norm));
    for(let k=0;k<w.length;k++)
      w[k]-=learningRate*(grad[k]*factor+l2*w[k]);
  }
  return validatePublicActionQ({...model,weights:w});
}
export function scorePublicActionQ(model,groups) {
  checkedGroups(model,groups,512);
  let mse=0;
  let examples=0;
  for(const group of groups){
    let groupMse=0;
    for(const row of group){
      const delta=center(row.candidates,model)[row.selectedIndex];
      const error=dot(model.weights,delta)-
        (row.outcome-row.baselineValue);
      groupMse+=error*error;
      examples++;
    }
    mse+=groupMse/group.length/groups.length;
  }
  return {seatMatches:groups.length,examples,matchBalancedResidualMse:mse,
    causalQIdentified:false};
}
