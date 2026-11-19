// RNAD I5: prospective seed-cluster paired inference; no legacy evidence rewrite.
// One observation is the whole seed block (both seats), never a single Match.
// Familywise alpha defaults to the 2026-11-17 prospective 80% policy.
const assert=(value,message)=>{if(!value)throw new TypeError('RNAD paired stats: '+message);};
const LAN=[.99999999999980993,676.5203681218851,-1259.1392167224028,
  771.32342877765313,-176.6150291621406,12.507343278686905,
  -.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];
function logGamma(z){
  if(z<.5)return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*z))-logGamma(1-z);
  z-=1;
  let x=LAN[0];
  for(let i=1;i<LAN.length;i++)x+=LAN[i]/(z+i);
  const t=z+LAN.length-1.5;
  return .5*Math.log(2*Math.PI)+(z+.5)*Math.log(t)-t+Math.log(x);
}
function betaCF(a,b,x){
  const FPMIN=1e-300,eps=3e-14;
  let c=1,d=1-(a+b)*x/(a+1);
  if(Math.abs(d)<FPMIN)d=FPMIN;
  d=1/d;
  let h=d;
  for(let m=1;m<=300;m++){
    let aa=m*(b-m)*x/((a+2*m-1)*(a+2*m));
    d=1+aa*d;if(Math.abs(d)<FPMIN)d=FPMIN;
    c=1+aa/c;if(Math.abs(c)<FPMIN)c=FPMIN;
    d=1/d;h*=d*c;
    aa=-(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1));
    d=1+aa*d;if(Math.abs(d)<FPMIN)d=FPMIN;
    c=1+aa/c;if(Math.abs(c)<FPMIN)c=FPMIN;
    d=1/d;
    const delta=d*c;h*=delta;
    if(Math.abs(delta-1)<eps)return h;
  }
  throw new RangeError('RNAD beta continued fraction did not converge');
}
function regBeta(x,a,b){
  if(x<=0)return 0;
  if(x>=1)return 1;
  const front=Math.exp(logGamma(a+b)-logGamma(a)-logGamma(b)+
    a*Math.log(x)+b*Math.log1p(-x));
  if(x<(a+1)/(a+b+2))return front*betaCF(a,b,x)/a;
  return 1-front*betaCF(b,a,1-x)/b;
}
export function studentTCdf(t,df){
  assert(Number.isFinite(t)&&Number.isInteger(df)&&df>=1,
    'invalid Student t argument');
  const tail=.5*regBeta(df/(df+t*t),df/2,.5);
  return t>=0?1-tail:tail;
}
export function studentTQuantile(p,df){
  assert(Number.isFinite(p)&&p>.5&&p<1&&
    Number.isInteger(df)&&df>=1,'invalid Student t quantile');
  let low=0,high=1;
  while(studentTCdf(high,df)<p){
    high*=2;
    if(high>1e15)throw new RangeError('Student t quantile overflow');
  }
  for(let i=0;i<64;i++){
    const mid=(low+high)/2;
    if(studentTCdf(mid,df)<p)low=mid;else high=mid;
  }
  return (low+high)/2;
}
export function pairedClusterCI(blockValues,{
  familywiseAlpha=.20,comparisons=1,range=[-1,1],nullValue=0,
}={}){
  assert(Array.isArray(blockValues)&&blockValues.length>=2&&
    blockValues.every(x=>Number.isFinite(x)&&x>=range[0]&&x<=range[1])&&
    Array.isArray(range)&&range.length===2&&
    Number.isFinite(range[0])&&Number.isFinite(range[1])&&
    range[0]<range[1]&&
    Number.isFinite(nullValue)&&nullValue>=range[0]&&nullValue<=range[1],
    'missing, out-of-range or invalid independent seed blocks');
  assert(Number.isFinite(familywiseAlpha)&&familywiseAlpha>0&&
    familywiseAlpha<1&&Number.isInteger(comparisons)&&comparisons>=1,
    'invalid familywise inference plan');
  const n=blockValues.length,mean=blockValues.reduce((x,y)=>x+y,0)/n;
  const variance=blockValues.reduce((v,x)=>v+(x-mean)**2,0)/(n-1);
  const standardDeviation=Math.sqrt(variance),standardError=standardDeviation/Math.sqrt(n);
  const individualAlpha=familywiseAlpha/comparisons;
  const tCritical=studentTQuantile(1-individualAlpha/2,n-1);
  const tStatistic=standardError===0
    ?(mean===nullValue?0:(mean>nullValue?Infinity:-Infinity))
    :(mean-nullValue)/standardError;
  const pValue=mean===nullValue?1:
    Number.isFinite(tStatistic)
      ?Math.min(1,2*studentTCdf(-Math.abs(tStatistic),n-1)):0;
  const low=Math.max(range[0],mean-tCritical*standardError);
  const high=Math.min(range[1],mean+tCritical*standardError);
  return {
    independent_seed_blocks:n,estimate:mean,
    sample_sd:standardDeviation,standard_error:standardError,
    null_value:nullValue,student_t_df:n-1,t_statistic:tStatistic,
    unadjusted_two_sided_p:pValue,
    bonferroni_adjusted_two_sided_p:Math.min(1,pValue*comparisons),
    familywise_alpha:familywiseAlpha,comparisons,
    per_comparison_alpha:individualAlpha,
    confidence_level_simultaneous:1-familywiseAlpha,
    method:'two-sided-paired-seed-block-Student-t-Bonferroni',
    confidence_interval:[low,high],
    positive_evidence:low>nullValue,
    negative_evidence:high<nullValue,
    // Student-t is model-based, not a finite-sample distribution-free bound.
    finite_sample_distribution_free:false,
  };
}
export function pairedSignSensitivity(blockValues,{nullValue=0}={}){
  assert(Array.isArray(blockValues)&&blockValues.length>=2&&
    blockValues.every(Number.isFinite)&&Number.isFinite(nullValue),
    'invalid paired sign inputs');
  let positive=0,negative=0,zero=0;
  for(const x of blockValues){
    if(x>nullValue)positive++;else if(x<nullValue)negative++;else zero++;
  }
  const n=positive+negative;
  if(n===0)return {positive,negative,zero,two_sided_exact_p:1,
    method:'exploratory-paired-block-sign-test'};
  const smaller=Math.min(positive,negative);
  let logSum=-Infinity;
  for(let k=0;k<=smaller;k++){
    const term=logGamma(n+1)-logGamma(k+1)-logGamma(n-k+1)-n*Math.LN2;
    const hi=Math.max(logSum,term);
    logSum=hi+Math.log(Math.exp(logSum-hi)+Math.exp(term-hi));
  }
  return {positive,negative,zero,two_sided_exact_p:Math.min(1,2*Math.exp(logSum)),
    method:'exploratory-paired-block-sign-test'};
}
