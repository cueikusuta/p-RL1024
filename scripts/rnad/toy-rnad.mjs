// Tabular R-NaD / NeuRD mathematics gate ONLY. No neural learner,
// V-trace, off-policy correction or Pokemon self-play is implemented here.
// Based on Perolat et al. (2022): regularized payoff, inner replicator
// dynamics in logit coordinates, periodic magnet/reference update.
const finite=n=>typeof n==='number'&&Number.isFinite(n);
function requireValue(test,message){if(!test)throw new TypeError(message);}
export function softmax(logits){
  requireValue(Array.isArray(logits)&&logits.length>=2&&logits.every(finite),
    'Expected finite action logits');
  const max=Math.max(...logits);
  const numerators=logits.map(x=>Math.exp(x-max));
  const total=numerators.reduce((x,y)=>x+y,0);
  requireValue(finite(total)&&total>0,'Unstable softmax');
  return numerators.map(x=>x/total);
}
const expected=(xs,p)=>xs.reduce((sum,x,i)=>sum+x*p[i],0);
const center=(xs,p)=>xs.map(x=>x-expected(xs,p));
function validateGames(games,priors){
  requireValue(Array.isArray(games)&&games.length>=1&&
    Array.isArray(priors)&&priors.length===games.length&&
    priors.every(p=>finite(p)&&p>0)&&
    Math.abs(priors.reduce((a,b)=>a+b,0)-1)<1e-9,
    'Expected private-type prior');
  const A=games[0]?.length,B=games[0]?.[0]?.length;
  requireValue(Number.isInteger(A)&&A>=2&&Number.isInteger(B)&&B>=2&&
    games.every(matrix=>matrix.length===A&&matrix.every(row=>
      Array.isArray(row)&&row.length===B&&row.every(finite))),
  'Expected consistent finite zero-sum payoff matrices');
  return {A,B,T:games.length};
}
// R0 = u - eta log(p0/magnet0) + eta log(p1/magnet1).
// R1 = -R0; this MUST remain zero-sum for both seats.
export function regularizedZeroSumPayoff({
  payoff,eta,actor0,actor1,magnet0,magnet1
}){
  requireValue([payoff,eta,actor0,actor1,magnet0,magnet1].every(finite)
    &&eta>0&&Math.min(actor0,actor1,magnet0,magnet1)>0,
    'Invalid regularized payoff probabilities');
  const adjusted=payoff-eta*Math.log(actor0/magnet0)+
    eta*Math.log(actor1/magnet1);
  return [adjusted,-adjusted];
}
export function exploitability(games,priors,p0,p1){
  const {A,B,T}=validateGames(games,priors);
  requireValue(p0.length===T&&p0.every(p=>p.length===A)&&p1.length===B,
    'Policy has invalid action dimensions');
  const ownBr=games.reduce((sum,matrix,t)=>sum+
    priors[t]*Math.max(...Array.from({length:A},(_,a)=>
      expected(matrix[a],p1))),0);
  const opponentBr=Math.min(...Array.from({length:B},(_,b)=>
    games.reduce((sum,matrix,t)=>sum+priors[t]*
      p0[t].reduce((v,pa,a)=>v+pa*matrix[a][b],0),0)));
  return ownBr-opponentBr;
}
function initSkew(n,mass){
  return Array.from({length:n},(_,a)=>Math.log(a===0?mass:(1-mass)/(n-1)));
}
export function runTabularRnad({
  games,priors,eta=.2,step=.1,innerSteps=350,outerSteps=10,
  skew0=.94,skew1=.9
}={}){
  const {A,B,T}=validateGames(games,priors);
  requireValue([eta,step,skew0,skew1].every(finite)&&eta>0&&step>0&&
    step<=1&&skew0>0&&skew0<1&&skew1>0&&skew1<1&&
    Number.isInteger(innerSteps)&&innerSteps>=1&&innerSteps<=20000&&
    Number.isInteger(outerSteps)&&outerSteps>=1&&outerSteps<=100,
    'Invalid bounded R-NaD toy parameters');
  let logits0=Array.from({length:T},()=>initSkew(A,skew0));
  let logits1=initSkew(B,skew1);
  let magnet0=logits0.map(softmax),magnet1=softmax(logits1);
  const initial=exploitability(games,priors,magnet0,magnet1);
  const history=[initial];
  for(let epoch=0;epoch<outerSteps;epoch++){
    // Reference policies stay frozen throughout EVERY inner epoch.
    for(let n=0;n<innerSteps;n++){
      const p0=logits0.map(softmax),p1=softmax(logits1);
      const q0=games.map((matrix,t)=>Array.from({length:A},(_,a)=>
        expected(matrix[a],p1)-eta*Math.log(p0[t][a]/magnet0[t][a])));
      const q1=Array.from({length:B},(_,b)=>-
        games.reduce((sum,matrix,t)=>sum+priors[t]*
          p0[t].reduce((total,prob,a)=>total+prob*matrix[a][b],0),0)
          -eta*Math.log(p1[b]/magnet1[b]));
      // NeuRD follows logit-space replicator dynamics, NOT standard PPO.
      const grad0=q0.map((values,t)=>center(values,p0[t]));
      const grad1=center(q1,p1);
      logits0=logits0.map((row,t)=>row.map((v,a)=>v+step*grad0[t][a]));
      logits1=logits1.map((v,b)=>v+step*grad1[b]);
      requireValue(logits0.every(row=>row.every(finite))&&logits1.every(finite),
        'Nonfinite toy NeuRD logit');
    }
    magnet0=logits0.map(softmax);
    magnet1=softmax(logits1);
    history.push(exploitability(games,priors,magnet0,magnet1));
  }
  return {schema:'rnad-tabular-neurd-gate-v1',
    initialExploitability:initial,finalExploitability:history.at(-1),
    magnetUpdates:outerSteps,innerUpdates:outerSteps*innerSteps,
    history,policy0:magnet0,policy1:magnet1};
}
