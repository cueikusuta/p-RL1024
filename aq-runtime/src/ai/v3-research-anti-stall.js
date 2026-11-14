// I877: bounded research-only anti-stall intervention.
// Operates ONLY on already authorized, seat-local public candidate vectors.
// An intervention is a legal engine-submitted move, NEVER a sampled action
// for Policy-gradient or contextual-bandit Q training. No private info.
export const I877_ACTION_FEATURE_CONTRACT='i877-legal-action-public-v1';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
export function createI877AntiStall({maxNonAttacks=5}={}){
  if(!Number.isInteger(maxNonAttacks)||maxNonAttacks<2||
     maxNonAttacks>32)
    throw new TypeError('Invalid anti-stall threshold');
  let nonAttacks=0,fallbacks=0,decisions=0;
  const isAttack=x=>Array.isArray(x)&&x.length>=23&&
    x[13]===1&&x[14]===0&&x[16]>0&&x[19]===0;
  function choose(candidates,sampledIndex){
    if(!Array.isArray(candidates)||!candidates.length||
       !Number.isInteger(sampledIndex)||sampledIndex<0||
       sampledIndex>=candidates.length||
       candidates.some(x=>!Array.isArray(x)||x.length<23||
         x.some(v=>!finite(v))))
      throw new TypeError('Invalid public legal candidate');
    decisions++;
    if(isAttack(candidates[sampledIndex])){
      nonAttacks=0;
      return {index:sampledIndex,forcedFallback:false};
    }
    nonAttacks++;
    if(nonAttacks<maxNonAttacks)
      return {index:sampledIndex,forcedFallback:false};
    const ranked=candidates.map((x,i)=>({
      i,
      score:isAttack(x)?
        x[16]*Math.max(0.1,x[17])*
          (x[22]===1?x[21]:0.5):Number.NEGATIVE_INFINITY,
    })).filter(x=>Number.isFinite(x.score))
      .sort((a,b)=>b.score-a.score||a.i-b.i);
    if(!ranked.length)
      // There is no legal damaging move in the policy's authorized
      // public candidate set. Never invent one; simulator fail-closed
      // still protects against true nontermination.
      return {index:sampledIndex,forcedFallback:false};
    nonAttacks=0;fallbacks++;
    return {index:ranked[0].i,forcedFallback:true};
  }
  return {
    choose,
    stats:()=>({decisions,forcedFallbacks:fallbacks,
      consecutiveNonAttackChoices:nonAttacks}),
  };
}
