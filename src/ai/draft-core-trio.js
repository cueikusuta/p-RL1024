// I1145 research draft policy "Core-Trio" (six-member chooseTeam only).
// Not registered in src/cpu-models.js or CPU_CONTROLLERS. Research and
// evaluation runners compose it with the unchanged Tactical selection and
// AQ-Base battle controllers.
//
// Input boundary: the own Match.view() offer (self.fresh, self.carry,
// self.credits, round, score) plus authoritative battle data and the public
// catalog. Opponent fields of the view are never read: at draft time the
// opponent's offer, team and species are unknown.
//
// Idea: only three of the six members fight, and the Tactical selector (not
// this module) picks them. So a roster is valued by
//   (1) the trio the Tactical selector is predicted to field, under several
//       hypothetical public opponent rosters drawn from the public catalog,
//   (2) how well that trio beats a broad reference population of catalog sets
//       in an approximate 1-vs-1 damage race (direct KO power is not
//       discounted; setup/status plans are trusted only partially; hazards,
//       screens, Trick Room and Baton Pass are not credited),
//   (3) the future value of unspent credits and of trio members that may
//       survive and carry over at cost zero.
// The search is an exact DFS over a bounded shortlist with all draft rules
// enforced during enumeration; validateTeam() checks the final roster.
import {validateTeam,speciesKey} from '../draft.js';
import {calculateChampionsStats} from '../champions-stats.js';
import {effectiveness} from '../battle-mechanics.js';
import {setValue} from './selection-basic.js';

// ---------------------------------------------------------------------------
// Tunable weights. Uncalibrated utilities, not probabilities.
// ---------------------------------------------------------------------------
export const CORE_TRIO_WEIGHTS=Object.freeze({
  // 1: predict the trio with the Tactical selector's own scoring (default).
  // 0: research ablation that assumes this module's best trio is fielded.
  selectorAware:1,
  // Trio value: mean over reference opponents of
  //   ANSWER * (best trio member's duel value) + (1-ANSWER) * (trio mean).
  trioAnswerShare:0.5,
  // Weight of the worst-covered reference opponents (robustness tail).
  trioTailShare:0.25,
  trioTailFraction:0.15,
  // Mix of population-wide value and value against the scenario's six
  // hypothetical opponents (rewards the selector's ability to adapt).
  scenarioShare:0.35,
  // Robustness across hypothetical opponent rosters: mean vs worst scenario.
  scenarioWorstShare:0.25,
  // Scale converting trio value (0..1 duel units) to roster utility.
  trioScale:100,
  // Catalog cost as a prior for effects the duel model does not see
  // (abilities, items, utility). Per trio member per cost point.
  costPrior:0.6,
  // Partial trust in multi-turn plans executed by AQ-Base.
  setupTrust:0.35,
  statusTrust:0.3,
  recoveryTrust:0.3,
  // Future value of one unspent credit (times the future-round factor),
  // saturating after creditCap credits.
  creditValue:0.9,
  creditCap:14,
  creditValueBeyondCap:0.2,
  // Fraction of a fielded member's paid cost expected to come back as a free
  // carry-over (survival and reuse probability), times future factor.
  carryReturn:0.25,
  // megaPreference: utility for a roster containing a Mega-capable set.
  megaPreference:4,
  // Small bonus for each kept zero-cost carried member (familiar, free).
  keptCarry:0.5,
});
// Search and model sizes. Bounded to keep a decision in the sub-second range.
export const CORE_TRIO_LIMITS=Object.freeze({
  maxShortlist:22,
  maxReferences:160,
  scenarioCount:8,
});

// Tactical/Basic selection constants mirrored for prediction only (the real
// selection is computed by selection-basic/selection-tactical at play time).
const SELECT_SPREAD=0.16,SELECT_SYNERGY=1.5,SELECT_SHARED=-4.5,
  SELECT_UNSHARED=0.65,SELECT_ONE_MEGA=18,SELECT_MULTI_MEGA=-12;

const TYPE_ITEMS=Object.freeze({
  mysticwater:'Water',charcoal:'Fire',spelltag:'Ghost',hardstone:'Rock',
  sharpbeak:'Flying',blackglasses:'Dark',miracleseed:'Grass',
  nevermeltice:'Ice',fairyfeather:'Fairy',
});
const TYPE_IMMUNE_ABILITIES=Object.freeze({
  levitate:'Ground',eartheater:'Ground',flashfire:'Fire',
  wellbakedbody:'Fire',waterabsorb:'Water',stormdrain:'Water',
  dryskin:'Water',voltabsorb:'Electric',lightningrod:'Electric',
  motordrive:'Electric',sapsipper:'Grass',
});
const TYPE_CHANGE_ABILITIES=Object.freeze({
  aerilate:'Flying',pixilate:'Fairy',refrigerate:'Ice',galvanize:'Electric',
});
const SLEEP_MOVES=new Set(['spore','sleeppowder','hypnosis','yawn']);

const finite=Number.isFinite;
// Unknown types are treated as neutral instead of aborting the draft.
function safeEffectiveness(type,types,data) {
  try{return effectiveness(type,types,data.typeChart);}catch{return 1;}
}
const cmp=(a,b)=>a<b?-1:a>b?1:0;

// ---------------------------------------------------------------------------
// Combatant model
// ---------------------------------------------------------------------------
function makeForm(set,data,mega) {
  const speciesId=mega?set.mega.speciesId:set.speciesId;
  const species=data.species?.[speciesId];
  let stats=null;
  if(!mega&&set.stats?.hp>0)stats=set.stats;
  else if(species?.baseStats)
    stats=calculateChampionsStats(species.baseStats,set.statPoints??{},
      data.natures?.[set.nature]??{});
  if(!stats)return null;
  const types=(!mega&&set.types)||species?.types||[];
  const abilityId=mega?(set.mega.abilityId??set.abilityId):set.abilityId;
  const moves=(set.moves??[]).map(id=>[id,data.moves?.[id]]).filter(([,m])=>m);
  return {stats,types,abilityId,itemId:set.itemId??null,moves};
}
function attackPower(id,move) {
  if(move.ohko)return 0;
  let bp=move.basePower??0;
  if(!(bp>0)){
    if(move.referenceHandlers?.includes('basePowerCallback'))bp=70;
    else return 0;
  }
  if(id==='knockoff')bp*=1.5;
  if(id==='tripleaxel')return 120;
  if(Array.isArray(move.multihit))bp*=3.1;
  else if(Number.isInteger(move.multihit))bp*=move.multihit;
  if(move.flags?.charge||move.flags?.recharge)bp*=0.5;
  return bp;
}
// Expected fraction of the defender's HP removed by one use of the move.
function moveFraction(att,def,id,move,data,{atkMul=1,defMul=1}={}) {
  if(move.category==='Status')return 0;
  let bp=attackPower(id,move);
  if(!bp)return 0;
  let type=move.type,mods=1;
  const ta=TYPE_CHANGE_ABILITIES[att.abilityId];
  if(ta&&type==='Normal'){type=ta;mods*=1.2;}
  let eff=safeEffectiveness(type,def.types,data);
  if(TYPE_IMMUNE_ABILITIES[def.abilityId]===type&&att.abilityId!=='moldbreaker')eff=0;
  if(type==='Ground'&&def.itemId==='airballoon')eff*=0.5;
  if(def.abilityId==='thickfat'&&(type==='Fire'||type==='Ice'))eff*=0.5;
  if(!eff)return 0;
  const physical=move.category==='Physical';
  let a=physical?att.stats.atk:att.stats.spa;
  if(move.overrideOffensiveStat==='def')a=att.stats.def;
  if(move.overrideOffensivePokemon==='target')a=def.stats.atk;
  if(physical&&['hugepower','purepower'].includes(att.abilityId))a*=2;
  a*=atkMul;
  const d=(physical?def.stats.def:def.stats.spd)*defMul;
  if(att.types.includes(type))mods*=att.abilityId==='adaptability'?2:1.5;
  if(att.itemId==='lifeorb')mods*=1.3;
  if(TYPE_ITEMS[att.itemId]===type)mods*=1.2;
  if(att.itemId==='expertbelt'&&eff>1)mods*=1.2;
  const f=move.flags??{};
  if(att.abilityId==='technician'&&(move.basePower??0)<=60)mods*=1.5;
  if(att.abilityId==='toughclaws'&&f.contact)mods*=1.3;
  if(att.abilityId==='sharpness'&&f.slicing)mods*=1.5;
  if(att.abilityId==='ironfist'&&f.punch)mods*=1.2;
  if(att.abilityId==='strongjaw'&&f.bite)mods*=1.5;
  if(att.abilityId==='megalauncher'&&f.pulse)mods*=1.5;
  if(att.abilityId==='sheerforce'&&(move.secondary||move.secondaries))mods*=1.3;
  if(att.abilityId==='reckless'&&move.recoil)mods*=1.2;
  if(physical&&def.abilityId==='intimidate')mods*=0.8;
  if(def.abilityId==='multiscale')mods*=0.85;
  const acc=move.accuracy===true?1:(move.accuracy??100)/100;
  const damage=(22*bp*a/Math.max(1,d))/50+2;
  return damage*mods*eff*0.925*acc/Math.max(1,def.stats.hp);
}
function bestAttack(att,def,data,opts) {
  let best=0,priority=0;
  for(const [id,move] of att.moves) {
    const x=moveFraction(att,def,id,move,data,opts);
    if(x>best)best=x;
    if((move.priority??0)>0&&id!=='fakeout'&&x>priority)priority=x;
  }
  return {best,priority};
}
const effectiveHp=p=>p.itemId==='leftovers'?1.1:p.itemId==='sitrusberry'?1.2:1;
function hitsToKo(rate,defender,multihit) {
  if(!(rate>0))return Infinity;
  let hits=Math.ceil(effectiveHp(defender)/rate-1e-9);
  if(hits===1&&defender.itemId==='focussash'&&!multihit)hits=2;
  return Math.max(1,hits);
}
// One plan: hits needed and whether this side acts first in each exchange.
function plan(att,def,data,opts={}) {
  const {best,priority}=bestAttack(att,def,data,opts);
  const multihit=att.moves.some(([,m])=>m.multihit);
  const hits=hitsToKo(best,def,multihit)+(opts.delay??0);
  const pri=hitsToKo(priority,def,multihit)+(opts.delay??0);
  return {rate:best,hits,pri};
}
const speedOf=(p,mul=1)=>p.stats.spe*(p.itemId==='choicescarf'?1.5:1)*mul;
// Approximate 1-vs-1 duel value in [0,1] for att against def.
// Win: 0.5 + 0.5*remaining HP. Loss: 0.5*damage dealt.
function duelValue(att,def,data,mine={},theirs={}) {
  const a=plan(att,def,data,mine),d=plan(def,att,data,theirs);
  const sa=speedOf(att,mine.speMul??1),sd=speedOf(def,theirs.speMul??1);
  const first=sa>sd?1:sa<sd?0:0.5;
  const time=(hits,isFirst)=>hits-(isFirst?0.5:0);
  const tA=Math.min(time(a.hits,first===1),time(a.pri,true)-(first===1?0:0.25));
  const tD=Math.min(time(d.hits,first===0),time(d.pri,true)-(first===0?0:0.25));
  if(!finite(tA)&&!finite(tD))return 0.5;
  const rateD=d.rate*(theirs.rateMul??1);
  if(tA<tD||(tA===tD&&first===1)) {
    // Opponent hits landed before our final hit.
    const hitsTaken=Math.max(0,Math.floor(tA));
    const left=Math.max(0,1-hitsTaken*rateD/effectiveHp(att));
    return 0.5+0.5*Math.min(1,left);
  }
  if(tA===tD&&first===0.5)return 0.5;
  const landed=Math.max(0,Math.floor(tD));
  return 0.5*Math.min(1,landed*a.rate/effectiveHp(def));
}
function stageMul(n){return n>=0?(2+n)/2:2/(2-n);}
// Plain duel, plus partially trusted setup/status plans for att only.
function duel(att,def,data,weights) {
  const plain=duelValue(att,def,data);
  let best=plain;
  for(const [id,move] of att.moves) {
    if(move.category!=='Status')continue;
    let alt=null,trust=0;
    const boosts=move.target==='self'?move.boosts:null;
    if(boosts&&(boosts.atk>0||boosts.spa>0||boosts.spe>0)) {
      const physical=att.moves.some(([,m])=>m.category==='Physical'),
        special=att.moves.some(([,m])=>m.category==='Special');
      const atkMul=Math.max(physical?stageMul(boosts.atk??0):1,
        special?stageMul(boosts.spa??0):1);
      alt=duelValue(att,def,data,{atkMul,delay:1,speMul:stageMul(boosts.spe??0)});
      trust=weights.setupTrust;
    } else if(move.status==='brn'&&!def.types.includes('Fire')&&
      def.stats.atk>def.stats.spa) {
      alt=duelValue(att,def,data,{delay:1},{rateMul:0.5});trust=weights.statusTrust;
    } else if(move.status==='par'&&!def.types.includes('Electric')) {
      alt=duelValue(att,def,data,{delay:1},{speMul:0.5});trust=weights.statusTrust;
    } else if(SLEEP_MOVES.has(id)&&!(id!=='yawn'&&def.types.includes('Grass'))) {
      alt=duelValue(att,def,data,{delay:1},{delay:2});trust=weights.statusTrust;
    } else if(move.heal||move.flags?.heal) {
      alt=Math.min(1,plain+0.15);trust=weights.recoveryTrust;
    }
    if(alt!==null&&alt>plain)best=Math.max(best,plain+trust*(alt-plain));
  }
  return best;
}

// ---------------------------------------------------------------------------
// Reference population (public catalog), cached per catalog/data pair.
// ---------------------------------------------------------------------------
const referenceCache=new WeakMap();
function hashText(text) {
  let h=2166136261;
  for(let i=0;i<text.length;i++)h=Math.imul(h^text.charCodeAt(i),16777619);
  h^=h>>>13;h=Math.imul(h,0x5bd1e995);h^=h>>>15;
  return h>>>0;
}
function references(catalog,data,limit) {
  let byData=referenceCache.get(catalog);
  if(!byData){byData=new WeakMap();referenceCache.set(catalog,byData);}
  const cached=byData.get(data);
  if(cached&&cached.limit===limit)return cached;
  const ordered=catalog.filter(s=>s&&typeof s.setId==='string'&&
    data.species?.[s.speciesId]&&!s.cpuSpecial)
    .map(s=>({s,h:hashText(s.setId)}))
    .sort((a,b)=>a.h-b.h||cmp(a.s.setId,b.s.setId)).map(x=>x.s);
  const step=Math.max(1,ordered.length/limit);
  const chosen=[];
  for(let k=0;k<limit&&Math.floor(k*step)<ordered.length;k++)
    chosen.push(ordered[Math.floor(k*step)]);
  const forms=chosen.map(s=>makeForm(s,data,Boolean(s.mega?.speciesId))
    ??makeForm(s,data,false)).filter(Boolean);
  const publicTypes=chosen.map(s=>data.species[s.speciesId]?.types??[]);
  const entry={limit,forms,publicTypes};
  byData.set(data,entry);
  return entry;
}

// ---------------------------------------------------------------------------
// Roster search
// ---------------------------------------------------------------------------
function futureFactor(view) {
  const score=Array.isArray(view.score)?view.score:[0,0];
  const need=Math.min(...[0,1].map(i=>4-(Number.isInteger(score[i])?score[i]:0)));
  if(need<=1)return 0.25;
  if(need===2)return 0.75;
  return 1;
}
function shortlist(pool,{carried,forcedSetId,quality,limit}) {
  if(pool.length<=limit)return pool;
  const keep=new Set();
  const cost=p=>carried.has(p.instanceId)?0:p.cost;
  for(const p of pool)if(carried.has(p.instanceId)||p.setId===forcedSetId||p.cpuSpecial)
    keep.add(p.instanceId);
  // Cheapest non-Mega members guarantee an affordable completion.
  for(const p of pool.filter(p=>!p.mega).sort((a,b)=>cost(a)-cost(b)||
    cmp(a.instanceId,b.instanceId)).slice(0,7))keep.add(p.instanceId);
  for(const p of pool.slice().sort((a,b)=>quality.get(b.instanceId)-
    quality.get(a.instanceId)||cmp(a.instanceId,b.instanceId))) {
    if(keep.size>=limit)break;
    keep.add(p.instanceId);
  }
  return pool.filter(p=>keep.has(p.instanceId));
}

export function chooseCoreTrioTeam(view,{data,catalog=null,forcedSetId=null,
  cpuHandicap=null,megaPreference=false,weights:override=null,
  limits:limitOverride=null,report=null}={}) {
  if(!data?.moves||!data?.typeChart)throw new TypeError('Expected battle definitions');
  const weights=override?{...CORE_TRIO_WEIGHTS,...override}:CORE_TRIO_WEIGHTS;
  const limits=limitOverride?{...CORE_TRIO_LIMITS,...limitOverride}:CORE_TRIO_LIMITS;
  const fresh=view?.self?.fresh??[],carry=view?.self?.carry??[];
  const credits=view?.self?.credits,round=view?.round;
  if(!Number.isSafeInteger(credits)||!Number.isInteger(round))
    throw new TypeError('Expected authorized draft view');
  const carried=new Set(carry.map(p=>p.instanceId));
  const fullPool=[...carry,...fresh].slice()
    .sort((a,b)=>cmp(a.instanceId,b.instanceId));
  if(forcedSetId&&!fullPool.some(p=>p.setId===forcedSetId))
    throw new Error('Forced set absent from public own offer');
  const refs=references(Array.isArray(catalog)&&catalog.length?catalog:fullPool,
    data,limits.maxReferences);
  const R=refs.forms.length;

  // Per-member duel rows against the reference population (base and Mega).
  const rows=new Map();
  for(const p of fullPool) {
    const base=makeForm(p,data,false);
    const mega=p.mega?.speciesId?makeForm(p,data,true):null;
    const row=f=>f?Float64Array.from(refs.forms,r=>duel(f,r,data,weights)):null;
    const b=row(base)??new Float64Array(R).fill(0.25);
    rows.set(p.instanceId,{base:b,mega:row(mega)});
  }
  const mean=a=>a.reduce((n,v)=>n+v,0)/Math.max(1,a.length);
  const quality=new Map(fullPool.map(p=>{
    const r=rows.get(p.instanceId);
    return [p.instanceId,Math.max(mean(r.base),r.mega?mean(r.mega):0)*
      weights.trioScale+p.cost*weights.costPrior];
  }));
  const pool=shortlist(fullPool,{carried,forcedSetId,quality,limit:limits.maxShortlist});
  const n=pool.length;
  const cost=pool.map(p=>carried.has(p.instanceId)?0:p.cost);
  const megaFlag=pool.map(p=>Boolean(p.mega));
  const prior=pool.map(p=>p.cost*weights.costPrior);

  // Hypothetical public opponent rosters (six species types each).
  const S=Math.max(1,Math.min(limits.scenarioCount,Math.floor(R/6)));
  const scenarios=Array.from({length:S},(_,s)=>{
    const idx=[];
    for(let k=0;k<6;k++)idx.push((s*6+k)%R);
    return idx;
  });
  // Selector-side per-member scenario score (Basic formula, megaPreference on).
  const selectorBase=pool.map(p=>setValue(p,data));
  const attacksOf=p=>(p.moves??[]).map(id=>data.moves[id])
    .filter(m=>m&&m.category!=='Status');
  const selScore=pool.map((p,i)=>scenarios.map(idx=>{
    const atk=attacksOf(p);
    const spread=idx.reduce((sum,r)=>{
      const types=refs.publicTypes[r];
      const best=atk.reduce((mx,m)=>{
        if(!m.basePower||!types.length)return mx;
        return Math.max(mx,Math.min(4,safeEffectiveness(m.type,types,data))*
          (p.types?.includes(m.type)?1.35:1)*m.basePower);
      },0);
      return sum+best/6;
    },0);
    return selectorBase[i]+spread*SELECT_SPREAD;
  }));
  const pairSyn=Array.from({length:n},(_,i)=>Float64Array.from({length:n},(_,j)=>{
    const a=pool[i].types??[],b=pool[j].types??[];
    const shared=a.filter(t=>b.includes(t)).length;
    return (shared?SELECT_SHARED*shared:SELECT_UNSHARED)*SELECT_SYNERGY;
  }));

  // Trio values (population and per scenario), computed lazily per trio.
  const trioCache=new Map();
  const tailCount=Math.max(1,Math.round(R*weights.trioTailFraction));
  function trioValue(i,j,k) {
    const key=(i*n+j)*n+k;
    let v=trioCache.get(key);
    if(v)return v;
    const members=[i,j,k].map(x=>rows.get(pool[x].instanceId));
    const options=[members.map(m=>m.base)];
    members.forEach((m,x)=>{
      if(m.mega)options.push(members.map((o,y)=>y===x?o.mega:o.base));
    });
    let best=null;
    for(const opt of options) {
      const answer=new Float64Array(R);
      let total=0;
      for(let r=0;r<R;r++) {
        const a=opt[0][r],b=opt[1][r],c=opt[2][r];
        const value=weights.trioAnswerShare*Math.max(a,b,c)+
          (1-weights.trioAnswerShare)*(a+b+c)/3;
        answer[r]=value;total+=value;
      }
      const sorted=Array.from(answer).sort((x,y)=>x-y);
      let tail=0;for(let t=0;t<tailCount;t++)tail+=sorted[t];
      const pop=(1-weights.trioTailShare)*total/R+weights.trioTailShare*tail/tailCount;
      const perScenario=scenarios.map(idx=>idx.reduce((s,r)=>s+answer[r],0)/idx.length);
      if(!best||pop>best.pop)best={pop,perScenario};
    }
    best.prior=prior[i]+prior[j]+prior[k];
    trioCache.set(key,best);
    return best;
  }

  const future=futureFactor(view);
  const creditUtility=left=>{
    const x=Math.max(0,left);
    return weights.creditValue*future*(Math.min(x,weights.creditCap)+
      weights.creditValueBeyondCap*Math.max(0,x-weights.creditCap));
  };
  const TRIOS=[];
  for(let a=0;a<4;a++)for(let b=a+1;b<5;b++)for(let c=b+1;c<6;c++)TRIOS.push([a,b,c]);
  function evaluate(team,spent,collect=null) {
    let sumValue=0,worst=Infinity,carryBack=0;
    for(let s=0;s<S;s++) {
      // Selector tie-break: first trio in roster order wins (strict >).
      let bestScore=-Infinity,bestT=0;
      for(let t=0;t<20;t++) {
        const [a,b,c]=TRIOS[t];
        const i=team[a],j=team[b],k=team[c];
        const megas=megaFlag[i]+megaFlag[j]+megaFlag[k];
        const score=selScore[i][s]+selScore[j][s]+selScore[k][s]+
          pairSyn[i][j]+pairSyn[i][k]+pairSyn[j][k]+
          (megas===1?SELECT_ONE_MEGA:megas>=2?SELECT_MULTI_MEGA:0);
        if(score>bestScore){bestScore=score;bestT=t;}
      }
      if(!weights.selectorAware) {
        // Ablation: assume the best trio by this module's own value is fielded.
        bestScore=-Infinity;
        for(let t=0;t<20;t++) {
          const [a,b,c]=TRIOS[t],tv=trioValue(team[a],team[b],team[c]);
          const score=tv.pop*weights.trioScale+tv.prior;
          if(score>bestScore){bestScore=score;bestT=t;}
        }
      }
      const [ta,tb,tc]=TRIOS[bestT];
      const i=team[ta],j=team[tb],k=team[tc];
      const tv=trioValue(i,j,k);
      const value=((1-weights.scenarioShare)*tv.pop+
        weights.scenarioShare*tv.perScenario[s])*weights.trioScale+tv.prior;
      sumValue+=value;if(value<worst)worst=value;
      carryBack+=cost[i]+cost[j]+cost[k];
      if(collect)collect.push([i,j,k].map(x=>pool[x].instanceId));
    }
    const trio=(1-weights.scenarioWorstShare)*sumValue/S+
      weights.scenarioWorstShare*worst;
    const left=credits-spent+weights.carryReturn*carryBack/S;
    let total=trio+creditUtility(left);
    if(megaPreference&&team.some(x=>megaFlag[x]))total+=weights.megaPreference;
    total+=weights.keptCarry*team.filter(x=>cost[x]===0&&carried.has(pool[x].instanceId)).length;
    return total;
  }

  // Exact DFS over the shortlist with every draft rule enforced.
  const species=pool.map(speciesKey),items=pool.map(p=>p.itemId||null);
  const forcedIndex=forcedSetId?pool.findIndex(p=>p.setId===forcedSetId):-1;
  let best=null,bestScore=-Infinity;
  const team=[];
  const usedSpecies=new Set(),usedItems=new Set();
  function visit(start,spent,megas) {
    if(team.length===6) {
      if(forcedIndex>=0&&!team.includes(forcedIndex))return;
      const score=evaluate(team,spent);
      const ids=team.map(x=>pool[x].instanceId);
      if(score>bestScore+1e-9||(Math.abs(score-bestScore)<=1e-9&&
        cmp(ids.join('|'),best.join('|'))<0)){best=ids;bestScore=score;}
      return;
    }
    for(let x=start;x<=n-(6-team.length);x++) {
      if(forcedIndex>=0&&x>forcedIndex&&!team.includes(forcedIndex))return;
      if(spent+cost[x]>credits||megas+megaFlag[x]>3)continue;
      if(usedSpecies.has(species[x])||(items[x]&&usedItems.has(items[x])))continue;
      team.push(x);usedSpecies.add(species[x]);if(items[x])usedItems.add(items[x]);
      visit(x+1,spent+cost[x],megas+megaFlag[x]);
      team.pop();usedSpecies.delete(species[x]);if(items[x])usedItems.delete(items[x]);
    }
  }
  visit(0,0,0);
  let fallback=false;
  if(!best){best=fallbackTeam(fullPool,{carried,credits,forcedSetId});fallback=true;}
  if(!best)throw new Error('No legal affordable six-member roster');
  validateTeam(best,fresh,carry,round,credits,{cpuHandicap});
  if(report&&typeof report==='object') {
    // Research diagnostics only; never consumed by game logic.
    report.fallback=fallback;report.score=bestScore;
    report.predictedTrios=[];
    if(!fallback) {
      const idx=best.map(id=>pool.findIndex(p=>p.instanceId===id));
      evaluate(idx,idx.reduce((nn,x)=>nn+cost[x],0),report.predictedTrios);
    }
  }
  return best;
}

// Legal-completion fallback over the complete offer: cheapest feasible roster
// (rare: only reachable if the shortlist dropped every affordable completion).
function fallbackTeam(pool,{carried,credits,forcedSetId}) {
  const cost=p=>carried.has(p.instanceId)?0:p.cost;
  const ordered=pool.slice().sort((a,b)=>(b.setId===forcedSetId)-(a.setId===forcedSetId)||
    cost(a)-cost(b)||cmp(a.instanceId,b.instanceId));
  const team=[];let found=null;
  function visit(start,spent,megas) {
    if(found)return;
    if(team.length===6) {
      if(!forcedSetId||team.some(p=>p.setId===forcedSetId))
        found=team.map(p=>p.instanceId);
      return;
    }
    for(let x=start;x<ordered.length;x++) {
      const p=ordered[x];
      if(spent+cost(p)>credits||megas+Boolean(p.mega)>3)continue;
      if(team.some(q=>speciesKey(q)===speciesKey(p)||(p.itemId&&q.itemId===p.itemId)))continue;
      team.push(p);visit(x+1,spent+cost(p),megas+Boolean(p.mega));team.pop();
      if(found)return;
    }
  }
  visit(0,0,0);
  return found;
}
