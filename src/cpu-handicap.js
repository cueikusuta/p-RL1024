import config from './cpu-ascension.json' with {type:'json'};

const fields=Object.keys(config.base);
const levels=[Object.freeze({...config.base})];
for(const step of config.ascensions){
  if(step.level!==levels.length||Object.keys(step.changes).some(key=>!fields.includes(key)))
    throw new Error('Invalid CPU ascension table');
  levels.push(Object.freeze({...levels.at(-1),...step.changes}));
}
if(levels.length!==config.difficultyRange.max+1)
  throw new Error('Incomplete CPU ascension table');

export const CPU_DIFFICULTY_MIN=config.difficultyRange.min;
export const CPU_DIFFICULTY_MAX=config.difficultyRange.max;

// Internal level 0 retains existing PvP/legacy rules; UI exposes 1..5.
// Every later row replaces
// only its changed fields, inheriting all other effective values.
export function resolveCpuHandicap(level=0){
  if(!Number.isInteger(level)||level<0||level>CPU_DIFFICULTY_MAX)
    throw new TypeError('Invalid CPU difficulty');
  return levels[level];
}

// Until a measured per-set strength metric is available, use the catalog's
// curated cost (1..7) as a reproducible and bounded strength proxy.
// This is not a statistical win-rate estimate.
export function cpuSetStrength(set){
  if(!Number.isInteger(set?.cost)||set.cost<1||
      (set.cost>7&&!(set.cost===8&&set.cpuSpecial==='cpu-legendary-v1')))
    throw new TypeError('Invalid catalog cost for CPU weighting');
  return (Math.min(set.cost,7)-1)/6;
}
export function cpuCandidateWeight(set,handicap){
  const score=cpuSetStrength(set);
  return (1+handicap.strongSetBias*score)*
    (set.cost===1?handicap.costOneWeightMultiplier:1);
}
export function cpuHandicapLabels(level){
  const h=resolveCpuHandicap(level);
  return [
    '初期クレジット +'+h.initialCreditBonus,
    '毎ラウンド追加クレジット +'+h.perRoundCreditBonus,
    '通常候補 +'+h.extraNormalCandidates+'体',
    '候補抽選 '+h.candidatePoolDraws+'回',
    '強い型の重み補正 +'+h.strongSetBias,
    'コスト1の抽選重み ×'+h.costOneWeightMultiplier,
    'メガ候補 +'+h.extraMegaCandidates+'体',
    'CPU専用の準伝説・伝説候補 '+h.specialLegendaryCandidates+'体',
  ];
}
