// I877 A: versioned, numeric-only observations for offline diagnosis and Value.
// Input MUST be the acting seat's authorized decisionViewForAI and PUBLIC
// matchup. Never accept server checkpoints, true opponent teams or future logs.
export const V3_FEATURE_SCHEMA='i877-public-observation-v2';
export const V3_FEATURE_NAMES=Object.freeze([
  'ownActiveHp','ownMeanAliveHp','ownAliveFraction',
  'foeActiveHp','foeRemainingFraction','turnFraction',
  'attackSignal','breakthroughRisk','switchInSignal',
  'roundWinsFraction','roundLossesFraction','trickRoom','knownMoveForecast',
]);
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const bounded=(v,min=0,max=1)=>Math.max(min,Math.min(max,v));
const ratio=p=>p&&finite(p.hp)&&finite(p.maxHP)&&p.maxHP>0
  ?bounded(p.hp/p.maxHP):0;
function signal(assessment,key) {
  const row=assessment?.modelVersion==='matchup-p1-v1'?assessment[key]:null;
  return row?.status==='evaluated'&&finite(row.score)
    ?bounded(row.score,-1,1):0;
}
export function extractV3Observation(view,{matchup=null,roundWins=0,roundLosses=0}={}) {
  if(!view?.self?.team||!Array.isArray(view.self.team)||
     !view.opponent||!Array.isArray(view.self.team)||
     !Number.isInteger(view.self.active))
    throw new TypeError('V3 requires a seat-local public decision view');
  const team=view.self.team;
  const active=team[view.self.active];
  if(!active)throw new TypeError('V3 active slot is invalid');
  const living=team.filter(p=>finite(p.hp)&&p.hp>0);
  const forecasts=matchup?.modelVersion==='matchup-p1-v1'
    ?matchup.moveEstimates:undefined;
  const features=[
    ratio(active),
    living.length?living.reduce((n,p)=>n+ratio(p),0)/living.length:0,
    bounded(living.length/Math.max(1,team.length)),
    bounded((finite(view.opponent.hpPercent)?view.opponent.hpPercent:100)/100),
    bounded((finite(view.opponent.remaining)?view.opponent.remaining:3)/3),
    bounded((finite(view.turn)?view.turn:0)/100),
    signal(matchup,'attack'),
    signal(matchup,'breakthroughRisk'),
    signal(matchup,'switchIn'),
    bounded((Number.isInteger(roundWins)?roundWins:0)/4),
    bounded((Number.isInteger(roundLosses)?roundLosses:0)/4),
    (view.field?.trickRoomTurns??0)>0?1:0,
    Array.isArray(forecasts)&&forecasts.some(f=>finite(f.expectedPercent))?1:0,
  ];
  return Object.freeze({schema:V3_FEATURE_SCHEMA,
    features:Object.freeze(features)});
}
export function validateV3Features(features) {
  if(!Array.isArray(features)||features.length!==V3_FEATURE_NAMES.length||
     features.some((v,i)=>!finite(v)||v<(i===6||i===7||i===8?-1:0)||
       v>1))throw new TypeError('Incompatible public V3 feature vector');
  return features;
}
// Return only a synthetic projection, never a modified Battle or a real
// opposing hidden state. The unknown portion remains at the observed value.
export function projectV3Observation(features,{ownDamage=0,foeDamage=0}={}) {
  validateV3Features(features);
  if(!finite(ownDamage)||!finite(foeDamage)||ownDamage<0||foeDamage<0)
    throw new TypeError('Invalid public damage forecast');
  const projected=[...features];
  projected[0]=bounded(projected[0]-ownDamage/100);
  projected[3]=bounded(projected[3]-foeDamage/100);
  return projected;
}

// Action choice is an index in the acting seat's own authorized candidates,
// never an opponent-hidden identifier. Strip all unknown or private fields.
function copyPublicAction(action) {
  if(!action||!['move','switch'].includes(action.kind)||
     !(Number.isInteger(action.index)&&action.index>=0&&action.index<6||
       action.kind==='move'&&action.index==='struggle'))
    throw new TypeError('V3 decision must reference an authorized action');
  const copied={kind:action.kind,index:action.index};
  if(action.mega===true)copied.mega=true;
  if(action.targetIndex!==undefined){
    if(!Number.isInteger(action.targetIndex)||action.targetIndex<0||
       action.targetIndex>5)
      throw new TypeError('Invalid public action target');
    copied.targetIndex=action.targetIndex;
  }
  return copied;
}
export const V3_DECISION_SCHEMA='i877-public-decision-v1';
export function createV3DecisionRecord({seat,features,legalActions,selectedAction}) {
  if(![0,1].includes(seat)||!Array.isArray(legalActions)||
     !legalActions.length||legalActions.length>32)
    throw new TypeError('Invalid public V3 decision candidate set');
  validateV3Features(features);
  const legal=legalActions.map(copyPublicAction);
  const selected=copyPublicAction(selectedAction);
  const keys=legal.map(x=>JSON.stringify(x));
  if(new Set(keys).size!==legal.length||
     !keys.includes(JSON.stringify(selected)))
    throw new TypeError('V3 choice is not a candidate in seat-local legal set');
  return {schema:V3_DECISION_SCHEMA,seat,features:[...features],
    legalActions:legal,selectedAction:selected};
}
