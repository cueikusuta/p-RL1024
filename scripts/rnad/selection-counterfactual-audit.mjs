// I5 public-only v5 feature sensitivity and possible representational aliases.
// Synthetic one-coordinate perturbations are diagnostic, not legal team samples.
// Never infer teacher action utility or performance from this probe.
import {writeFileSync} from 'node:fs';
import {playableSets} from '../../src/production-runtime.js';
import {projectRnadObservation,rnadLegalActions} from '../../src/ml/legal-action-adapter.js';
import {encodeRnadThreePhaseDecisionV5} from '../../src/ml/rnad-phase-features-v5.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};

const unique=new Map();
for(const p of playableSets){
  if(!p.mega&&p.speciesNameJa&&data.species?.[p.speciesId]?.baseStats&&
     !unique.has(p.speciesNameJa))unique.set(p.speciesNameJa,p);
}
const pool=[...unique.values()];
if(pool.length<30)throw new Error('No public six-on-six selection examples');
const own=pool.slice(0,6).map((p,i)=>({
  ...structuredClone(p),instanceId:'audit-'+i,
  stats:{...data.species[p.speciesId].baseStats},
  baseStats:{...data.species[p.speciesId].baseStats},
  types:[...data.species[p.speciesId].types],
  hp:100,maxHP:100
}));
const shown=pool.slice(8,14).map(p=>p.speciesNameJa);
function encode(team=own,preview=shown,privateMarker='secret-A'){
  const observation=projectRnadObservation({
    phase:'selection',round:1,score:[0,0],
    self:{team,credits:8},
    opponent:{species:preview,privateTeam:[{moves:[privateMarker],itemId:privateMarker}]}
  },{seat:0});
  const legal=rnadLegalActions(observation);
  return encodeRnadThreePhaseDecisionV5(observation,legal);
}
const reference=JSON.stringify(encode());
const variants = (values,field,apply)=>{
  const out={total:0,distinguished:0,indistinguishable:0,examples:[]};
  for(const value of values){
    if(value===field)continue;
    const team=structuredClone(own);
    apply(team,value);
    const same=JSON.stringify(encode(team))===reference;
    out.total++;
    out[same?'indistinguishable':'distinguished']++;
    if(same&&out.examples.length<5)out.examples.push(value);
  }
  return out;
};
const ids=(values,field)=>[...new Set(values.filter(x=>typeof x==='string'&&x&&x!==field))].slice(0,80);
const moveId=x=>typeof x==='string'?x:x?.id;
const abilities=ids(playableSets.map(p=>p.abilityId),own[0].abilityId);
const items=ids(playableSets.map(p=>p.itemId),own[0].itemId);
const moves=ids(playableSets.flatMap(p=>(p.moves??[]).map(moveId)),
                moveId(own[0].moves?.[0]));
const alteredSpeed=structuredClone(own);
alteredSpeed[0].stats.spe+=75;
const privateSame=JSON.stringify(encode(own,shown,'secret-B'))===reference;
const enemyPreviewDifferent=JSON.stringify(encode(
  own,pool.slice(17,23).map(p=>p.speciesNameJa)))!==reference;
const report={
  schema:'rnad-i5-selection-counterfactual-feature-sensitivity-v1',
  actual_authority_accepted_teacher_matches:0,
  reference_known_own_team_size:6,
  reference_public_opponent_species:6,
  feature_schema:'rnad-three-phase-features-v5',
  legal_canonical_selection_candidates:60,
  probe_scope:'single public/own coordinate changed, potentially illegal hypothetical set',
  teacher_utility_compared:false,
  teacher_top1_conflict_proven:false,
  v5_complete_public_information_proven:false,
  own_ability_id_changes:variants(abilities,own[0].abilityId,
    (team,id)=>{team[0].abilityId=id;}),
  own_item_id_changes:variants(items,own[0].itemId,
    (team,id)=>{team[0].itemId=id;}),
  own_move_id_changes:variants(moves,moveId(own[0].moves?.[0]),
    (team,id)=>{team[0].moves=[id,...(team[0].moves??[]).slice(1)];}),
  known_own_speed_changes_features:
    JSON.stringify(encode(alteredSpeed))!==reference,
  public_opponent_preview_changes_features:enemyPreviewDifferent,
  opponent_private_set_is_noninterfering:privateSame
};
if(!privateSame||!enemyPreviewDifferent||
   !report.known_own_speed_changes_features||
   [report.own_ability_id_changes,report.own_item_id_changes,
    report.own_move_id_changes].some(x=>x.total<1))
  throw new Error('Selection observation boundary or sensitivity contract failed');
if(process.argv.length!==4||process.argv[2]!=='--out')
  throw new Error('Usage: node selection-counterfactual-audit.mjs --out report.json');
writeFileSync(process.argv[3],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify(report));
