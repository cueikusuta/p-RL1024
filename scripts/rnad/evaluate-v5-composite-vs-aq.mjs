// Versioned v5 public-feature phase swaps with frozen v3 battle against
// original AQ-Base. The existing v4 composite actor enforces acceptance,
// local GRU memory, public observations and the anti-loop guard.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256 as AQ_SHA}
  from '../../models/aq-base.mjs';
import {createV4CompositeAQSession}
  from './evaluate-v4-composite-vs-aq.mjs';
import {pairedHoeffding95} from './aggregate-eval.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};
const hash=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
export const V5_AQ_SCHEMA='rnad-v5-v3-composite-vs-original-aq-paired-v1';
const modes={'all-v3':'all-v3','selection-v5':'selection-v4',
  'draft-v5':'draft-v4','both-v5':'both-v4'};
export function evaluateV5CompositeAgainstOriginalAQ({
  v5Path,v3Path,seed=2250001,pairs=4,mode='all-v3'}={}){
  if(typeof v5Path!=='string'||!v5Path||
     typeof v3Path!=='string'||!v3Path||
     !Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
     !Number.isSafeInteger(pairs)||pairs<1||pairs>100||
     seed+pairs-1>0xffffffff||!Object.hasOwn(modes,mode))
    throw new TypeError('Invalid bounded v5/v3 paired AQ evaluation');
  const digest5=hash(v5Path),digest3=hash(v3Path);
  const aq=createOpponent(),rows=[];
  let wins=0;
  for(let i=0;i<pairs;i++)for(const candidateSeat of [0,1]){
    const actor=createV4CompositeAQSession({
      v4Path:v5Path,v3Path,seat:candidateSeat,
      mode:modes[mode],expectedPhaseSchema:'rnad-three-phase-features-v5'});
    const r=runStrategyMatch({
      seed:seed+i,
      policies:[0,1].map(s=>s===candidateSeat?'pure-rl-linear':'aq-base'),
      pureRlSessions:[0,1].map(s=>s===candidateSeat?actor:null),
      pureRlAllPhases:true,strategyMode:'integrated',
      v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq});
    const stats=r.pureRlStats?.[candidateSeat];
    if(!r.completed||![0,1].includes(r.winner)||r.score[r.winner]!==4||
       !stats||stats.pending||stats.updates!==0||
       stats.accepted_decisions!==r.actionCounts[candidateSeat]+2*r.rounds)
      throw new Error('Incomplete v5 authority-accepted original AQ Match');
    const won=r.winner===candidateSeat;
    wins+=Number(won);
    rows.push({seed:seed+i,candidateSeat,won,score:r.score,rounds:r.rounds,
      accepted_decisions:stats.accepted_decisions});
  }
  return {schema:V5_AQ_SCHEMA,source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    aq_base_model_sha256:AQ_SHA,v5_model_sha256:digest5,
    v3_model_sha256:digest3,mode,seed,pairs,matches:2*pairs,
    wins,win_rate:wins/(2*pairs),
    paired_hoeffding_95pct:pairedHoeffding95(wins,pairs),
    policy:'frozen-greedy-v5-v3-anti-loop-v1',
    strength_established:false,champion_promoted:false,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const args=process.argv.slice(2),o=new Map();
  if(args.length!==12)throw new Error('Expected six v5 AQ argument pairs');
  for(let i=0;i<args.length;i+=2){
    if(!['--v5','--v3','--seed','--pairs','--mode','--out'].includes(args[i])||
       !args[i+1]||o.has(args[i]))throw new Error('Invalid v5 AQ CLI');
    o.set(args[i],args[i+1]);
  }
  const r=evaluateV5CompositeAgainstOriginalAQ({
    v5Path:o.get('--v5'),v3Path:o.get('--v3'),
    seed:Number(o.get('--seed')),pairs:Number(o.get('--pairs')),
    mode:o.get('--mode')});
  writeFileSync(o.get('--out'),JSON.stringify(r,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({mode:r.mode,matches:r.matches,wins:r.wins}));
}
