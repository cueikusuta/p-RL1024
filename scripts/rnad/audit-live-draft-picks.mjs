// Actual authority-accepted v4 draft choices against frozen original AQ.
// Diagnostic single-seed match; no training or statistical strength claims.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256 as AQ_SHA}
  from '../../models/aq-base.mjs';
import {createV4CompositeAQSession} from './evaluate-v4-composite-vs-aq.mjs';
import {projectRnadObservation,rnadLegalActions}
  from '../../src/ml/legal-action-adapter.js';
import {scoreStrategyDraftRoster} from '../../src/ai/team-assessment.js';
import {summarizeAcceptedDraftPick} from './draft-pick-review.mjs';
import {playableSets} from '../../src/production-runtime.js';
import battle from '../../data/catalog/battle-data.json' with {type:'json'};
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const sha=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
function names(report,codes){
  return codes.map(i=>report.roster[i].species).join(' / ');
}
function costs(report,codes){
  return codes.reduce((n,i)=>n+report.roster[i].credit_cost,0);
}
export function runLiveDraftPickAudit({v4Path,v3Path,seed,seat,out}){
  if(typeof v4Path!=='string'||!v4Path||typeof v3Path!=='string'||!v3Path||
     !Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||![0,1].includes(seat))
    throw new TypeError('Invalid bounded accepted draft audit input');
  const source=verifiedRunningSourceSha();
  const actor=createV4CompositeAQSession({v4Path,v3Path,seat,mode:'draft-v4'});
  let pending=null;
  const drafts=[];
  const wrapped={
    select(view){
      if(pending)throw new Error('Unaccepted draft review boundary');
      const choice=actor.select(view);
      const observation=projectRnadObservation(view,{seat});
      if(observation.phase==='draft'){
        const legalActions=rnadLegalActions(view);
        const index=legalActions.findIndex(x=>same(x,choice));
        if(index<0)throw new Error('Model proposed non-authority draft pick');
        pending={observation,legalActions,index};
      }
      return choice;
    },
    commit(){
      // Only the authoritative Match callback can commit a draft review.
      actor.commit();
      if(!pending)return;
      const {observation,legalActions,index}=pending;
      const pool=[...(observation.self?.carry??[]),...(observation.self?.fresh??[])];
      const byId=new Map(pool.map(p=>[p.instanceId,p]));
      const utilities=legalActions.map(candidate=>scoreStrategyDraftRoster(
        candidate.map(id=>byId.get(id)),{
          data:battle,credits:observation.self.credits,
          carry:observation.self.carry??[]}));
      const review=summarizeAcceptedDraftPick(
        observation,legalActions,index,utilities,battle,false);
      drafts.push(review);
      pending=null;
    },
    stats:()=>actor.stats(),
    finish:reward=>actor.finish(reward)
  };
  const aq=createOpponent();
  if(aq.modelSha256!==AQ_SHA)
    throw new Error('Unexpected exact AQ-Base identity');
  const result=runStrategyMatch({
    seed,policies:[0,1].map(s=>s===seat?'pure-rl-linear':'aq-base'),
    pureRlSessions:[0,1].map(s=>s===seat?wrapped:null),
    pureRlAllPhases:true,strategyMode:'integrated',
    v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq});
  if(!result.completed||result.score?.[result.winner]!==4||
     drafts.length!==result.rounds||pending||
     result.pureRlStats?.[seat]?.pending)
    throw new Error('Incomplete actual authority-accepted student draft Match');
  const report={schema:'rnad-v4-accepted-live-draft-picks-v1',
    source_sha:source,engine_sha:provenance.sourceCommit,
    aq_base_model_sha256:AQ_SHA,
    v4_model_sha256:sha(v4Path),v3_model_sha256:sha(v3Path),
    policy:'draft-v4,selection+combat-v3',seed,seat,
    rounds:result.rounds,winner:result.winner,score:result.score,
    student_won:result.winner===seat,completed:true,
    is_champion:false,statistical_strength_established:false,drafts};
  const md=[
    '# 実際に受理されたR-NaDドラフトピック',
    '',
    '**ここに表示する学生ピックは、教師データ上の予測ではなく、AQ-Base-001との完全4勝Matchで受理された行動です。**',
    '',
    '- Mode: draft-v4 / selection+combat-v3',
    '- Seed: '+seed+', AI席: '+seat+', 完全Match: '+result.score.join(' : '),
    '- 試合結果: '+(report.student_won?'R-NaD勝利':'R-NaD敗北')+'（1 Matchのみ、強さの証明ではありません）',
    '- Model: '+report.v4_model_sha256,
    '- Source/engine: '+source+' / '+report.engine_sha,
    ''
  ];
  for(const [n,d] of drafts.entries()){
    md.push('## Round '+(n+1)+' / クレジット '+d.credits+' / 合法候補 '+d.candidate_codes.length);
    md.push('');
    md.push('- 前ラウンドから継続: '+(d.roster.filter(p=>p.is_carry).map(p=>p.species).join('、')||'なし'));
    md.push('- 新規提示: '+d.roster.filter(p=>!p.is_carry).map(p=>
      p.species+'（'+p.credit_cost+'）').join('、'));
    md.push('- **モデルの受理済み選択6体:** '+names(d,d.accepted_roster_codes));
    md.push('- 使用クレジット: '+d.accepted_spent+'/'+d.credits);
    md.push('- 公開AQ評価での順位: '+d.teacher_rank+
      '位 / '+d.candidate_codes.length+'候補');
    md.push('');
    md.push('| AQ公開スコア上位 | チーム6体 | 費用 | スコア |');
    md.push('|---:|---|---:|---:|');
    for(const [i,c] of d.teacher_top10.slice(0,5).entries()){
      md.push('| '+(i+1)+' | '+names(d,d.candidate_codes[c.candidate_index])+
        ' | '+costs(d,d.candidate_codes[c.candidate_index])+
        ' | '+c.utility.toFixed(3)+' |');
    }
    md.push('');
  }
  writeFileSync(out+'/LIVE_DRAFT_PICKS.md',md.join('\n')+'\n',{flag:'wx'});
  writeFileSync(out+'/live-draft-picks.json',JSON.stringify(report,null,2)+'\n',
                {flag:'wx'});
  return report;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const a=process.argv.slice(2),o=new Map(),keys=
    new Set(['--v4','--v3','--seed','--seat','--out']);
  if(a.length!==10)throw new Error('Expected five draft audit pairs');
  for(let i=0;i<a.length;i+=2){
    if(!keys.has(a[i])||o.has(a[i])||!a[i+1])
      throw new Error('Invalid draft audit argument');
    o.set(a[i],a[i+1]);
  }
  const r=runLiveDraftPickAudit({
    v4Path:o.get('--v4'),v3Path:o.get('--v3'),
    seed:Number(o.get('--seed')),seat:Number(o.get('--seat')),
    out:o.get('--out')});
  console.log(JSON.stringify({seed:r.seed,score:r.score,rounds:r.rounds,
    drafts:r.drafts.length,student_won:r.student_won}));
}
