// Fail closed on any missing, duplicated, malformed or cross-run shard.
// Only a fully complete fixed-N preregistered result has a strength verdict.
import {readFileSync,readdirSync,mkdirSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {readConfirmationPlan,aggregateConfirmationShards}
  from './draft-confirmation-protocol.mjs';

export function aggregateConfirmationDirectory({planPath,expectedPlanSha,
  expectedSourceSha,artifactDir,outDir,runId}){
  const {plan,plan_sha256}=readConfirmationPlan(planPath);
  if(plan_sha256!==expectedPlanSha||!Number.isSafeInteger(runId)||runId<1)
    throw new Error('Unexpected fixed-N plan / run identity');
  const dirs=readdirSync(artifactDir,{withFileTypes:true})
    .filter(x=>x.isDirectory()).map(x=>x.name);
  const prefix='i5-confirm-shard-';
  if(dirs.some(x=>!new RegExp('^'+prefix+'[0-9]+$').test(x)))
    throw new Error('Unexpected artifact alongside frozen shard results');
  if(dirs.length!==plan.seed_count/plan.shard_size)
    throw new Error('Missing required shard artifacts; no statistical verdict');
  const shards=dirs.map(name=>{
    const file=join(artifactDir,name,'shard.json');
    const data=JSON.parse(readFileSync(file,'utf8'));
    if(name!==prefix+data.shard_index)
      throw new Error('Mismatched shard artifact directory/index');
    return data;
  });
  const report=aggregateConfirmationShards(shards,plan,plan_sha256,expectedSourceSha);
  const final={...report,github_run_id:runId,
    benchmark:plan.benchmark,candidate_mode:plan.candidate_mode,
    baseline_mode:plan.baseline_mode,candidate_sha256:plan.candidate_sha256,
    v3_sha256:plan.v3_sha256,aq_sha256:plan.aq_sha256,
    engine_sha:plan.engine_sha,alpha:plan.alpha,
    test_plan_seed_namespace:plan.seed_namespace,
    stop_rule:plan.stop_rule,independent_review_complete:false};
  mkdirSync(outDir,{recursive:true});
  writeFileSync(join(outDir,'fixed-n-report.json'),
    JSON.stringify(final,null,2)+'\n',{flag:'wx'});
  const columns=['seed','control_seat0','control_seat1',
    'candidate_seat0','candidate_seat1','paired_delta','aq_win_rate'];
  const lines=[columns.join(','),
    ...final.by_seed.map(x=>columns.map(k=>x[k]).join(','))];
  writeFileSync(join(outDir,'per-seed.csv'),lines.join('\n')+'\n',{flag:'wx'});
  const asPercent=x=>(100*x).toFixed(2)+'%';
  const range=c=>'['+asPercent(c.lower)+', '+asPercent(c.upper)+']';
  const md=[
    '# 事前登録・固定N：R-NaDドラフト対オリジナルAQの対応あり強度検証',
    '',
    '**事前に固定した全seed×両席の完全4勝Matchが揃った場合にのみ結果を算出。**',
    '',
    '- 事前登録Plan ID: '+plan.id,
    '- Plan SHA256: '+plan_sha256,
    '- Source SHA: '+expectedSourceSha,
    '- Candidate draft SHA256: '+plan.candidate_sha256,
    '- Frozen v3 SHA256: '+plan.v3_sha256,
    '- Original AQ SHA256: '+plan.aq_sha256,
    '- 検証単位: '+plan.seed_count+'独立seed×両席、合計'+(4*plan.seed_count)+'完全4勝Match',
    '- 停止規則: '+plan.stop_rule,
    '- CI: seedクラスターに対する固定N経験Bernstein区間。2評価基準でBonferroni補正。',
    '',
    '| 比較指標 | 観測値 | 同時95%以上の信頼区間 | 採用判定用条件 |',
    '|---|---:|---|---|',
    '| Deep Setsドラフト−凍結v3 対応ありΔ | '+
      asPercent(report.paired.mean)+' | '+range(report.paired)+
      ' | 下限 > 0、かつ観測差 ≥ '+asPercent(plan.min_delta)+' |',
    '| Deep Sets+v3選出戦闘 対オリジナルAQ勝率 | '+
      asPercent(report.aq_absolute.mean)+' | '+range(report.aq_absolute)+
      ' | 下限 > 50% |',
    '',
    '- v3勝利: '+report.baseline_wins+'/'+(2*plan.seed_count),
    '- 候補勝利: '+report.candidate_wins+'/'+(2*plan.seed_count),
    '- 対v3差の統計的方向: '+report.paired_direction,
    '- 対AQ絶対勝率の統計的方向: '+report.aq_absolute_direction,
    '- **統計レビュー候補: '+(report.eligible_for_independent_review?'YES':'NO')+'**',
    '- Champion昇格: **未実施**（別ChatGPT独立レビューとCI確認が必須）',
    '',
    '生データは同一Workflow Runの各 i5-confirm-shard-* Artifactに保存。',
    '各Artifactに全Matchの受理済みドラフト・所持候補・予算・結果とSHAを保持。',
    '集約した全seedの対応あり観測値は per-seed.csv を参照。',
    '',
    '**注意:** 固定N区間であり、途中停止や結果を見た再試行には使えません。'
  ];
  writeFileSync(join(outDir,'CONFIRMATION_REPORT.md'),md.join('\n')+'\n',{flag:'wx'});
  return {plan_id:plan.id,shards:shards.length,
    complete_seed_blocks:report.independent_seed_blocks,
    complete_four_win_matches:report.complete_four_win_matches,
    verdict:report.verdict,champion_promoted:false};
}
function parse(argv,names){
  if(argv.length!==names.length*2)throw new Error('Missing aggregation options');
  const p=new Map();
  for(let i=0;i<argv.length;i+=2){
    if(!names.includes(argv[i])||p.has(argv[i])||!argv[i+1])
      throw new Error('Unknown aggregation option');
    p.set(argv[i],argv[i+1]);
  }
  return p;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const p=parse(process.argv.slice(2),
    ['--plan','--plan-sha','--source-sha','--artifacts','--out','--run-id']);
  console.log(JSON.stringify(aggregateConfirmationDirectory({
    planPath:p.get('--plan'),expectedPlanSha:p.get('--plan-sha'),
    expectedSourceSha:p.get('--source-sha'),artifactDir:p.get('--artifacts'),
    outDir:p.get('--out'),runId:Number(p.get('--run-id'))
  })));
}
