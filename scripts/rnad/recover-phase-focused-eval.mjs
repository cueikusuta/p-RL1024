// Re-assess an already completed 16-seed selection-only R-NaD experiment.
// Each all-v3 control has its own v4 provenance: never substitute v4 SHA in
// one report, and never rerun expensive training to repair JSON aggregation.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {compareV4PhaseArms,compareV4PhaseModels}
  from './compare-v4-phase-arms.mjs';

export function recoverFocusedPhaseExperiment(reports) {
  const modes=['all','selection-only'];
  for(const mode of modes)for(const phase of ['all-v3','selection-v4']){
    if(!reports?.[mode]?.[phase])
      throw new Error('Missing independently accepted full Match ' + mode+'/'+phase);
  }
  const a=reports.all['all-v3'],b=reports['selection-only']['all-v3'];
  if(a.mode!=='all-v3'||b.mode!=='all-v3'||
     a.source_sha!==b.source_sha||a.engine_sha!==b.engine_sha||
     a.v3_model_sha256!==b.v3_model_sha256||
     a.aq_base_model_sha256!==b.aq_base_model_sha256||
     a.seed!==b.seed||a.pairs!==b.pairs||a.matches!==b.matches)
    throw new Error('Unmatched frozen v3 control provenance');

  // The all-v3 control should not depend on the v4 weights loaded alongside it.
  // Compare actual accepted outcomes rather than forging the same v4 model SHA.
  const outcome=r=>new Map(r.rows.map(x=>
    [x.seed+':'+x.candidateSeat,Number(x.won)]));
  const A=outcome(a),B=outcome(b);
  if(A.size!==a.matches||B.size!==b.matches||
    [...A].some(([k,v])=>B.get(k)!==v))
    throw new Error('All-v3 controls changed due to unrelated phase actor');

  const all=compareV4PhaseArms(a,reports.all['selection-v4'],{familySize:3});
  const isolated=compareV4PhaseArms(
    b,reports['selection-only']['selection-v4'],{familySize:3});
  const difference=compareV4PhaseModels(
    reports.all['selection-v4'],
    reports['selection-only']['selection-v4'],{familySize:3});

  return {schema:'rnad-i5-recovered-selection-only-phase-ab-v1',
    evaluated_source_sha:a.source_sha,engine_sha:a.engine_sha,
    paired_seed_start:a.seed,independent_seeds:a.pairs,
    matches_per_arm:a.matches,
    all_phase:all,selection_only:isolated,
    selection_only_vs_all:difference,
    previous_job_failure:'wrong-v4-model-control-id-in-summary-only',
    evidence_reused_not_retrained:true,exploratory_only:true,
    strength_established:false,champion_promoted:false};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const [folder,out,...rest]=process.argv.slice(2);
  if(!folder||!out||rest.length)throw new Error('Usage: RESULTS_DIRECTORY OUTPUT_JSON');
  const read=(profile,phase)=>JSON.parse(
    readFileSync(folder+'/'+profile+'-'+phase+'.json','utf8'));
  const report=recoverFocusedPhaseExperiment({
    all:{'all-v3':read('all','all-v3'),
         'selection-v4':read('all','selection-v4')},
    'selection-only':{'all-v3':read('selection-only','all-v3'),
       'selection-v4':read('selection-only','selection-v4')}
  });
  writeFileSync(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({seed:report.paired_seed_start,
    all_wins:report.all_phase.candidate_wins,
    isolated_wins:report.selection_only.candidate_wins,
    v3_wins:report.all_phase.baseline_wins,
    isolated_vs_all_delta:report.selection_only_vs_all.paired_delta,
    confidence:report.selection_only_vs_all.familywise_95pct_delta_bounds,
    conclusion:'exploratory-only-no-proven-improvement'}));
}
