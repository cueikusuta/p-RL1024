"""Selection follow-up: one development-selected candidate versus frozen reference
on an entirely new accepted 256-Match AQ teacher block.

This confirms imitation only, never actual AQ match strength or league readiness.
The candidate is fixed by old reused development data BEFORE fresh traces are read.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from scripts.rnad.train import current_git_sha
from scripts.rnad.train_selection_specialist import (
    SelectionSpecialist, load_traces, evaluate, evaluate_recurrent_baseline,
    feature_sufficiency_audit, SOURCE_SHA, TEACHER,
)
from scripts.rnad.neural import (
    RNADNetwork, load_exported_json, THREE_PHASE_FEATURE_SCHEMA_V5,
    STATE_FEATURES_V5, ACTION_FEATURES_V5,
)
from scripts.rnad.selection_specialist_followup import gap_diagnostics

FRESH_START = 2473001
FRESH_MATCHES = 256
REFERENCE_SHA = "1b5f39cee6f00d1fe5c095442996fb706086dd4456d84ce5974a861f9e4e81cd"
FROZEN_V5_SHA = "6580738ace427364d89e83b94b465ea4f7febead1cb4e18f578d7fb8cb79f5f6"
FOLLOWUP_RUN = 44042088126


def choose_from_development(report):
    if (report.get("schema")!="rnad-i5-selection-followup-exploratory-v1" or
        report.get("heldout_reused_for_model_selection") is not True or
        report.get("confirmed_independent_generalization") is not False or
        report.get("teacher_match_source_run")!=44032488678 or
        report.get("teacher_trace_source_sha")!=SOURCE_SHA or
        sorted(report.get("arms",{}))!=sorted(
            f"structured-hard-m{n}-w{w}"
            for n in (128,256) for w in (64,96,128))):
        raise ValueError("Untrusted development-only model-selection report")
    # All six arms were explored on reused development holdout.
    # Freeze exactly ONE candidate before reading fresh teacher traces.
    key=min(report["arms"],key=lambda name:(
        -report["arms"][name]["development_heldout"]["top1"],
        report["arms"][name]["development_heldout"]["regret"],
        report["arms"][name]["network_width"],
        name))
    return key,report["arms"][key]


def from_archive(path,sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError("Frozen archived reference model SHA mismatch")
    data=json.loads(raw)
    if data.get("arm")!="structured-hard" or data.get("structured") is not True:
        raise ValueError("Wrong selection baseline model")
    net=SelectionSpecialist(structured=True,width=96)
    original=net.state_dict()
    if set(original)!=set(data.get("weights",{})):
        raise ValueError("Archived baseline schema mismatch")
    states={}
    for k,v in original.items():
        entry=data["weights"][k]
        if entry["shape"]!=list(v.shape):
            raise ValueError("Archived baseline tensor dimensions differ")
        t=torch.tensor(entry["values"],dtype=v.dtype).reshape(v.shape)
        states[k]=t
    net.load_state_dict(states,strict=True)
    return net.eval()


def paired_cluster_delta95(base,new):
    seeds=sorted(base)
    if len(seeds)!=FRESH_MATCHES or seeds!=sorted(new):
        raise ValueError("Incomplete paired independent Match seed clusters")
    diffs=[new[s]["top1"]-base[s]["top1"] for s in seeds]
    rng=random.Random(9291)
    n=len(diffs)
    samples=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n
                   for _ in range(5000))
    return {
        "seed_clusters":n,
        "mean_paired_cluster_top1_delta":sum(diffs)/n,
        "approximate_two_sided_95pct_cluster_bootstrap_CI":
            [samples[125],samples[4874]],
        "significant_positive_at_approximate_95pct":samples[125]>0.,
        "bootstrap_is_approximate":True,
    }


def run(followup_report,followup_models,reference_model,v5_baseline_model,fresh_dir,out):
    torch.set_num_threads(1)
    source=json.loads(Path(followup_report).read_bytes())
    selected,row=choose_from_development(source)
    filename=row["weight_file"]
    if filename!=selected+".pt":
        raise ValueError("Unexpected selected research model pathname")
    selected_path=Path(followup_models)/filename
    if hashlib.sha256(selected_path.read_bytes()).hexdigest()!=row["weight_sha256"]:
        raise ValueError("Selected study model failed frozen artifact SHA")
    model=SelectionSpecialist(structured=True,width=row["network_width"])
    model.load_state_dict(torch.load(selected_path,map_location="cpu",
                                     weights_only=True),strict=True)
    model.eval()
    frozen=from_archive(reference_model,REFERENCE_SHA)
    v5=RNADNetwork(state_features=STATE_FEATURES_V5,
                   action_features=ACTION_FEATURES_V5)
    v5sha=load_exported_json(v5,v5_baseline_model,
                             expected_feature_schema=THREE_PHASE_FEATURE_SCHEMA_V5)
    if v5sha!=FROZEN_V5_SHA:
        raise ValueError("Frozen v5 selection-focus model is not the recorded baseline")
    v5.eval()
    # Candidate selection has happened; only now read independent fresh seeds.
    fresh,engine=load_traces(fresh_dir,FRESH_START,FRESH_MATCHES)
    if engine!=source["engine_sha"] or source["teacher"]!=TEACHER:
        raise ValueError("Fresh teacher engine or provenance differs")
    baseline,base_clusters=evaluate_recurrent_baseline(
        v5,fresh_dir,seed_start=FRESH_START,expected_matches=FRESH_MATCHES)
    prior_structured,_=evaluate(frozen,fresh,expected_matches=FRESH_MATCHES)
    candidate,cand_clusters=evaluate(model,fresh,expected_matches=FRESH_MATCHES)
    delta=paired_cluster_delta95(base_clusters,cand_clusters)
    result={
        "schema":"rnad-i5-selection-fresh-confirmation-v1",
        "source_sha":current_git_sha(),
        "trace_original_source_sha":SOURCE_SHA,
        "engine_sha":engine,
        "teacher":TEACHER,
        "candidate_selection_source_run":FOLLOWUP_RUN,
        "selected_on_reused_development_holdout_only":selected,
        "fresh_independent_start_seed":FRESH_START,
        "fresh_independent_matches":FRESH_MATCHES,
        "archived_structured_reference_sha256":REFERENCE_SHA,
        "frozen_v5_baseline_sha256":FROZEN_V5_SHA,
        "candidate_sha256":row["weight_sha256"],
        "prior_structured_top1":prior_structured["top1"],
        "base_top1":baseline["top1"],
        "candidate_top1":candidate["top1"],
        "base_top10":baseline["top10"],
        "candidate_top10":candidate["top10"],
        "base_regret":baseline["regret"],
        "candidate_regret":candidate["regret"],
        "paired_cluster_result":delta,
        "gap_and_trio_lead_diagnostics":gap_diagnostics(model,fresh),
        "fresh_full_observation_alias_diagnostics":feature_sufficiency_audit(fresh),
        "teacher_visited_only":True,
        "selection_phase_only":True,
        "league_training_allowed":False,
        "strength_established":False,
        "champion_promoted":False,
    }
    folder=Path(out)
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"selection-fresh-comparison.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    text=f"""# I5 選出・未使用256教師Matchによる独立比較

- 比較元: 凍結v5選出重点モデル（同じ未使用Match、全フェーズのrecurrent memoryを保持）
- 参考値: archived structured-hard 256Match×96幅の新規データTop1 {prior_structured["top1"]:.2%}
- 事前固定候補: {selected}（既使用開発データから選択）
- 新規未使用教師Match: {FRESH_MATCHES}独立seed、旧学習・開発seedと非重複
- Top1: 対照 {baseline["top1"]:.2%} / 候補 {candidate["top1"]:.2%}
- 対応ありMatch単位差: {delta["mean_paired_cluster_top1_delta"]:+.2%}
- 近似95% CI: [{delta["approximate_two_sided_95pct_cluster_bootstrap_CI"][0]:+.2%}, {delta["approximate_two_sided_95pct_cluster_bootstrap_CI"][1]:+.2%}]
- 正の95%近似CI下限: {delta["significant_positive_at_approximate_95pct"]}

**教師Top1一致率のみ。実戦勝率、3相同一方策の80%ゲート、Champion昇格を意味しない。**
"""
    (folder/"FRESH_REPORT.md").write_text(text)
    print(text,flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--followup-report",required=True)
    p.add_argument("--followup-models",required=True)
    p.add_argument("--reference-model",required=True)
    p.add_argument("--v5-baseline-model",required=True)
    p.add_argument("--fresh-dir",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    run(args.followup_report,args.followup_models,args.reference_model,
        args.v5_baseline_model,args.fresh_dir,args.out)


if __name__=="__main__":
    main()
