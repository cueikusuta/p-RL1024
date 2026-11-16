"""Separate held-out greedy, both-seat four-win comparison of reward shaping.
Runs only against verified immutable archived weights and a matching new
training-source checkout. No Champion claim or omitted/estimated outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.rnad.train import current_git_sha, frozen_engine_sha


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def paired(payload, start, key, source, engine, *, mode):
    if (payload.get("source_sha") != source or
        payload.get("engine_sha") != engine or
        (payload.get("seed",start) != start if mode=="direct" else False) or
        payload.get("pairs") != 64 or payload.get("matches") != 128 or
        payload.get("inference_selection_mode") != "greedy" or
        not isinstance(payload.get("rows"), list) or
        len(payload["rows"]) != 128 or
        payload.get("wins") != sum(1 for x in payload["rows"]
                                   if x.get("won") is True)):
        raise ValueError("Untrusted or incomplete paired results")
    expected = {(s, seat) for s in range(start, start+64)
                for seat in (0, 1)}
    observed = set()
    for row in payload["rows"]:
        token = (row.get("seed"), row.get(key))
        score = row.get("score")
        if (token not in expected or token in observed or
            type(row.get("won")) is not bool or
            not isinstance(score, list) or len(score) != 2 or
            any(type(v) is not int or not 0 <= v <= 4 for v in score) or
            max(score) != 4 or
            score[token[1] if row["won"] else 1-token[1]] != 4):
            raise ValueError("Invalid four-win authority/seed/seat evidence")
        observed.add(token)
    if observed != expected:
        raise ValueError("Missing independent seed pairs")
    if mode == "direct":
        if (payload.get("schema") != "rnad-direct-switch-guard-paired-v1"
            or payload.get("deployment_policy") !=
            {"teacher_switch_guard": "anti-loop-v1",
             "rnad_switch_guard": "anti-loop-v1"} or
            payload.get("is_champion") is not False):
            raise ValueError("Wrong guarded direct-eval policy identity")
    else:
        if (payload.get("schema") !=
              "rnad-three-phase-switch-guard-vs-basic-paired-v1"
            or payload.get("deployment_policy") !=
                {"switch_guard": "anti-loop-v1"} or
            payload.get("opponent_id") != "basic-all-phases" or
            payload.get("champion_promoted") is not False):
            raise ValueError("Wrong guarded Basic-eval policy identity")


def evaluate(*, model, initial, trained, out):
    if model not in ("base128", "base256", "c005"):
        raise ValueError("Wrong model family")
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Do not overwrite existing eval outcomes")
    root.mkdir(parents=True, exist_ok=True)
    original = Path(initial)
    source = current_git_sha()
    engine = frozen_engine_sha()
    study = json.loads((Path(trained)/"study-report.json").read_text())
    if not (study["source_sha"] == source and
        study["engine_sha"] == engine and
        study["initial_model"] == model and
        study["original_model_sha256"] == sha(original) and
        study["reward_objective"] == "opportunity-v1" and
        study["guard"] == "anti-loop-v1" and
        study["complete_matches"] == (128 if model == "c005" else 64) and
        study["is_champion"] is False):
        raise ValueError("Untrusted frozen source or trained model")
    snapshots = {}
    for amount, digest in study["model_snapshots"].items():
        p = Path(trained)/f"model-u{amount}.json"
        if sha(p) != digest or digest == sha(original):
            raise ValueError("Missing changed/shahashed model")
        snapshots[amount] = p
    expected = {"64","128"} if model=="c005" else {"64"}
    if set(snapshots) != expected:
        raise ValueError("Wrong study budget snapshots")
    candidates={"original":original,
                "u64":snapshots["64"]}
    if model=="c005":
        candidates["u128"]=snapshots["128"]
    scores={}
    def invoke(name, args, key, seed, *, a=None, z=None, mode):
        target=root/f"{name}.json"
        if target.exists():raise ValueError("Refusing previous evaluation")
        subprocess.run(["node",*args,"--out",str(target)],
                       check=True,timeout=2100)
        report=json.loads(target.read_text())
        paired(report,seed,key,source,engine,mode=mode)
        if mode=="direct":
            if (report["teacher_model_sha256"] != sha(candidates[a])
                or report["rnad_model_sha256"] != sha(candidates[z])):
                raise ValueError("Wrong paired actor identity")
        else:
            if report["model_sha256"] != sha(candidates[z]):
                raise ValueError("Wrong Basic actor identity")
        scores[name]={"wins":report["wins"],"matches":128}
        print(json.dumps({"model":model,"evaluation":name,
                          **scores[name]}),flush=True)
    for label,p in candidates.items():
        invoke(f"{label}-basic",
            ["scripts/rnad/evaluate-neural-basic.mjs",
             "--model",str(p),"--seed","1781001",
             "--pairs","64","--selection-mode","greedy",
             "--switch-guard","anti-loop-v1"],
            "candidateSeat",1781001,z=label,mode="basic")
    for left,right in ([("original","u64")]+
                       ([("original","u128"),("u64","u128")] if model=="c005"
                        else [])):
        invoke(f"{right}-vs-{left}",
            ["scripts/rnad/evaluate-direct.mjs",
             "--teacher",str(candidates[left]),
             "--rnad",str(candidates[right]),
             "--seed","1780001","--pairs","64",
             "--selection-mode","greedy",
             "--guard-teacher","anti-loop-v1",
             "--guard-rnad","anti-loop-v1"],
            "rnadSeat",1780001,a=left,z=right,mode="direct")
    final={"schema":"rnad-switch-opportunity-paired-evaluation-v1",
           "source_sha":source,"engine_sha":engine,
           "initial_model":model,"reward_objective":"opportunity-v1",
           "guard":"anti-loop-v1","train_matches":study["complete_matches"],
           "model_sha256":{k:sha(v) for k,v in candidates.items()},
           "training_seed":study["training_seed"],
           "holdout_direct_seed":1780001,
           "holdout_basic_seed":1781001,
           "results":scores,"all_full_four_win":True,
           "statistical_improvement_confirmed":False,
           "is_champion":False}
    (root/"evaluation-report.json").write_text(
        json.dumps(final,indent=2,allow_nan=False)+"\n")
    print(json.dumps(final),flush=True)
    return final


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=("base128","base256","c005"),required=True)
    p.add_argument("--initial",required=True)
    p.add_argument("--trained",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    evaluate(model=a.model,initial=a.initial,trained=a.trained,out=a.out)
