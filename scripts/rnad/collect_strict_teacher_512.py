"""Fixed-index utility-verified AQ teacher holdout collection, never cherry-picked by student outcomes.

The old frozen AQ teacher either exports a fully verified four-win Match or raises
'Accepted public AQ teacher does not maximize its declared utility'.  In the latter
case record and *exclude* the invalid teacher-utility trace, without changing the
strict invariant, source, engine, teacher action, or model.  Each of four streams
collects the first 128 verified Matches in modulo-4 seed order.

Result can only support claims conditional on a utility-consistent teacher.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE="022aa9cd5b4ac37514df1d845ea00c11c3c13a5c"
SEED_START=2522001
SHARDS=4
PER_SHARD=128
MAX_ATTEMPTS=256
SCHEMA="rnad-i5-strict-utility-qualified-teacher-shard-v1"
INVARIANT="Accepted public AQ teacher does not maximize its declared utility"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get_seed(shard,attempt):
    if type(shard)!=int or shard not in range(SHARDS) or type(attempt)!=int or not 0<=attempt<MAX_ATTEMPTS:
        raise ValueError("Invalid public teacher holdout shard/attempt")
    return SEED_START+SHARDS*attempt+shard


def collect(shard, output):
    if type(shard)!=int or shard not in range(SHARDS):
        raise ValueError("Unregistered independent teacher shard")
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    trace_dir=root/"traces"
    trace_dir.mkdir()
    accepted=[]
    rejected=[]
    for attempt in range(MAX_ATTEMPTS):
        seed=get_seed(shard,attempt)
        path=trace_dir/f"teacher-{seed}.json"
        cmd=["node","scripts/rnad/export-teacher.mjs",
             "--seed",str(seed),"--out",str(path),
             "--three-phase","true",
             "--teacher","aq-public-strategy-selection",
             "--features-v5","true",
             "--utility-labels","true",
             "--draft-review","true"]
        result=subprocess.run(cmd,stdin=subprocess.DEVNULL,
            capture_output=True,text=True,timeout=480)
        if result.returncode:
            if (INVARIANT not in result.stderr or path.exists() or
                "Traceback" in result.stderr):
                raise RuntimeError(f"Nonqualified public teacher failed unexpectedly at seed {seed}: {result.stderr[-500:]}")
            rejected.append({"seed":seed,"reason":"strict_public_teacher_utility_invariant_failed"})
        else:
            data=path.read_bytes()
            trace=json.loads(data)
            if (trace.get("source_sha")!=SOURCE or trace.get("seed")!=seed or
                trace.get("teacher")!="rnad-aq-public-strategy-selection-v1" or
                trace.get("feature_schema")!="rnad-three-phase-features-v5" or
                trace.get("completed") is not True or
                trace.get("score",[])[trace["winner"]]!=4 or
                trace.get("utility_labels")!="aq-public-phase-rank-v1" or
                not trace.get("examples") or
                any(e.get("phase")=="draft" and (
                    e.get("candidate_utilities") is None or
                    e.get("draft_review",{}).get("teacher_rank")!=1)
                    for e in trace["examples"])):
                raise ValueError(f"Invalid authority/teacher trace {seed}")
            accepted.append({"seed":seed,"sha256":sha(data)})
        if len(accepted)==PER_SHARD:
            break
        if attempt % 32==31:
            print(json.dumps({"shard":shard,"attempts":attempt+1,
              "verified_complete_matches":len(accepted),
              "rejected_utility_inconsistent":len(rejected)}),flush=True)
    if len(accepted)!=PER_SHARD:
        raise ValueError(f"Insufficient strict verified teacher Match clusters shard {shard}: {len(accepted)}/{PER_SHARD}")
    manifest={"schema":SCHEMA,"source_sha":SOURCE,"shard":shard,
        "seed_modulus":SHARDS,"accepted_matches":len(accepted),
        "attempted_seed_count":len(accepted)+len(rejected),
        "teacher":"rnad-aq-public-strategy-selection-v1",
        "utility_audit_unchanged":True,"students_used_in_filter":False,
        "qualification":"strict_teacher_utility_invariant_pass",
        "accepted":accepted,"rejected":rejected}
    (root/"qualified-shard.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"shard":shard,"accepted":len(accepted),
        "excluded":len(rejected),"manifest_sha256":sha(
            (root/"qualified-shard.json").read_bytes())}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int,required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    collect(a.shard,a.out)


if __name__=="__main__":
    main()
