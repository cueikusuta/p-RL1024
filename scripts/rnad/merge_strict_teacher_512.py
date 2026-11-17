"""Audit all 4 independent frozen teacher shard artifacts before scoring students."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.rnad.collect_strict_teacher_512 import (
    SOURCE, SCHEMA, SEED_START, PER_SHARD, SHARDS, get_seed)

FULL_SCHEMA="rnad-i5-utility-qualified-teacher-512-v1"
RULE="strict_utility_audit_pass_before_any_model_comparison"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def merge(root,out):
    root=Path(root);out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    accepted=[];rejected=[];attempted=set();shard_receipts=[]
    for shard in range(SHARDS):
        base=root/str(shard)
        path=base/"qualified-shard.json"
        raw=path.read_bytes()
        r=json.loads(raw)
        if (r.get("schema")!=SCHEMA or r.get("source_sha")!=SOURCE
            or r.get("shard")!=shard or r.get("seed_modulus")!=SHARDS
            or r.get("accepted_matches")!=PER_SHARD
            or r.get("utility_audit_unchanged") is not True
            or r.get("students_used_in_filter") is not False
            or r.get("qualification")!="strict_teacher_utility_invariant_pass"
            or len(r.get("accepted",[]))!=PER_SHARD
            or r.get("attempted_seed_count")!=
                PER_SHARD+len(r.get("rejected",[]))):
            raise ValueError("Untrusted strict utility-qualified Match shard")
        per_attempt = {a["seed"]:{"status":"accepted","sha256":a["sha256"]}
                       for a in r["accepted"]}
        if len(per_attempt)!=PER_SHARD:
            raise ValueError("Repeated accepted Match seeds")
        for row in r["rejected"]:
            seed=row.get("seed")
            if (row.get("reason")!="strict_public_teacher_utility_invariant_failed"
                    or seed in per_attempt):
                raise ValueError("Untrusted excluded Match reason or duplicate")
            per_attempt[seed]={"status":"rejected"}
            rejected.append({"seed":seed,"reason":row["reason"]})
        expected={get_seed(shard,i) for i in range(len(per_attempt))}
        if set(per_attempt)!=expected or len(attempted & expected):
            raise ValueError("Qualified holdout is not the registered first-128 prefix")
        attempted.update(expected)
        if per_attempt[max(expected)]["status"]!="accepted":
            raise ValueError("Qualified holdout did not stop on 128th verified Match")
        for row in r["accepted"]:
            seed=row["seed"]
            file=base/"traces"/f"teacher-{seed}.json"
            raw_match=file.read_bytes()
            if digest(raw_match)!=row["sha256"]:
                raise ValueError("Accepted teacher Match SHA mismatch")
            match=json.loads(raw_match)
            if (match.get("source_sha")!=SOURCE or match.get("seed")!=seed
                    or match.get("completed") is not True or
                    match["score"][match["winner"]]!=4 or
                    match.get("utility_labels")!="aq-public-phase-rank-v1"):
                raise ValueError("Invalid frozen teacher Match evidence")
            shutil.copyfile(file,out/file.name)
            accepted.append(seed)
        shard_receipts.append({"shard":shard,"manifest_sha256":digest(raw),
            "accepted":PER_SHARD,"excluded":len(r["rejected"]),
            "first_seed":get_seed(shard,0),
            "last_attempted_seed":max(expected)})
    if len(set(accepted))!=512 or len(attempted)!=len(accepted)+len(rejected):
        raise ValueError("Independent strict-qualified Match clusters incomplete")
    if not set(accepted).isdisjoint(set(range(2471001,2471257)) |
                                   set(range(2472001,2472129)) |
                                   set(range(2520001,2520257))):
        raise ValueError("Teacher training/development data contaminated holdout")
    if 2522144 not in {r["seed"] for r in rejected}:
        raise ValueError("Previously verified strict audit failure unexpectedly absent")
    manifest={"schema":FULL_SCHEMA,"source_sha":SOURCE,"rule":RULE,
        "population":"accepted_complete_matches_whose_public_AQ_teacher_utility_invariant_passed",
        "filter_not_based_on_model_outputs":True,
        "accepted_seeds":sorted(accepted),
        "excluded_seeds":sorted(rejected,key=lambda v:v["seed"]),
        "shard_receipts":shard_receipts,
        "independent_complete_match_clusters":512,
        "student_model_selection_uses_holdout":False,
        "qualifier_is_strict_no_invariant_weakening":True}
    canonical=json.dumps({"accepted_seeds":manifest["accepted_seeds"],
        "excluded_seeds":manifest["excluded_seeds"]},
        sort_keys=True,separators=(",",":")).encode()
    manifest["qualification_sha256"]=digest(canonical)
    (out/"qualified-manifest.json").write_text(
        json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"strict_teacher_complete_matches":len(accepted),
        "excluded":len(rejected),"old_failure_seed":2522144,
        "qualified_manifest_sha256":digest((out/"qualified-manifest.json").read_bytes())}),
        flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shards",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    merge(a.shards,a.out)


if __name__=="__main__":
    main()
