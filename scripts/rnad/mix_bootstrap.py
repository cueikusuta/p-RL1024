"""Fail-closed isolated G422 opponent-mix research; never updates source snapshots."""
import argparse
import json
import shutil
import zipfile
from collections import Counter, deque
from pathlib import Path

from scripts.rnad.g137_strength_resume_migration import verify_pins, same
from scripts.rnad.shared_league_bootstrap import digest, write_json
from scripts.rnad.shared_league_contract import _opponent, verify_manifest
from scripts.rnad.shared_league_handoff import _seal_bundle
from scripts.rnad.train import current_git_sha, frozen_engine_sha
import torch

KEYS = ("model", "target", "magnet", "optimizer", "scheduler",
        "python_rng", "torch_rng", "completed_generation_match_ids")
REPS = ("rep0", "rep1")
ARMS = ("control", "weakness")


def read(p):
    return json.loads(p.read_bytes())


def need(valid, why):
    if not valid:
        raise ValueError(why)


def seal(plan, archive, original, common):
    pinned = plan["bootstrap"]
    need(digest(archive) == pinned["sha256"], "Wrong source ZIP")
    original.mkdir()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)), "Duplicate bootstrap ZIP entry")
        for name in names:
            rp = Path(name)
            need(not rp.is_absolute() and ".." not in rp.parts
                 and not name.endswith("/"), "Invalid archive member")
            reply = original / rp
            reply.parent.mkdir(exist_ok=True, parents=True)
            reply.write_bytes(z.read(name))
    verify_pins(original)
    old = read(original / "generation-manifest.json")
    head = current_git_sha()
    source = plan["baseSource"]
    engine = frozen_engine_sha()
    old_index = read(original / "checkpoint-index.json")
    old_cp = original / "checkpoint-g0.pt"
    need(head != source and old["generation"] == 422
         and old["source_sha"] == source and old["engine_sha"] == engine == plan["engine"]
         and old["main_sha256"] == pinned["main"]
         and digest(old_cp) == old_index["sha256"] == pinned["checkpoint"],
         "Wrong G421 full-state base")
    cp = torch.load(old_cp, map_location="cpu", weights_only=True)
    need(cp["source_sha"] == source and cp["main_sha256"] == old["main_sha256"]
         and cp["magnet_sha256"] == old["magnet_sha256"]
         and cp["manifest_sha256"] == verify_manifest(old) and cp["generation"] == 422
         and cp["engine_sha"] == engine and cp["step"] == 0
         and cp["training_match_ids"] == [] and cp["is_champion"] is False
         and len(set(cp["completed_generation_match_ids"])) == 1280
         and all(k in cp for k in KEYS), "Incomplete or contaminated starting state")
    common.mkdir()
    def reseal(rel, generation, expected):
        need(digest(original / rel / "manifest.json") == expected, "Wrong input actor pin")
        return _seal_bundle(original / rel, common / rel, expected_generation=generation,
                            old_sha=source,new_sha=head,engine=engine)
    main = reseal("actor", 422, old["main_sha256"])
    magnet = reseal("magnet", 422, old["magnet_sha256"])
    history = []
    for item in old["opponents"]["historical"]:
        ident = item["id"]
        need(ident in plan["screen"]["history"], "Unauthorized historical model")
        history.append({"id": ident,
            "sha256": reseal(Path("past") / ident, int(ident[6:]), item["sha256"])})
    need([x["id"] for x in history] == plan["screen"]["history"],
         "Missing historical models")
    shutil.copytree(original / "fixed", common / "fixed")
    for item in old["opponents"]["fixed"]:
        name = item["id"].removeprefix("fixed-")
        suffix = "-v1.json" if name in ("aqbase001","coretrioaq001") else "-v3.json"
        need(digest(common / "fixed" / (name + suffix)) == item["sha256"], "Fixed-opponent drift")
    for name in ("original-draft.json", "original-selection-js.json"):
        shutil.copyfile(original / name, common / name)
    manifest = {**old, "source_sha":head, "main_sha256":main,
                "policy_manifest_sha256":main, "magnet_sha256":magnet,
                "opponents":{**old["opponents"], "historical":history}}
    verify_manifest(manifest)
    write_json(common / "generation-manifest.json", manifest)


def plan_slots(baseline, rank, quotas):
    remaining = dict(zip(rank, quotas))
    slots = []
    for x in baseline:
        if remaining[x]:
            remaining[x] -= 1
            slots.append(x)
        else:
            slots.append(None)
    fill = deque(k for k in rank for _ in range(remaining[k]))
    weighted = [fill.popleft() if x is None else x for x in slots]
    need(not fill and Counter(weighted) == dict(zip(rank,quotas)),
         "Weighted mapping mismatch")
    return weighted


def arms(plan_path, original, common, screen_path, output):
    p, screen = read(plan_path), read(screen_path)
    head, plan_sha = current_git_sha(), digest(plan_path)
    m = read(common / "generation-manifest.json")
    registry = m["opponents"]["historical"]
    need(screen["schema"] == "rnad-i5-mix-screen-v1"
         and screen["head"] == head and screen["planSha"] == plan_sha
         and screen["screenStart"] == p["screen"]["start"]
         and screen["blocks"] == p["screen"]["blocks"]
         and screen["pins"] == {"G421":m["main_sha256"], **{x["id"]:x["sha256"] for x in registry}},
         "Unsealed screen or wrong seed/provenance")
    counts = screen["counts"]
    ids = [x["id"] for x in registry]
    need(sorted(x["id"] for x in counts) == sorted(ids)
         and counts == sorted(counts,key=lambda x:(x["wins"],x["id"])), "Invalid screen ranking")
    for x in counts:
        rows = x["rows"]
        need(len(rows)==p["screen"]["blocks"] and all(isinstance(pair,list)
               and len(pair)==2 and all(type(v) is int and v in (0,1) for v in pair)
               for pair in rows) and x["wins"] == sum(map(sum,rows)), "Wrong screen seed block totals")
    rank = [ids.index(x["id"]) for x in counts]
    baseline = [ids.index(_opponent(m,i)[1]["id"]) for i in range(8,16)]
    treatment = plan_slots(baseline,rank,p["training"]["treatmentHistoryCountsByRank"])
    previous = torch.load(original / "checkpoint-g0.pt", map_location="cpu", weights_only=True)
    old_index = read(original / "checkpoint-index.json")
    for rep in REPS:
        for arm in ARMS:
            dest = output / (rep+"-"+arm)
            shutil.copytree(common,dest)
            manifest = {**m,"seed_start":p["training"][rep]}
            if arm == "weakness":
                manifest["pilot_historical_actor_indices"] = treatment
            manifest_sha = verify_manifest(manifest)
            (dest / "generation-manifest.json").unlink()
            write_json(dest / "generation-manifest.json", manifest)
            cp = {**previous, "source_sha":head, "main_sha256":m["main_sha256"],
                  "magnet_sha256":m["magnet_sha256"], "manifest_sha256":manifest_sha}
            cp_out = dest / "checkpoint-g0.pt"
            torch.save(cp,cp_out)
            after = torch.load(cp_out,map_location="cpu",weights_only=True)
            need(all(same(previous[k],after[k]) for k in KEYS), "Starting training state drift")
            write_json(dest/"checkpoint-index.json",{**old_index,"sha256":digest(cp_out)})
            verify_roles = {role: Counter(_opponent(manifest,i)[1]["id"] for i in range(20)
                                         if _opponent(manifest,i)[0]==role)
                            for role in ("self","historical","fixed")}
            need({k:sum(v.values()) for k,v in verify_roles.items()} ==
                 {"self":8,"historical":8,"fixed":4}, "Role counts changed")
            write_json(dest/"pilot-input.json",{
                "rep":rep,"arm":arm,"head":head,"planSha":plan_sha,
                "originalCheckpoint":p["bootstrap"]["checkpoint"],
                "manifestSha":manifest_sha,"seedStart":manifest["seed_start"],
                "roleOpponentCounts":{k:dict(v) for k,v in verify_roles.items()},
                "screenRank":[x["id"] for x in counts],
                "screenSha":digest(screen_path)})
            (dest/"file-pins.sha256").write_text("".join(
                f"{digest(f)}  {f.relative_to(dest)}\n" for f in sorted(dest.rglob("*"))
                if f.is_file() and f.name!="file-pins.sha256"))
            verify_pins(dest)
            print(json.dumps(read(dest/"pilot-input.json"),sort_keys=True),flush=True)


def receipt(plan_path, bootstrap, result, rep, arm):
    p = read(plan_path)
    proof = read(bootstrap/"pilot-input.json")
    manifest = read(bootstrap/"generation-manifest.json")
    rep_out = read(result/"generation423-report.json")
    index = read(result/"generation423-index.json")
    actor = digest(result/"generation423-actor"/"manifest.json")
    full = digest(result/"checkpoint-g423.pt")
    need(proof["rep"]==rep and proof["arm"]==arm
         and proof["head"]==current_git_sha() and proof["planSha"]==digest(plan_path)
         and proof["manifestSha"]==verify_manifest(manifest)==rep_out["manifest_sha256"]
         and proof["seedStart"]==p["training"][rep]==manifest["seed_start"]
         and rep_out["generation"]==422 and rep_out["next_generation"]==423
         and rep_out["accepted_complete_four_win_matches"]==1280
         and rep_out["independent_seed_clusters"]==640
         and rep_out["opponent_roles"]=={"self":8,"historical":8,"fixed":4}
         and rep_out["optimizer_updates_completed"]==40
         and rep_out["champion_promoted"] is False
         and actor==rep_out["single_writer_main_sha256"]==index["main_sha256"]
         and full==index["full_checkpoint_sha256"], "Wrong completed training receipt")
    write_json(result/"training-receipt.json",{**proof,"trainSeed":proof["seedStart"],
        "acceptedMatches":1280,"updates":40,"actorSha":actor,"checkpointSha":full})
    print(json.dumps(read(result/"training-receipt.json"),sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("operation",choices=("seal","arms","receipt"))
    parser.add_argument("plan",type=Path)
    parser.add_argument("args",nargs="+")
    a=parser.parse_args()
    if a.operation=="seal":
        seal(read(a.plan),*map(Path,a.args))
    elif a.operation=="arms":
        arms(a.plan,*map(Path,a.args))
    else:
        b,o,rep,arm=a.args
        receipt(a.plan,Path(b),Path(o),rep,arm)
