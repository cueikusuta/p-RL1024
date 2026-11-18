"""Fail-closed synchronization contract for RNAD-001 IntegratedPolicyV1 research.

A generation's 20 actor files must all be accepted and verified before the one
shared learner can run. This module is NOT an actor, learner, or authenticity
proof. Call it only inside a trusted GitHub Actions collector that runs the
authority Match receipt consumer and verifies artifact digests. In particular,
synthetic unit-test fixtures MUST NEVER become training data.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

SCHEMA = "rnad-shared-league-generation-v1"
ACTOR_SCHEMA = "rnad-shared-league-actor-v1"
MATCH_SCHEMA = "rnad-integrated-three-phase-accepted-match-v1"
POLICY_SCHEMA = "rnad-integrated-three-phase-v1"
ACTORS = 20
SEEDS_PER_ACTOR = 32
MATCHES_PER_ACTOR = 2 * SEEDS_PER_ACTOR
MATCHES_PER_GENERATION = ACTORS * MATCHES_PER_ACTOR
OPTIMIZER_BATCH_MATCHES = 32
OPTIMIZER_STEPS = MATCHES_PER_GENERATION // OPTIMIZER_BATCH_MATCHES
PHASES = ("draft", "selection", "battle")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FORBIDDEN = frozenset((
    "teacher", "teacher_utility", "utility_labels", "teacher_logits",
    "opponent_hidden", "opponent_private", "uncommitted_opponent_action",
))


def _need(test, message):
    if not test:
        raise ValueError(message)


def _sha64(s):
    return isinstance(s, str) and HEX64.fullmatch(s) is not None


def _sha40(s):
    return isinstance(s, str) and HEX40.fullmatch(s) is not None


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def actor_role(i, *, has_exploiter):
    _need(type(i) is int and 0 <= i < ACTORS, "Invalid actor ID")
    if i < 8:
        return "self"
    if i < 14:
        return "historical"
    if i < 18:
        return "fixed"
    return "exploiter" if has_exploiter else "historical"


def _opponent(manifest, actor_id):
    pool = manifest["opponents"]
    role = actor_role(actor_id, has_exploiter=bool(pool["exploiter"]))
    if role == "self":
        return role, {"id": manifest["policy_id"] + ":self",
                      "sha256": manifest["main_sha256"]}
    registry = pool[role]
    _need(bool(registry), "Missing immutable opponent pool: " + role)
    return role, registry[(actor_id + manifest["generation"]) % len(registry)]


def verify_manifest(manifest):
    _need(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA,
          "Missing versioned generation contract")
    gen = manifest.get("generation")
    _need(type(gen) is int and gen >= 0, "Invalid generation")
    _need(isinstance(manifest.get("policy_id"), str) and
          1 <= len(manifest["policy_id"]) <= 128, "Invalid integrated policy ID")
    for key in ("policy_manifest_sha256", "main_sha256", "magnet_sha256"):
        _need(_sha64(manifest.get(key)), "Missing model SHA256: " + key)
    _need(_sha40(manifest.get("source_sha")) and
          isinstance(manifest.get("engine_sha"), str) and
          (bool(HEX40.fullmatch(manifest["engine_sha"])) or
           bool(HEX64.fullmatch(manifest["engine_sha"]))),
          "Missing immutable source or engine SHA")
    first = manifest.get("seed_start")
    _need(type(first) is int and 1 <= first <= 0xffffffff -
          ACTORS * SEEDS_PER_ACTOR, "Invalid independent seed window")
    _need(manifest.get("feature_schema") == POLICY_SCHEMA,
          "A legacy single-network feature schema is not an integrated policy")
    opp = manifest.get("opponents")
    _need(isinstance(opp, dict) and set(opp) ==
          {"historical", "fixed", "exploiter"}, "Wrong opponent categories")
    for role, maximum in (("historical", 8), ("fixed", 16), ("exploiter", 8)):
        rows = opp[role]
        _need(isinstance(rows, list) and len(rows) <= maximum and
              (role == "exploiter" or len(rows) > 0), "Missing/oversized frozen pool")
        ids = set()
        for entry in rows:
            _need(isinstance(entry, dict) and
                  set(entry) == {"id", "sha256"} and
                  isinstance(entry["id"], str) and entry["id"] and
                  len(entry["id"]) <= 128 and
                  _sha64(entry["sha256"]) and
                  entry["id"] not in ids, "Invalid frozen opponent registry")
            ids.add(entry["id"])
    _need(manifest.get("optimizer_batch_matches") == OPTIMIZER_BATCH_MATCHES and
          manifest.get("optimizer_steps") == OPTIMIZER_STEPS and
          manifest.get("actor_count") == ACTORS and
          manifest.get("seeds_per_actor") == SEEDS_PER_ACTOR,
          "Wrong 1280 Match / 40 update plan")
    return canonical_hash(manifest)


def _matrix(rows, count, width, *, integer=False):
    return (isinstance(rows, list) and len(rows) == count and
            all(isinstance(row, list) and len(row) == width and
                all(type(x) is int if integer else
                    type(x) in (int, float) and math.isfinite(x)
                    for x in row) for row in rows))


def _verify_events(record):
    from scripts.rnad.train_selection_specialist import TRIPLES
    events = record.get("events")
    rounds = record.get("rounds")
    action_counts = record.get("action_counts")
    _need(type(rounds) is int and 1 <= rounds <= 64 and
          isinstance(action_counts, list) and len(action_counts) == 2 and
          all(type(n) is int and n >= 0 for n in action_counts) and
          isinstance(events, list) and len(events) ==
          4 * rounds + sum(action_counts),
          "Incomplete accepted 3-phase timeline")
    counts = Counter()
    for i, e in enumerate(events):
        _need(isinstance(e, dict) and
              set(e) == {"actor", "phase", "phase_input",
                         "action_index", "behavior_prob",
                         "magnet_log_prob", "policy_pi", "magnet_pi",
                         "predicted_value", "reward"} and
              e.get("phase") in PHASES and type(e.get("actor")) is int and
              e["actor"] in (0, 1), "Illegal/privileged phase or seat")
        phase = e["phase"]
        counts[phase] += 1
        p = e["phase_input"]
        _need(isinstance(p, dict) and not (set(p) & FORBIDDEN),
              "Missing public/own phase actor input")
        if phase == "draft":
            _need(set(p) == {"pool", "codes", "extra"} and
                  isinstance(p["pool"], list) and
                  6 <= len(p["pool"]) <= 100 and
                  _matrix(p["pool"], len(p["pool"]), 44) and
                  isinstance(p["codes"], list) and
                  len(p["codes"]) > 0 and
                  _matrix(p["codes"], len(p["codes"]), 6, integer=True) and
                  _matrix(p["extra"], len(p["codes"]), 4) and
                  all(len(set(code)) == 6 and
                      all(0 <= index < len(p["pool"]) for index in code)
                      for code in p["codes"]),
                  "Invalid authoritative draft public candidate tensor")
            n = len(p["codes"])
        elif phase == "selection":
            _need(set(p) == {"own_cat", "foe_cat", "own_num", "foe_num",
                             "legal"} and
                  _matrix(p["own_cat"], 6, 9, integer=True) and
                  _matrix(p["foe_cat"], 6, 3, integer=True) and
                  _matrix(p["own_num"], 6, 16) and
                  _matrix(p["foe_num"], 6, 6) and
                  p["legal"] == [list(t) for t in TRIPLES],
                  "Invalid canonical public selection 60 candidate contract")
            n = 60
        else:
            _need(set(p) == {"state", "actions"} and
                  isinstance(p["state"], list) and
                  len(p["state"]) == 96 and
                  all(type(x) in (int,float) and math.isfinite(x)
                      for x in p["state"]) and
                  isinstance(p["actions"], list) and p["actions"] and
                  _matrix(p["actions"], len(p["actions"]), 48),
                  "Invalid public v5 battle feature/action tensor")
            n = len(p["actions"])
        index, behavior = e.get("action_index"), e.get("behavior_prob")
        pi, magnet = e.get("policy_pi"), e.get("magnet_pi")
        value, log_magnet = e.get("predicted_value"), e.get("magnet_log_prob")
        _need(type(index) is int and 0 <= index < n and
              isinstance(pi, list) and isinstance(magnet, list) and
              len(pi) == len(magnet) == n and
              all(type(x) in (int,float) and math.isfinite(x) and x > 0
                  for x in pi + magnet) and
              abs(sum(pi)-1) < 1e-5 and abs(sum(magnet)-1) < 1e-5 and
              type(behavior) in (int, float) and math.isfinite(behavior) and
              0 < behavior <= 1 and
              abs(behavior - ((1-.02) * pi[index] + .02/n)) < 1e-6 and
              type(log_magnet) in (int,float) and math.isfinite(log_magnet) and
              abs(log_magnet - math.log(magnet[index])) < 1e-5 and
              type(value) in (int,float) and math.isfinite(value) and
              abs(value) <= 1.00001,
              "Illegal Actor behavior/magnet probability/value")
        reward = e.get("reward")
        _need(isinstance(reward, list) and len(reward) == 2 and
              all(type(x) in (int,float) and math.isfinite(x)
                  for x in reward) and abs(sum(reward)) < 1e-8,
              "Invalid zero-sum payoff")
        expected = ([1,-1] if record["winner"] == 0 else [-1,1]) if (
            i == len(events)-1) else [0,0]
        _need(all(abs(x-y) < 1e-8 for x,y in zip(reward,expected)),
              "Unexpected nonterminal or terminal payoff")
    _need(counts["draft"] == counts["selection"] == 2*rounds and
          counts["battle"] == sum(action_counts),
          "Incomplete accepted phase coverage")
    for seat in (0,1):
        _need(sum(e["actor"] == seat and e["phase"] == "battle"
                  for e in events) == action_counts[seat],
              "Wrong accepted battle counts by seat")

def verify_actor(manifest, payload):
    manifest_sha = verify_manifest(manifest)
    _need(isinstance(payload, dict) and payload.get("schema") == ACTOR_SCHEMA,
          "Wrong actor envelope schema")
    actor_id = payload.get("actor_id")
    _need(type(actor_id) is int and 0 <= actor_id < ACTORS, "Bad actor id")
    role, opponent = _opponent(manifest, actor_id)
    for key in ("generation", "policy_id", "main_sha256", "magnet_sha256",
                "source_sha", "engine_sha"):
        _need(payload.get(key) == manifest[key], "Mixed generation provenance " + key)
    _need(payload.get("manifest_sha256") == manifest_sha and
          payload.get("role") == role and
          payload.get("opponent_id") == opponent["id"] and
          payload.get("opponent_sha256") == opponent["sha256"],
          "Wrong frozen opponent or integrated manifest")
    matches = payload.get("matches")
    _need(isinstance(matches, list) and len(matches) == MATCHES_PER_ACTOR,
          "Missing actor's 64 complete four-win Matches")
    observed = set()
    for row in matches:
        _need(isinstance(row, dict) and row.get("schema") == MATCH_SCHEMA,
              "Legacy or unaccepted Match is not an integrated rollout")
        seed, seat = row.get("seed"), row.get("learner_seat")
        _need(type(seed) is int and type(seat) is int and seat in (0, 1) and
              manifest["seed_start"] + actor_id * SEEDS_PER_ACTOR <= seed <
              manifest["seed_start"] + (actor_id + 1) * SEEDS_PER_ACTOR,
              "Actor seed or learner seat outside isolated range")
        key = (seed, seat)
        _need(key not in observed, "Duplicate seed/seat Match")
        observed.add(key)
        ident = f"g{manifest['generation']}-a{actor_id}-s{seed}-seat{seat}"
        _need(row.get("match_id") == ident and
              row.get("policy_id") == manifest["policy_id"] and
              row.get("main_sha256") == manifest["main_sha256"] and
              row.get("magnet_sha256") == manifest["magnet_sha256"] and
              row.get("opponent_sha256") == opponent["sha256"] and
              row.get("source_sha") == manifest["source_sha"] and
              row.get("engine_sha") == manifest["engine_sha"] and
              row.get("completed") is True and
              row.get("winner") in (0, 1) and
              isinstance(row.get("score"), list) and
              len(row["score"]) == 2 and
              all(type(n) is int and 0 <= n <= 4 for n in row["score"]) and
              row["score"][row["winner"]] == 4 and
              row["score"][1 - row["winner"]] < 4,
              "Incomplete, stale or incorrect authoritative Match metadata")
        _verify_events(row)
    expected = {(seed, seat) for seed in range(
        manifest["seed_start"] + actor_id * SEEDS_PER_ACTOR,
        manifest["seed_start"] + (actor_id + 1) * SEEDS_PER_ACTOR)
        for seat in (0, 1)}
    _need(observed == expected, "Missing paired seats or non-deterministic seeds")
    return {f"g{manifest['generation']}-a{actor_id}-s{seed}-seat{seat}"
            for seed, seat in observed}


def verify_generation(manifest, actor_payloads):
    verify_manifest(manifest)
    _need(isinstance(actor_payloads, list) and len(actor_payloads) == ACTORS,
          "Must have all 20 independently collected actor artifacts")
    ids = set()
    match_ids = set()
    roles = Counter()
    for payload in actor_payloads:
        _need(isinstance(payload, dict) and
              payload.get("actor_id") not in ids,
              "Duplicate/malformed actor artifact")
        ids.add(payload["actor_id"])
        roles[payload.get("role")] += 1
        new = verify_actor(manifest, payload)
        _need(not (new & match_ids), "Cross-actor duplicate match identities")
        match_ids.update(new)
    _need(ids == set(range(ACTORS)) and
          len(match_ids) == MATCHES_PER_GENERATION,
          "Missing actor or incomplete generation")
    expected_roles = {"self": 8, "historical": 6, "fixed": 4, "exploiter": 2}
    if not manifest["opponents"]["exploiter"]:
        expected_roles["historical"] += expected_roles.pop("exploiter")
    _need(dict(roles) == expected_roles, "Opponent category quota mismatch")
    return {
        "generation": manifest["generation"],
        "independent_seed_clusters": ACTORS * SEEDS_PER_ACTOR,
        "accepted_complete_four_win_matches": MATCHES_PER_GENERATION,
        "optimizer_minibatch_matches": OPTIMIZER_BATCH_MATCHES,
        "optimizer_steps_one_epoch": OPTIMIZER_STEPS,
        "manifest_sha256": canonical_hash(manifest),
        "opponent_roles": dict(roles),
    }
