"""Render accepted AQ draft picks and optional student counterfactual predictions.

Reads opt-in --draft-review teacher traces. Student argmax is evaluated on
TEACHER-REACHED states, never mislabelled as an accepted student Match action.
No hidden opponent sets are read. Results are diagnostic, not win claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from scripts.rnad.neural import (
    RNADNetwork, load_exported_json, THREE_PHASE_FEATURE_SCHEMA_V4,
    THREE_PHASE_FEATURE_SCHEMA_V5, STATE_FEATURES_V3, ACTION_FEATURES_V3,
    STATE_FEATURES_V5, ACTION_FEATURES_V5)
from scripts.rnad.train import current_git_sha, frozen_engine_sha

def _candidate(view, index):
    codes = view["candidate_codes"][index]
    roster = view["roster"]
    if len(codes) != 6 or len(set(codes)) != 6:
        raise ValueError("Malformed authority six-Pokemon candidate")
    mons = [roster[i] for i in codes]
    return {
        "candidate_index": index,
        "pokemon": [p["species"] for p in mons],
        "set_ids": [p["set_id"] for p in mons],
        "cost": sum(p["credit_cost"] for p in mons),
        "roles": [p["roles"] for p in mons],
    }

def _validate(record, event, view):
    if (view.get("schema") != "rnad-accepted-draft-pick-review-v1"
        or event.get("phase") != "draft"
        or not isinstance(view.get("credits"), int)
        or not 0 <= view["credits"] <= 500
        or not isinstance(view.get("roster"), list)
        or not 6 <= len(view["roster"]) <= 100
        or any(not isinstance(p.get("species"), str) for p in view["roster"])
        or not isinstance(view.get("candidate_codes"), list)
        or len(view["candidate_codes"]) != len(event["actions"])
        or not isinstance(event.get("candidate_utilities"), list)
        or len(event["candidate_utilities"]) != len(event["actions"])
        or view.get("accepted_choice_index") != event.get("choice")
        or event["choice"] >= len(event["actions"])
        or any(len(c) != 6 or any(type(i) is not int or not 0 <= i < len(view["roster"]) for i in c)
               for c in view["candidate_codes"])
        or any(not math.isfinite(v) for v in event["candidate_utilities"])
        or view["accepted_roster_codes"] != view["candidate_codes"][event["choice"]]
        or record["score"][record["winner"]] != 4):
        raise ValueError("Invalid or inconsistent authority draft review")
    best = max(event["candidate_utilities"])
    if best-event["candidate_utilities"][event["choice"]] > 1e-5:
        raise ValueError("Accepted AQ utility mismatch")
    if _candidate(view, event["choice"])["cost"] != view["accepted_spent"]:
        raise ValueError("Draft budget review mismatch")

def generate(traces, model_path=None, max_picks=40):
    if not 1 <= max_picks <= 1000:
        raise ValueError("Review bounds exceeded")
    records = [json.loads(Path(path).read_text()) for path in traces]
    if not records or len({r.get("seed") for r in records}) != len(records):
        raise ValueError("Missing or duplicate source seed")
    schemas = {r.get("feature_schema") for r in records}
    if len(schemas) != 1 or schemas.pop() not in (
            THREE_PHASE_FEATURE_SCHEMA_V4, THREE_PHASE_FEATURE_SCHEMA_V5):
        raise ValueError("Review requires one consistent v4/v5 public schema")
    schema = records[0]["feature_schema"]
    source = current_git_sha()
    engine = frozen_engine_sha()
    if any(r.get("source_sha") != source or r.get("engine_sha") != engine
           or r.get("completed") is not True or
           r.get("teacher") != "rnad-aq-public-strategy-selection-v1"
           for r in records):
        raise ValueError("Unverified source/engine/teacher or incomplete Match")
    model_sha = None
    net = None
    if model_path is not None:
        if schema == THREE_PHASE_FEATURE_SCHEMA_V5:
            net = RNADNetwork(state_features=STATE_FEATURES_V5,
                              action_features=ACTION_FEATURES_V5)
        else:
            net = RNADNetwork(state_features=STATE_FEATURES_V3,
                              action_features=ACTION_FEATURES_V3)
        model_sha = load_exported_json(net, Path(model_path),
                                      expected_feature_schema=schema)
        net.eval()
    picks = []
    with torch.no_grad():
        for record in records:
            memory = [net.initial_memory() for _ in range(2)] if net else []
            counts = [0, 0]
            for event in record["examples"]:
                seat = event["actor"]
                if seat not in (0, 1):
                    raise ValueError("Invalid accepted seat")
                prediction = None
                if net:
                    state = torch.tensor([event["state"]], dtype=torch.float32)
                    actions = torch.tensor([event["actions"]], dtype=torch.float32)
                    logits, _, hidden = net(state, actions, memory[seat])
                    memory[seat] = hidden.detach()
                    if event["phase"] == "draft":
                        logp = torch.log_softmax(logits[0], dim=0)
                        probs = logp.exp()
                        order = torch.argsort(logits[0], descending=True, stable=True)
                        prediction = (probs, order)
                if event["phase"] != "draft":
                    continue
                view = event.get("draft_review")
                if not isinstance(view, dict):
                    raise ValueError("Missing opt-in authority draft review")
                _validate(record, event, view)
                counts[seat] += 1
                chosen = event["choice"]
                utility = event["candidate_utilities"]
                ordered_utility = sorted(range(len(utility)),
                                         key=lambda i: (-utility[i], i))
                teacher_top = [
                    {**_candidate(view, i), "utility": utility[i]}
                    for i in ordered_utility[:5]]
                item = {
                    "seed": record["seed"], "seat": seat,
                    "draft_decision_number": counts[seat],
                    "match_score": record["score"], "match_winner": record["winner"],
                    "credits": view["credits"],
                    "carried": [p for p in view["roster"] if p["is_carry"]],
                    "offered": [p for p in view["roster"] if not p["is_carry"]],
                    "legal_candidate_count": len(view["candidate_codes"]),
                    "teacher_accepted": {
                        **_candidate(view, chosen), "utility": utility[chosen],
                        "teacher_rank": ordered_utility.index(chosen)+1},
                    "teacher_top5": teacher_top,
                    "student_prediction_only": None,
                }
                if prediction:
                    probs, order = prediction
                    student_index = int(order[0])
                    best, bottom = max(utility), min(utility)
                    gap = (best - utility[student_index])/(best-bottom) if best-bottom>1e-12 else 0.0
                    item["student_prediction_only"] = {
                        "on_teacher_reached_state": True,
                        "NOT_authority_accepted_student_pick": True,
                        "greedy": {**_candidate(view, student_index),
                                   "probability": float(probs[student_index]),
                                   "teacher_utility": utility[student_index],
                                   "teacher_utility_regret": gap,
                                   "teacher_utility_rank": ordered_utility.index(student_index)+1},
                        "top5": [
                            {**_candidate(view, int(i)), "probability": float(probs[int(i)]),
                             "teacher_utility": utility[int(i)]}
                            for i in order[:5]],
                        "teacher_choice_probability": float(probs[chosen]),
                        "matches_teacher_top1": student_index == chosen,
                        "teacher_in_student_top10": chosen in [int(i) for i in order[:10]],
                    }
                picks.append(item)
    return {
        "schema": "rnad-draft-pick-human-audit-v1",
        "source_sha": source, "engine_sha": engine,
        "feature_schema": schema, "student_model_sha256": model_sha,
        "teacher_id": "rnad-aq-public-strategy-selection-v1",
        "teacher_matches": len(records),
        "total_draft_decisions": len(picks),
        "shown_draft_decisions": min(max_picks, len(picks)),
        "student_is_counterfactual": True,
        "is_champion": False,
        "picks": picks[:max_picks],
    }

def render_markdown(report):
    lines = [
        "# R-NaD ドラフトピック実例・検証ログ",
        "",
        "**注意:** 教師ピックは権威Matchに受理された実際の選択です。"
        "学生の予測は教師が訪れた局面でのgreedy反実仮想であり、"
        "学生が実戦で受理した行動や勝利とは異なります。",
        "",
        f"- Teacher: \`{report['teacher_id']}\`",
        f"- Model SHA: \`{report['student_model_sha256'] or 'none'}\`",
        f"- Source/engine: \`{report['source_sha']}\` / \`{report['engine_sha']}\`",
        f"- 完走教師Match: {report['teacher_matches']}、全ドラフト判断: {report['total_draft_decisions']}、表示: {report['shown_draft_decisions']}",
        "",
    ]
    def mons(c):
        return " / ".join(c["pokemon"]) + f"（費用 {c['cost']}）"
    for p in report["picks"]:
        lines.extend([
            f"## seed {p['seed']} / 席 {p['seat']} / ドラフト {p['draft_decision_number']}",
            "",
            f"- 試合結果: {p['match_score']}（勝者 席{p['match_winner']}）",
            f"- クレジット: {p['credits']}、合法候補: {p['legal_candidate_count']}",
            "- 継続所持: " + (", ".join(x["species"] for x in p["carried"]) or "なし"),
            "- 提示候補: " + (", ".join(f"{x['species']}({x['credit_cost']})"
                                for x in p["offered"]) or "なし"),
            "- **教師が実際に採用:** " + mons(p["teacher_accepted"]),
            "",
            "| 教師順位 | 6体の組合せ | 教師スコア |",
            "|---:|---|---:|",
        ])
        for k, c in enumerate(p["teacher_top5"], 1):
            lines.append(f"| {k} | {mons(c)} | {c['utility']:.4f} |")
        student = p["student_prediction_only"]
        if student:
            top = student["greedy"]
            lines.extend([
                "",
                "- **学生greedy予測（教師局面・未受理）:** " + mons(top),
                f"- 教師一致: {student['matches_teacher_top1']}、"
                f"教師utility順位: {top['teacher_utility_rank']}、"
                f"正規化regret: {top['teacher_utility_regret']:.4f}",
                "",
                "| 学生順位 | 6体の組合せ | モデル確率 | 教師スコア |",
                "|---:|---|---:|---:|",
            ])
            for k, c in enumerate(student["top5"], 1):
                lines.append(
                    f"| {k} | {mons(c)} | {c['probability']:.3%} | {c['teacher_utility']:.4f} |")
        lines.append("")
    return "\n".join(lines)+"\n"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trace", nargs="+", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--max-picks", type=int, default=40)
    args=p.parse_args()
    folder=Path(args.out)
    folder.mkdir(parents=True, exist_ok=False)
    result=generate(args.trace, args.model, args.max_picks)
    (folder/"draft-picks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (folder/"DRAFT_PICKS.md").write_text(render_markdown(result))
    print(json.dumps({k:v for k,v in result.items() if k!="picks"},
                     ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
