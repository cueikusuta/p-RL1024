"""I5 selection research A/B: teacher-score factorization vs raw-token Attention.

ONLY PUBLIC/OWN actor tensor enters each network. The teacher core/lead/utility
scores exist exclusively in the supervised training loss and evaluation.
The old selection specialist is a fixed baseline, not used for warm starting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.rnad.train_selection_specialist import (
    TRIPLES, TRIO_IDS, LEAD_IDS, KEYS,
    SelectionSpecialist, assess, feature_sufficiency_audit,
)
from scripts.rnad.selection_specialist_fresh_holdout import from_archive, REFERENCE_SHA

SCHEMA = "rnad-i5-selection-public-rich-and-teacher-targets-v1"
ACTOR_SCHEMA = "rnad-i5-selection-public-actor-tensors-v1"
VOCAB_SCHEMA = "rnad-i5-selection-rich-vocabulary-v1"
SOURCE_TEACHER = "rnad-aq-public-strategy-selection-v1"
MATCHES_TRAIN, MATCHES_DEV = 256, 128
SEED_TRAIN, SEED_DEV = 2471001, 2472001
WIDTH, EPOCHS, BATCH = 64, 10, 32
TRIO_INDEX = torch.tensor([KEYS.index(tuple(sorted(t))) for t in TRIPLES],
                          dtype=torch.long)
TRIO_ROWS = torch.tensor(KEYS, dtype=torch.long)
LEAD_INDEX = torch.tensor(LEAD_IDS, dtype=torch.long)
KEYS_VOCAB = ("species", "names", "abilities", "items", "moves", "types")


def read_events(folder, start, n, *, expected_sha=None):
    root = Path(folder)
    events = []
    vocab = None
    engine = None
    source = None
    for seed in range(start, start + n):
        raw = json.loads((root / f"selection-{seed}.json").read_bytes())
        if (raw.get("schema") != SCHEMA or raw.get("seed") != seed or
            raw.get("teacher") != SOURCE_TEACHER or
            raw.get("completed") is not True or
            raw.get("score", [])[raw["winner"]] != 4 or
            len(raw.get("examples", [])) != 2 * raw.get("rounds", -1) or
            len(raw["examples"]) < 1):
            raise ValueError("Invalid complete accepted rich selection Match " + str(seed))
        sha=raw.get("source_sha")
        if not isinstance(sha,str) or len(sha)!=40 or any(
                x not in "0123456789abcdef" for x in sha):
            raise ValueError("Invalid immutable research source SHA")
        if source is None: source=sha
        if sha!=source:
            raise ValueError("Mixed rich teacher export sources")
        if engine is None: engine = raw.get("engine_sha")
        if raw.get("engine_sha") != engine:
            raise ValueError("Mixed game-engine snapshots")
        if expected_sha is not None and raw.get("source_sha") != expected_sha:
            raise ValueError("Mismatched teacher source")
        meta = raw.get("vocab")
        if (not isinstance(meta, dict) or meta.get("schema") != VOCAB_SCHEMA or
            not isinstance(meta.get("sha256"), str) or len(meta["sha256"]) != 64 or
            set(meta.get("sizes", {})) != set(KEYS_VOCAB)):
            raise ValueError("Invalid immutable vocabulary")
        if vocab is None: vocab = meta
        if meta != vocab: raise ValueError("Mixed rich actor vocabularies")
        for event in raw["examples"]:
            validate_event(event, vocab)
            event["trace_source_sha"]=sha  # metadata only; never an actor input
            events.append((seed, event))
    if len(events) < 2 * n:
        raise ValueError("Missing accepted selection decisions")
    return events, vocab, engine


def validate_event(e, vocab):
    a = e.get("actor")
    if (not isinstance(a, dict) or a.get("schema") != ACTOR_SCHEMA or
        a.get("vocab_sha256") != vocab["sha256"] or
        len(a.get("legal", [])) != 60 or
        a["legal"] != [list(t) for t in TRIPLES] or
        len(a.get("own_cats", [])) != 6 or
        len(a.get("foe_cats", [])) != 6 or
        len(a.get("own_numeric", [])) != 6 or
        len(a.get("foe_numeric", [])) != 6):
        raise ValueError("Wrong legal/actor selection tensor")
    for rows, size in ((a["own_cats"], 9), (a["foe_cats"], 3),
                       (a["own_numeric"], 16), (a["foe_numeric"], 6)):
        if any(len(row) != size for row in rows):
            raise ValueError("Wrong rich token width")
    own_fields = ("species", "abilities", "items", "moves", "moves",
                  "moves", "moves", "types", "types")
    foe_fields = ("names", "types", "types")
    for rows, fields in ((a["own_cats"], own_fields),
                         (a["foe_cats"], foe_fields)):
        for row in rows:
            for idx, kind in zip(row, fields):
                if type(idx) is not int or not 0 <= idx < vocab["sizes"][kind]:
                    raise ValueError("Invalid bounded public/own categorical index")
    for rows in (a["own_numeric"], a["foe_numeric"]):
        if any(type(v) not in (float, int) or not math.isfinite(v) or
               not 0 <= v <= 2 for row in rows for v in row):
            raise ValueError("Invalid bounded public/own numeric")
    targets = e.get("targets")
    if (not isinstance(targets, dict) or
        len(targets.get("core", [])) != 20 or
        len(targets.get("lead", [])) != 6 or
        len(targets.get("utilities", [])) != 60 or
        type(e.get("choice")) is not int or not 0 <= e["choice"] < 60 or
        len(e.get("state", [])) != 96 or
        len(e.get("actions", [])) != 60 or
        any(len(row) != 48 for row in e["actions"])):
        raise ValueError("Missing teacher training-only component targets")
    if not all(math.isfinite(v) for seq in (
        targets["core"], targets["lead"], targets["utilities"],
        e["state"], *e["actions"]) for v in seq):
        raise ValueError("Non-finite selection example")
    for i, (trio, lead) in enumerate(zip(TRIO_IDS, LEAD_IDS)):
        if abs(targets["core"][trio] + targets["lead"][lead] -
               targets["utilities"][i]) > 1e-5:
            raise ValueError("Decomposition cannot explain actual teacher labels")
    if max(targets["utilities"]) - targets["utilities"][e["choice"]] > 1e-5:
        raise ValueError("Selected choice did not maximize teacher scores")


class PublicEmbed(nn.Module):
    """Embeddings are learned on fixed, public catalog-derived vocabularies."""
    def __init__(self, sizes):
        super().__init__()
        self.species = nn.Embedding(sizes["species"], 12)
        self.names = nn.Embedding(sizes["names"], 12)
        self.abilities = nn.Embedding(sizes["abilities"], 12)
        self.items = nn.Embedding(sizes["items"], 12)
        self.moves = nn.Embedding(sizes["moves"], 12)
        self.types = nn.Embedding(sizes["types"], 8)
        self.own_proj = nn.Sequential(nn.Linear(72, WIDTH), nn.ReLU())
        self.foe_proj = nn.Sequential(nn.Linear(26, WIDTH), nn.ReLU())

    def forward(self, a):
        own_cat, foe_cat = a["own_cat"], a["foe_cat"]
        own = torch.cat([
            self.species(own_cat[..., 0]),
            self.abilities(own_cat[..., 1]),
            self.items(own_cat[..., 2]),
            self.moves(own_cat[..., 3:7]).mean(dim=-2),
            self.types(own_cat[..., 7:9]).mean(dim=-2),
            a["own_num"],
        ], dim=-1)
        foe = torch.cat([
            self.names(foe_cat[..., 0]),
            self.types(foe_cat[..., 1:3]).mean(dim=-2),
            a["foe_num"],
        ], dim=-1)
        return self.own_proj(own), self.foe_proj(foe)


class DecomposedTeacher(nn.Module):
    """A: directly supervise 20 team scores and 6 lead scores separately."""
    def __init__(self, vocab):
        super().__init__()
        self.embed = PublicEmbed(vocab["sizes"])
        self.pair = nn.Linear(WIDTH, WIDTH)
        self.opponent = nn.Linear(WIDTH, WIDTH)
        self.core = nn.Sequential(nn.Linear(WIDTH * 4, 96), nn.ReLU(),
                                  nn.Linear(96, 1))
        self.lead = nn.Sequential(nn.Linear(WIDTH * 3, 96), nn.ReLU(),
                                  nn.Linear(96, 1))
        self.register_buffer("trio_index", TRIO_INDEX)
        self.register_buffer("lead_index", LEAD_INDEX)
        self.register_buffer("trio_rows", TRIO_ROWS)

    def forward(self, a):
        own, foes = self.embed(a)
        pair = F.relu(self.pair(own)[:, :, None, :] +
                      self.opponent(foes)[:, None, :, :])
        enriched = own + pair.mean(dim=2)
        chosen = enriched[:, self.trio_rows, :]  # [batch,20,3,64]
        pooled = chosen.mean(dim=2)
        highest = chosen.max(dim=2).values
        global_team = enriched.mean(dim=1)
        opponents = foes.mean(dim=1)
        core = self.core(torch.cat([
            pooled, highest,
            global_team[:, None, :].expand(-1, 20, -1),
            opponents[:, None, :].expand(-1, 20, -1)], dim=-1)).squeeze(-1)
        lead = self.lead(torch.cat([
            enriched,
            global_team[:, None, :].expand(-1, 6, -1),
            opponents[:, None, :].expand(-1, 6, -1)], dim=-1)).squeeze(-1)
        return core[:, self.trio_index] + lead[:, self.lead_index], core, lead


class DetailedAttention(nn.Module):
    """B: 12 raw public/own Pokémon tokens with self-attention, score 60 trios."""
    def __init__(self, vocab):
        super().__init__()
        self.embed = PublicEmbed(vocab["sizes"])
        self.kind = nn.Embedding(2, WIDTH)
        block = nn.TransformerEncoderLayer(
            d_model=WIDTH, nhead=4, dim_feedforward=128,
            dropout=0., batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(block, num_layers=2,
                                                  enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.Linear(WIDTH * 4, 128), nn.ReLU(),
                                    nn.Linear(128, 1))
        self.register_buffer("choices", torch.tensor(TRIPLES, dtype=torch.long))

    def forward(self, a):
        own, foe = self.embed(a)
        tokens = torch.cat([own + self.kind.weight[0],
                            foe + self.kind.weight[1]], dim=1)
        encoded = self.transformer(tokens)
        own_tokens = encoded[:, :6, :]
        combo = own_tokens[:, self.choices, :] # [B,60,3,64]
        lead = combo[:, :, 0, :]
        tri = combo.mean(dim=2)
        global_context = encoded.mean(dim=1)[:, None, :].expand(-1, 60, -1)
        paired = lead * tri
        logits = self.scorer(torch.cat([lead, tri, paired, global_context],
                                       dim=-1)).squeeze(-1)
        return logits, None, None


def pack(rows, *, training=False):
    a = [x[1]["actor"] for x in rows]
    device = "cpu"
    inputs = {
        "own_cat": torch.tensor([x["own_cats"] for x in a], dtype=torch.long, device=device),
        "foe_cat": torch.tensor([x["foe_cats"] for x in a], dtype=torch.long, device=device),
        "own_num": torch.tensor([x["own_numeric"] for x in a], dtype=torch.float32, device=device),
        "foe_num": torch.tensor([x["foe_numeric"] for x in a], dtype=torch.float32, device=device),
    }
    if not training:
        return inputs
    teacher = [x[1]["targets"] for x in rows]
    util = torch.tensor([x["utilities"] for x in teacher], dtype=torch.float32)
    core = torch.tensor([x["core"] for x in teacher], dtype=torch.float32)
    lead = torch.tensor([x["lead"] for x in teacher], dtype=torch.float32)
    choices = torch.tensor([x[1]["choice"] for x in rows], dtype=torch.long)
    # One scale shared by the two target components: normalized core+lead
    # reconstructs the true teacher candidate utility up to a common offset.
    scale = util.std(dim=1, unbiased=False).clamp(min=1.)
    core = (core-core.mean(dim=1,keepdim=True))/scale[:,None]
    lead = (lead-lead.mean(dim=1,keepdim=True))/scale[:,None]
    util = (util-util.mean(dim=1,keepdim=True))/scale[:,None]
    return inputs,core,lead,util,choices


def objective(net, rows, kind):
    inputs, core, lead, util, choice = pack(rows, training=True)
    logits, predicted_core, predicted_lead = net(inputs)
    if kind == "A":
        # Regress teacher intermediate components; labels stay out of actor inputs.
        loss = (F.mse_loss(predicted_core,core) +
                F.mse_loss(predicted_lead,lead) +
                0.5 * F.mse_loss(logits,util) +
                0.5 * F.cross_entropy(logits,choice))
    elif kind == "B":
        # Direct legal-choice policy, plus a mild teacher rank target.
        target = F.softmax(util, dim=-1)
        loss = F.cross_entropy(logits,choice) + 0.2 * (
            -(target*F.log_softmax(logits,dim=-1)).sum(dim=1).mean())
    else:
        raise ValueError("Unknown A/B research arm")
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite supervised selection loss")
    return loss


def train(net, events, kind):
    if kind not in ("A", "B") or not events:
        raise ValueError("Missing controlled A/B training data")
    torch.manual_seed(90251)
    rng = random.Random(90251)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0007, weight_decay=0.0001)
    curves = []
    for epoch in range(EPOCHS):
        net.train()
        order = list(range(len(events)))
        rng.shuffle(order)
        total = 0.
        for off in range(0, len(order), BATCH):
            batch=[events[i] for i in order[off:off+BATCH]]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(net,batch,kind)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(batch)
        curves.append(total/len(events))
        print(json.dumps({"arm":kind,"epoch":epoch+1,"train_loss":curves[-1]}),
              flush=True)
    return net.eval(),curves


def evaluated_logits(net, rows):
    net.eval()
    result=[]
    with torch.no_grad():
        for off in range(0,len(rows),BATCH):
            inputs=pack(rows[off:off+BATCH])
            logits,*_=net(inputs)
            if not bool(torch.isfinite(logits).all()) or logits.shape[1]!=60:
                raise ValueError("Invalid finite A/B selection logits")
            result.extend(logits.tolist())
    return result


def summarize(logits, events):
    if len(logits)!=len(events):
        raise ValueError("Incomplete heldout predictions")
    stats=defaultdict(list)
    for (seed,event), values in zip(events,logits):
        scores=torch.tensor(values,dtype=torch.float32)
        sample=assess(scores,event_as_old(event))
        predicted=int(scores.argmax())
        trio=int(TRIO_IDS[predicted]==TRIO_IDS[event["choice"]])
        stats[seed].append({**sample,"trio":trio,
            "lead_given_trio":int(trio and LEAD_IDS[predicted]==LEAD_IDS[event["choice"]])})
    metric={}
    by_seed={}
    for seed,rows in stats.items():
        by_seed[seed]=sum(x["top1"] for x in rows)/len(rows)
    allrows=[r for rows in stats.values() for r in rows]
    n=len(allrows)
    for k in ("top1","top10","regret","trio"):
        metric[k]=sum(row[k] for row in allrows)/n
    numtrio=sum(x["trio"] for x in allrows)
    metric["lead_accuracy_given_correct_trio"]=(
        sum(x["lead_given_trio"] for x in allrows)/numtrio
        if numtrio else None)
    metric["decisions"]=n
    metric["complete_match_seed_clusters"]=len(stats)
    return metric,by_seed


def event_as_old(event):
    # Teacher utilities are only consumed by the training loss/evaluator,
    # never passed to the public actor tensors.
    return {"choice":event["choice"],
            "candidate_utilities":event["targets"]["utilities"]}


def baseline_scores(baseline, events):
    logits=[]
    baseline.eval()
    with torch.no_grad():
        for offset in range(0,len(events),BATCH):
            rows=events[offset:offset+BATCH]
            states=torch.tensor([e["state"] for _,e in rows],dtype=torch.float32)
            actions=torch.tensor([e["actions"] for _,e in rows],dtype=torch.float32)
            scores=baseline(states,actions)
            logits.extend(scores.tolist())
    return logits


def clustered_delta(base,can,*,seed=9131):
    if set(base)!=set(can) or len(base)<128:
        raise ValueError("Insufficient independent paired teacher Match seeds")
    values=[can[s]-base[s] for s in sorted(base)]
    mean=sum(values)/len(values)
    rng=random.Random(seed)
    n=len(values)
    samples=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                   for _ in range(5000))
    return {"paired_cluster_mean_top1_delta":mean,
            "approximate_95pct_two_sided_ci":[samples[125],samples[4874]],
            "independent_seed_clusters":n}


def run(train_dir, dev_dir, reference, out):
    torch.set_num_threads(1)
    train_rows,vocab,engine=read_events(train_dir,SEED_TRAIN,MATCHES_TRAIN)
    dev_rows,dev_vocab,dev_engine=read_events(dev_dir,SEED_DEV,MATCHES_DEV)
    if (vocab!=dev_vocab or engine!=dev_engine or
        train_rows[0][1]["trace_source_sha"]!=dev_rows[0][1]["trace_source_sha"]):
        raise ValueError("Unmatched train/dev encoder, source or engine")
    if set(seed for seed,_ in train_rows) & set(seed for seed,_ in dev_rows):
        raise ValueError("Reused train and dev Match seed")
    dir=Path(out)
    dir.mkdir(parents=True,exist_ok=False)
    baseline=from_archive(reference,REFERENCE_SHA)
    baseline_metrics,baseline_seeds=summarize(
        baseline_scores(baseline,dev_rows),dev_rows)
    comparisons={"structured-hard-frozen":{"dev":baseline_metrics,
                                            "selected":False}}
    for name, cls in (("A",DecomposedTeacher),("B",DetailedAttention)):
        torch.manual_seed(90251)
        net=cls(vocab)
        model,curve=train(net,train_rows,name)
        metrics,by_seed=summarize(evaluated_logits(model,dev_rows),dev_rows)
        filename=f"selection-{name}.pt"
        torch.save(model.state_dict(),dir/filename)
        digest=hashlib.sha256((dir/filename).read_bytes()).hexdigest()
        comparisons[name]={"dev":metrics,"train_epoch_losses":curve,
                           "paired_vs_frozen_on_reused_dev":
                               clustered_delta(baseline_seeds,by_seed),
                           "weights":filename,"sha256":digest}
        print(json.dumps({"arm":name,"dev":metrics}),flush=True)
    # The previous 128 Match block is DEVELOPMENT data. Pick ONE arm before
    # reading any new confirmation Match labels; do not declare a victory.
    selected=min(("A","B"),key=lambda k:(
        -comparisons[k]["dev"]["top1"], comparisons[k]["dev"]["regret"], k))
    comparisons[selected]["selected"]=True
    result={
        "schema":"rnad-i5-selection-AB-exploratory-v1",
        "train_seed":SEED_TRAIN,"train_complete_matches":MATCHES_TRAIN,
        "development_seed":SEED_DEV,"development_matches":MATCHES_DEV,
        "development_block_previously_used":True,
        "teacher_source_sha":train_rows[0][1]["trace_source_sha"],
        "engine_sha":engine,"vocab":vocab,
        "teacher":"rnad-aq-public-strategy-selection-v1",
        "teacher_components_training_only":True,
        "actor_has_no_teacher_utility_or_hidden_opponent":True,
        "models":comparisons,
        "preselected_for_fresh_confirmation":selected,
        "confirmation_result":None,
        "league_training_allowed":False,
        "champion_promoted":False,
        "strength_established":False,
    }
    (dir/"selection-AB-development.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=[
        "# 選出A（教師分解）／B（詳細情報Attention）探索比較",
        "",
        "256受理完全Match学習／既使用128受理完全Matchによる開発評価。",
        "Top1は教師の選択との一致であり、実戦勝率・最終未使用確認ではない。",
        "",
        "| モデル | Top1 | Top10 | 3体一致 | Utility regret ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for k,v in comparisons.items():
        m=v["dev"]
        lines.append(f"| {k} | {m['top1']:.2%} | {m['top10']:.2%} | "
                     f"{m['trio']:.2%} | {m['regret']:.4f} |")
    lines.extend(["",f"新規データを見ずに固定した1候補: {selected}",
                  "3フェーズ統合の選出80%・リーグ開始ゲートには使用できない。"])
    (dir/"SELECTION_AB_DEVELOPMENT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-dir",required=True)
    p.add_argument("--dev-dir",required=True)
    p.add_argument("--reference",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.train_dir,a.dev_dir,a.reference,a.out)


if __name__=="__main__":
    main()
