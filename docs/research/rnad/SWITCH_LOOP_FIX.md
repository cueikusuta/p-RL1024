# RNAD-003 — accepted-only switch-loop guard (Issue #5 / Draft PR #6)

## Evidence: what actually failed

The archived c005 vs normal gen128 greedy study stopped on two direct-paired 64-seed blocks with `Decision cap exceeded in round 7`. Do **not** increase the 3,000-decision safety cap or fill in missing wins. On immutable seed `1720101`, [switch trace #43778153444](https://github.com/cueikusuta/p-RL1024/actions/runs/43778153444) reproduced a seventh-round deterministic mutual switch loop. **Each model proposed exactly 1,501 ordinary switches, zero moves and 1,500 immediate reversals** in that round. Both were switching back and forth between two party positions. Basic-vs-Intermediate on the same seed completed in six rounds without an ordinary switch by either agent.

On the single normal seed `1720001` (not representative of the population), c005 proposed **13/60 (21.7%)** ordinary switches and gen128 **7/60 (11.7%)**; Basic **0/23** and Intermediate **1/23 (4.3%)**. The issue affects **both** neural policies, not just c005. The run traced actor proposals, which are diagnostic and are **not independently certifiable as accepted trajectories** in an aborted Match.

## Root-cause hypotheses and alternative interventions

1. **Lack of switch history / nonstationarity in the actor.** The 32-dimensional observation includes the current actor and foe information, but no explicit accepted action cycle history. The GRU implicitly learns a latent state; it did not prevent a self-reinforcing A→B→A transition. Correct longer-term option: v3 battle state features for previous accepted move/switch and bounded consecutive voluntary switches, explicitly source/feature-versioned with fresh training data.
2. **Action-value and imitation bias.** Frozen v2 action logits compare switch and move candidates, with weak/no learned cost for losing a turn or yielding initiative by repeatedly switching. Better training: self-play or imitation with an explicit negative return for unnecessary voluntary switches and reward/counterfactuals for effective pivots. Requires rerunning teacher/rollout experiments; do **not** relabel existing frozen v2 checkpoint or claim that two equal logits are true equivalence.
3. **No runtime hysteresis.** Basic and Intermediate contain ordinary-switch recency checks; frozen neural inference uses plain greedy argmax. A runtime configurable penalty is straightforward but a small finite penalty might not break a pathological high-logit cycle. Increasing the safety cap only hides the problem; always forcing an attack would break legitimate emergency or compulsory swaps.

## Implemented, narrow initial fix

`scripts/rnad/switch-loop-guard.mjs` exports `RnadSwitchLoopGuard` with explicit version **`anti-loop-v1`**. It wraps an original `RNADMixedPolicy` without changing PyTorch weights, exported model bytes, feature schema, GRU computations, game engine, legal action authority or hidden opponent information.

- Only a **normal `requestType=turn`** battle decision with at least one already legal `move` can be filtered.
- If an accepted **ordinary** switch moved A→B and the next ordinary decision starts at B, the immediate B→A switch is excluded **for that one decision** when a legal move exists. Other legal switches remain.
- After **two consecutive accepted ordinary switches** without an intervening accepted move or mandatory replacement, the next ordinary decision must be a legal move when available; an additional switch is not proposed.
- **Mandatory replacement, pivot, no-move positions, and draft/selection** are untouched. A move or forced switch clears the consecutive-switch history. Each Match creates separate seat-scoped guard objects and each new round cannot inherit a previous round's switch history. Legal actions are filtered, never invented.
- The guard state moves forward **only after the authority receipt was consumed** by the trusted collector and the accepted callback fires. Unaccepted proposals or mismatched seat, round, stage and action cannot mutate it.
- This is an **opt-in deployment policy**, not a new trained RNAD model. The old model, old simulation evaluations, saved complete-state optimizer/magnet/RNG, and comparison reports keep their original meaning. The evaluation schema records the guard version explicitly. A guarded behavior distribution must not be silently substituted into an old unguarded R-NaD training trajectory.

## CI and comparison gates

- `test/rnad-switch-loop-guard.test.js` exercises immediate reversal, 3-Pokémon cycles, two consecutive switches, forced and pivot replacements, no-move fallbacks, accepted-only state, seat/round boundaries, and draft/selection.
- [Guard CI](https://github.com/cueikusuta/p-RL1024/actions/workflows/rnad-switch-guard-i5.yml) checks pinned engine snapshots, the new/legacy tests, and real 4-win Match completion for previously failing `1720101` **both seats** with fixed c005 and gen128 models protected by anti-loop-v1. SHA identities and distinct versioned policies are verified. This first gate succeeded in [run #43779235844](https://github.com/cueikusuta/p-RL1024/actions/runs/43779235844).
- That run also schedules a **new-seed A/B**: the exact *same* frozen c005 model with vs without guard (32 paired seeds / 64 Matches); and c005 vs Basic both with and without guard on the same separate 32 paired seeds. This is exploratory only. Judge `complete 4-win Match rate`, voluntary switch ratio, reversals, and direct paired wins before rolling out. Do **not** use one tiny 64-Match comparison as a Champion promotion or proof of superiority.

## Decision gate

The right immediate production safety move is a **small, opt-in, accepted-only guard** as an explicitly distinct model behavior. Do not globally suppress all switches, which would weaken pivot/survival play. After sufficient independent evaluation, a better durable improvement should teach proper switch economics to the model using a new public/seat-local feature schema and properly accepted training traces.

### Completed initial acceptance and archived A/B verification

[Full Match run #43779235844](https://github.com/cueikusuta/p-RL1024/actions/runs/43779235844) completed `guard-contracts-and-repro` successfully: **10/10 focused + legacy tests passed**, and previously failing seed `1720101` finished **two full four-win Matches** on both seats using **c005 guard vs normal-gen128 guard**, with immutable models verified. The later A/B job finished all three real evaluations, but its original report-aggregation step errored by assuming a top-level `seed` field in the Basic evaluator output. Do not confuse that schema discrepancy with an actual Match failure.

To avoid rerunning matches, [archive-only audit run #43779757849](https://github.com/cueikusuta/p-RL1024/actions/runs/43779757849) downloaded the exact archived frozen c005 weights and all three completed paired evaluations. The archive audit **passed** all model SHA, source/engine, both-seat seed sets, win records and four-win terminal checks; its normalized output is retained as `rnad-i5-switch-guard-archive-verification-43779757849`. The source hashes of evaluated policies are distinct from the original training source, but the frozen c005 weight bytes are identical:

| Exact same frozen c005 model | Paired independent result |
| --- | --- |
| Guarded-vs-unguarded direct head-to-head | **33/64 wins = 51.56%** for anti-loop-v1 |
| Unguarded vs Basic, same new seed set | **34/64 wins = 53.13%** |
| Guarded vs Basic, same new seed set | **33/64 wins = 51.56%** |

Direct seed block `1841001..1841032` and Basic seed block `1842001..1842032`, both seats. These exploratory samples have very wide uncertainty, so **no statistically demonstrated strength gain or loss**; one-win difference vs Basic must not be treated as significant. The guard is therefore a **verified loop-stability option, not a new Champion or an unconditional default**.

The audit code is `scripts/rnad/audit_switch_guard_ab.py`. The original heavy workflow's optional top-level Basic seed access was fixed, and that workflow is now manual-only to avoid redundant expensive executions.

As of these checks: **reproduction fixed and archived A/B verified**, no Champion promotion, no merge of Draft PR #6 to `main`. A separate ChatGPT Chat session must complete independent review before final integration.
