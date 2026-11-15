# RNAD-001 — execution plan and Stage 0 gate

Status: **P0 contracts in progress; R-NaD learner NOT yet implemented**.
Specification: user-provided `p-RL1024 次世代ポケモンAI開発・詳細実行仕様書 v2.0 (2026-11-15)`.
Goal: [Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5).

## Source of truth and boundaries

- Game rules: [pokemon-battle-factory/develop/SPEC.md](https://github.com/cueikusuta/pokemon-battle-factory/blob/develop/SPEC.md). Source snapshot SHA must accompany every experimental dataset and checkpoint.
- Reuse `src/ai/phase-pipeline.js` and `scripts/dev/simulator/phase-match.js` merged in [PR #4](https://github.com/cueikusuta/p-RL1024/pull/4). Do not duplicate the three-phase orchestration.
- The provenance snapshot, RL-Linear and RL-Deep training paths are immutable for this work. Do not alter [Issue #1](https://github.com/cueikusuta/p-RL1024/issues/1) speed benchmarking or its in-flight [PR #3](https://github.com/cueikusuta/p-RL1024/pull/3).
- Do not assume that the Stage 0 contract files are R-NaD itself. Never substitute a PPO+KL learner for the Perolat et al. NeuRD / V-trace / magnet update.
- Actor sees own observation and accepted public history only. Training-only privileged signals, if adopted for a critic, must flow in a separate and explicitly tested channel.

## Stage 0 contracts introduced in this branch

- `src/ml/legal-action-adapter.js`: `rnadLegalActions(view)` reuses existing authoritative research enumerators for draft, ordered 60-way selection and battle moves/switches. `projectRnadObservation(view,{seat,context})` explicitly projects own/public fields and rejects context from another seat.
- `src/ml/trajectory-collector.js`: one pending proposal at a time; only `commit(acceptedAction)` adds to training data, `reject()` drops a failed proposal, `finish()` is restricted to an authoritative four-win outcome. Per-decision behavior and magnet log-probabilities and experiment identifiers are retained.
- `test/rnad-stage0-contract.test.js`: action count, privacy invariance under opponent unobserved-set substitution, authority-accepted logging and seat reward signs.
- `.github/workflows/rnad-ci.yml`: focused Stage 0 contract check; **not a training workflow**.

## Next actions and acceptance gates

1. Review actor observation fields against current Battle `decisionViewForAI` and Match `view`; add missing *public* features and test noninterference for all hidden set fields, not merely representative keys. Verify phase pipeline snapshots and cross-round historical observations.
2. Implement the exact R-NaD policy dynamics with frozen magnet policy, per-player regularized reward, NeuRD update, V-trace off-policy correction, and replay sequence boundaries. Compare the relevant mathematical terms to the original [DeepNash/R-NaD paper](https://arxiv.org/abs/2206.15378) and [regularization theory](https://arxiv.org/abs/2002.08456).
3. Validate on Rock–Paper–Scissors, Matching Pennies, and a genuinely imperfect-information toy game: action normalization, correct sign for both seats, finite gradients, magnet refresh, nontrivial mixed strategy, exploitability against exact best response. **Do not mark this gate complete on a 50% self-play win rate.**
4. Implement PyTorch model, full resumable checkpoint (policy, critic, optimizer, magnet, scheduler, RNG, game/code/schema SHA, update number, completed match IDs), export/ONNX CPU numerical parity, and a persistent Node inference worker. Never launch Python anew per decision.
5. Wire to authority-accepted three-phase Match actions, ensure both players choose without seeing the opponent's simultaneous action, and run deterministic small training smoke. Snapshot pinning and no-secret tests are mandatory.
6. Add bounded GitHub-hosted training workflow, immutable long-term checkpoint promotion, paired unseen-seed 4-win Match evaluation, Champion selection. Keep untrained baseline separate from champion.

## Metamon teacher

The [Metamon repository](https://github.com/UT-Austin-RPL/metamon) is MIT licensed, with pretrained weights and large Showdown replay/self-play datasets available separately. Before ingesting those assets, verify each dataset/model's own redistribution/use terms and document origin, format, generation and legal-action mapping. Metamon exposes a 13-choice Showdown action space; it does **not** exactly match our battle/mega/selection/draft action vocabulary. Exclude incompatible samples. Teacher availability is optional and must not block R-NaD self-play.

## Compute policy

**Latest explicit user decision (2026-11-15): GitHub-hosted runners ONLY** for RNAD-001, for short tests and actual CPU training. Do not use `self-hosted` labels. Run bounded `ubuntu-latest` jobs, serialise training/checkpoint publication, enforce hard timeout, quota/seed/match limits and no unbounded retries. Each workflow must bound job duration, total job count, target matches, aggregate compute, retries and stop flags. One writer per model, immutable evaluation inputs, SHA-checked checkpoint compare-and-swap.

## 2026-11-15 independent-review remediation checkpoint

Independent review of HEAD `27b4f93` reported four P1 findings. The existing Stage 0 PR was reused. This increment removes untrusted `history` from draft/selection actor projection and all nested `current/completed/handoff` from pipeline context (only `round` remains), fails closed on empty action sets during a required battle decision, and rejects contradictory terminal scores. Regression tests cover these specific cases.

**Still open (P1):** The trajectory collector's public `commit(acceptedAction)` is *not* authority-authenticated. A matching caller-provided action can still be committed without Match acceptance. Do not feed these records to learning or claim accepted-action integrity until wired to a trusted Match acceptance event and tested with rejection, silent fallback and simultaneous decisions. A recursive allowlist for useful public history, authoritative match integration, provenance SHA verification, R-NaD mathematics, toy-game exploitability, full checkpoint and actual training smoke remain unimplemented.

No PPO+KL surrogate was added. No training was launched. This increment is not Stage 0 completion and must remain Draft pending fresh CI and independent re-review.


## 2026-11-15 continuation: new executable Stage 0 evidence

The preceding review note predates these changes. The following code now exists on the same Draft PR (do **not** mark the full Stage 0 complete):

- `scripts/dev/simulator/rnad-match.js` wraps the merged `runPhaseMatch` and keeps one trajectory collector per seat **inside** its trusted runner. An Actor callback receives only a projected observation, legal actions and deterministic seed RNG; not `Match`, `accepted()`, or `collector.commit()`. Each `accepted` callback is invoked by the existing Match runner after the authoritative `confirmTeam`, `confirmSelection` or `battleAction` returns. An invalid Actor candidate fails closed, and the runner publishes complete trajectories only after a completed four-win Match. This is a stage-0 **fixed/untrained policy probe**; it is not R-NaD learning.
- `test/rnad-authority-match.test.js` tests accepted decision counts for both seats over a real one-round Battle, deterministic runs, policy callback isolation and invalid action refusal. `test/rnad-full-match.test.js` requires an actual completed four-win Match with both seat-local trajectories and terminal rewards. These tests are now part of `rnad-ci.yml`.
- `scripts/rnad/toy-rnad.mjs` implements **tabular** zero-sum R-NaD dynamics: fixed magnet policy within each inner epoch, correct own/opponent log-ratio signs, logit-space NeuRD replicator updates, and magnet refresh. `test/rnad-toy.test.js` computes **exact best-response exploitability** on RPS, Matching Pennies, and a two-type Bayesian zero-sum toy in which player 1 cannot observe player 0's type. This is an initial mathematics gate; it does not validate the DeepNash two-player V-trace estimator, neural NeuRD gradients, policy learning under function approximation or full-game Nash convergence.
- `scripts/rnad/checkpoint.py` stores PyTorch Policy, Value, magnet, Optimizer, Scheduler, Python/Torch/CUDA RNG and match-boundary provenance in a SHA-256-addressed immutable `.pt`, with an atomic `latest.json` index, expected-prior-SHA check, safe `weights_only=True` restore and mismatch rejection. `test/rnad_checkpoint_test.py` exercises optimizer and RNG restore, partial-match refusal, compare-and-swap rejection and corruption rejection on a dummy PyTorch network. In the available local Python 3.13 / torch 2.10.0+cpu environment these two tests passed. GitHub-hosted Node CI **only compiles Python syntax**, not PyTorch runtime; end-to-end learner+simulator checkpoint parity remains unverified.
- `src/ml/legal-action-adapter.js` now recognizes actual `Battle.phase='turn'/'pivot'/'replacement'` views as battle decisions and recursively filters public side effects. Arbitrary nested pipeline handoffs/history remain intentionally blocked pending a safe cross-round contract.

### Remaining quality gates

- The bare `createRnadTrajectoryCollector().commit()` API can still be misused by other trusted local code; the integrated runner hides it from the Actor. The accepted-action integrity of *published training data*, including authority error paths, needs independent review. The P1-3 concern cannot be considered fully closed before that review.
- Node and Python currently have **no** persistent ONNX model inference or R-NaD learner connection. Real accepted trajectory records are not proof of learning.
- DeepNash two-player V-trace, neural NeuRD, frozen magnet refresh, PyTorch numerical gradients, PyTorch/ONNX parity, and full-game deterministic checkpoint continuation must be implemented and tested.
- Training workflows, GitHub-hosted runner availability, durable releases, teacher import and Champion evaluation remain outstanding.
- The original provenance manifest and pre-existing RL models/experiments were not edited. Review the latest head and required CI before any merge.

## Runner policy override (2026-11-15)

The explicit current requirement is **GitHub-hosted GitHub Actions runners (`ubuntu-latest`) for all RNAD-001 training and evaluation**. Older self-hosted recommendations in the Issue description, earlier PR comments or handoff are superseded. The RNAD training workflow must be opt-in, bounded, use the GitHub-hosted runner, enforce single-writer concurrency and non-persistent artifacts must not be treated as the durable checkpoint store. Avoid unrelated changes to original RL workflows.
