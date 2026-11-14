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
6. Add bounded self-hosted training workflow, immutable long-term checkpoint promotion, paired unseen-seed 4-win Match evaluation, Champion selection. Keep untrained baseline separate from champion.

## Metamon teacher

The [Metamon repository](https://github.com/UT-Austin-RPL/metamon) is MIT licensed, with pretrained weights and large Showdown replay/self-play datasets available separately. Before ingesting those assets, verify each dataset/model's own redistribution/use terms and document origin, format, generation and legal-action mapping. Metamon exposes a 13-choice Showdown action space; it does **not** exactly match our battle/mega/selection/draft action vocabulary. Exclude incompatible samples. Teacher availability is optional and must not block R-NaD self-play.

## Compute policy

Training on the user's selected **self-hosted runner** once availability and isolation are verified. GitHub-hosted CPU may run short portability/contract CI. Each workflow must bound job duration, total job count, target matches, aggregate compute, retries and stop flags. One writer per model, immutable evaluation inputs, SHA-checked checkpoint compare-and-swap.
