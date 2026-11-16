# Issue #5 — frozen standard and gacha actors: guarded new self-play learning

**Experimental start (2026-11-16):** [GitHub-hosted CPU Actions #43781276450](https://github.com/cueikusuta/p-RL1024/actions/runs/43781276450). The workflow event's PR HEAD is `275cd6cbf3b0149ab0307fb5a93b91cbab0d3f94`, but default `pull_request` checkout uses GitHub's synthetic merge commit. The actual training report independently recorded `source_sha=571bac81d62373ab063c2e1508b0647122786e80` for base128. Source SHA must be taken from the accepted-run model report, not confused with the PR HEAD. Editing the workflow dispatch trigger later does not modify the already-running checkout.

## Scope, objectives and interpretation

Use the successfully archived `rnad-i5-queue-models-43757054212` files as **immutable starting weights** for three independent groups:

| Group | Frozen actor initialization | Reason |
| --- | --- | --- |
| normal 128 | `base128.json` | Reliable prior reference |
| normal 256 | `base256.json` | Prior standard continuation |
| gacha c005 | `c005.json` | Most interesting mutation from the 100-candidate screen |

All three use `rnad-three-phase-features-v2`, **the exact same 64 new self-play seed IDs `1750001..1750064`** and the opt-in **`anti-loop-v1`** deployment policy for *both seats*. Learning runs on separate ephemeral Linux `ubuntu-latest` runners with `max-parallel:2`, 180-minute per training job timeout, four 16-Match checkpoint segments, pinned Node 22.14.0 / Python 3.11 / CPU torch 2.5.1, no external model service, writes only to run-specific artifacts. The original archived weights, original R-NaD optimizer state and frozen AQ-Base are untouched.

**Important:** this is a **new optimizer/magnet/value-target initialization from a frozen model JSON** rather than a bit-for-bit continuation of the old 128/256 optimizer checkpoint. The game environment and source SHA differ from earlier research; old and new Match samples MUST NOT be combined as one continuous training run or existing checkpoints reinterpreted. Learned model provenance contains `original_model`, original SHA-256, new SHA-256, training source SHA, engine SHA, policy version, completed Match IDs and checkpoint SHA.

## Mathematically aligned switch filtering

The old `anti-loop-v1` runtime wrapper chose among a **filtered subset** of the Match's full legal candidates. If it is applied naively to R-NaD training, the stored behavior/reference log-probabilities are calculated over a different support than Python's tensor of all legal actions. This can corrupt NeuRD regularization and V-trace despite apparently valid wins.

The guarded online learner now uses an *explicit*, separately labelled `policy_eligible_actions` subset on every accepted Battle event. Each record still preserves the Match-authority `legal_actions` complete set and one-use accepted-action receipt. The collector validates every eligible action belongs to authority's set, checks the accepted action is in its support, rejects duplicates and missing or forged actions, and does not record rejected proposals. For guarded event export, the Python actor/target/magnet all re-evaluate **exactly the reduced support**: same model features, action index, prior/reference normalization and on-policy mixture as used at execution. For unguarded learning, the old event shape is maintained. The same guards keep per-seat, accepted-only action memory (not opponent private information).

New model/policy ID: `RNAD-003-3phase-v2-anti-loop-v1`. The extra `switch_guard` checkpoint expectation means that a guarded checkpoint cannot resume as unguarded or vice versa. The existing full-state SHA, source SHA, engine snapshot, seed, model feature schema, optimizer/scheduler state, RNG and accepted Match ID gates continue to apply.

## Training and evaluation contracts

- `scripts/rnad/export-trajectory.mjs` flag `--switch-guard anti-loop-v1` yields a distinct accepted-trajectory schema. `scripts/rnad/train.py` flag `--switch-guard anti-loop-v1` enables the new online self-play experiment. Unspecified flag stays `none` and must be backward-compatible.
- `test/rnad-switch-training-support.test.js` checks exact legal authority vs eligible support and rejected forged subsets. Existing direct, collector, phase and guard unit tests remain enabled.
- The [workflow](../../../.github/workflows/rnad-guard-warmstarts-i5.yml) first gates all three training jobs behind Node contract tests and `python -m py_compile`. Each model then must produce **64 complete 4-win self-play Matches**, 64 optimizer updates, positive accepted actions, correct two-seat policy history, finite loss and hashed complete-state checkpoint at each boundary.
- **Independent held-out comparison**, only after all three training jobs succeed: per model, trained guarded vs original guarded (seed `1760001..1760064`, both seats: 128 Matches); trained guarded vs frozen Basic, and original guarded vs frozen Basic (same seed window `1761001..1761064`, each 128 Matches). Every result must match original/new model SHA, exact seed+seat coverage, match terminal 4-win scores, inference mode `greedy` and anti-loop policy version, and must not claim Champion.
- Final same-run aggregate lists three models with same training and evaluation seed windows. A gain must be supported by independent paired uncertainty, not a single small pilot. No champion promotion, PR merge, new Issue, or production deployment.

## Current status / continuation

The initial research run was submitted and its **`contracts` job succeeded**, after which the three model training jobs entered the GitHub Actions queue. **Results and performance superiority have NOT yet been verified as completed** at the time of writing. Use Actions run #43781276450 and this document to continue work; do not refer to these jobs as done if they are queued, running, failed or skipped. If one model fails due to a code defect, preserve valid independently completed artifacts and fix the narrow cause, retaining the original seed/instance provenance. Do not rerun all expensive successful jobs solely to update report presentation. Review must be performed by a separate ChatGPT Chat session before integrating Draft PR #6 to `main`.
