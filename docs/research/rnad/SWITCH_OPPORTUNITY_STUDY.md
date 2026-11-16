# RNAD-003: opportunity-cost learning for voluntary switches

**Status:** opt-in experimental code and [GitHub-hosted run #43787170970](https://github.com/cueikusuta/p-RL1024/actions/runs/43787170970). No Champion promotion; evaluation must finish before any strength claim. This work uses the existing [Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5) / [Draft PR #6](https://github.com/cueikusuta/p-RL1024/pull/6).

## Hypotheses and rationale

Previously, accepted nonterminal events had only `[0,0]` reward and the winner received `[+1,-1]` at a completed four-win Match. Opportunity cost from switching was **implicit in eventual loss**, not explicitly shaped. The prior `anti-loop-v1` inference guard could block two consecutive switches, but it could not teach the actor that unnecessary switching wastes a chance to attack. The GRU does maintain private actor memory, but no explicitly versioned previous-switch vector currently exists; this experiment changes **the reward objective only**, not the 32-D state features. Future richer history-state experiments must have a separate feature schema.

This trial does **not** apply a fixed punishment to every switch. `scripts/rnad/switch-opportunity.mjs` calculates a small negative reward only after an **authority-accepted voluntary normal switch**, when:
- A currently **legal damaging move** with inferred public type-effectiveness exists and its normalized expected power is above 0.30;
- Current own Pokémon has at least 65% HP and no persistent status;
- Public foe type(s) do not imply a potentially super-effective incoming STAB threat;
- The switch target's projected attack pressure is not more than 0.10 better than the outgoing Pokémon's (an attempt to spare beneficial offensive pivots).

Otherwise no penalty: **mandatory replacements, pivot requests, no available attacks, dangerous type matchups, low HP or unknown opponent type are exempt**. The public-type attack-pressure proxy is a heuristic, *not* a calculated counterfactual turn outcome or exact damage model. Its calibration is a research choice and may need adjustment.

The shaped reward `cost=0.02 * clamp((attackPressure-0.30)/0.70)` is capped to 0.02 per accepted voluntary switch and 0.20 per seat for a full Match. It transfers the exact opposite amount to the opponent: for an actor in seat 0, `[-cost,+cost]`; for seat 1, `[+cost,-cost]`. The genuine terminal four-win reward remains `[+1,-1]` or `[-1,+1]` and is **added** to any final switch cost. This preserves zero-sum accounting but is **a separate shaped game objective**; it is not policy-invariant potential shaping, and wins against the original unshaped baseline must still be measured. Do not misrepresent small penalties as official Pokémon rules.

## Source, data integrity and methods

- The `reward_objective=opportunity-v1` CLI is valid **only with** `--three-phase --features-v2 --switch-guard anti-loop-v1`. Original legacy unguarded trajectories remain terminal-only and byte-schema-compatible.
- `scripts/rnad/export-trajectory.mjs` uses only **authority-accepted** seat-local Battle observations and actions to calculate the reward. The exact Match legal set and the narrower guard-eligible policy support are both preserved, and the NeuRD/target/magnet distributions use the same reduced support as the executed actor. The actor never sees the opponent's private roster, hidden items/moves or raw Match object.
- `scripts/rnad/train.py` validates per-event zero-sum, correct actor sign, voluntary-switch eligibility, capped costs per seat, terminal win sign, model/feature/engine SHA, exact seed progression and full-state checkpoint `reward_objective`. Objective mismatch prevents accidental checkpoint continuation as legacy RNAD. The original `rnad_targets` calculations are unchanged; they already accept per-step zero-sum rewards.
- Tests: `test/rnad-switch-opportunity.test.js` and existing `test/rnad-switch-loop-guard.test.js`, `test/rnad-switch-training-support.test.js` plus regression checks; first corrected [workflow #43787170970](https://github.com/cueikusuta/p-RL1024/actions/runs/43787170970) passed the contracts gate (25/25 tests). The original attempted run #43787068102 failed a regression assertion distinguishing `-0` from `0`, since fixed in the reward module. No learning occurred in that failed attempt.

## Learning and comparison: normal 64, gacha 64→128

Run `rnad-switch-opportunity-i5.yml` downloads immutable `base128`, `base256` and `c005` model JSONs from research run #43757054212 and checks their SHA against its manifest. The original full optimizer/magnet checkpoints stay archived and unmodified. All experimental variants are **new-optimizer warm starts**, not a bit-for-bit resume of the older unshaped run.

| Frozen initialization | New training budget | Periodic snapshot |
| --- | --- | --- |
| normal base128 | 64 real four-win self-play Matches | updated actor at 64 |
| normal base256 | 64 real four-win self-play Matches | updated actor at 64 |
| mutation c005 | **128 real four-win self-play Matches**, retaining optimizer/magnet/RNG across 64→128 | actors at both **64 and 128** |

Every model uses distinct isolated GitHub-hosted Ubuntu CPU work, Node 22.14.0/Python 3.11/torch 2.5.1, 16-Match atomic segments, full SHA-bound checkpoints, 180-min job upper bound, max two learners simultaneously. The same training seed interval begins at `1770001`; c005 continues through `1770128`. Nonzero recorded shaped-switch counts are required to call this an actual opportunity-cost experiment, and zero-sum budgets are recorded per Match.

After all three model jobs finish successfully, evaluate with new seed blocks, always both seats and full four-win Matches:
- new 64 vs its frozen original (128 Matches per model; direct seed `1780001..1780064`), all `anti-loop-v1` greedy;
- each model before and after 64 Matches vs Basic on independent identical seeds `1781001..1781064` (128 Matches per variant);
- **c005 only** additional 128-vs-original, **128-vs-64** direct 128-Match comparisons and c005 128-vs-Basic on the same Basic-seed block. This tests whether extra self-play update budget improves c005 or merely drifts away.

Each evaluation verifies source/engine/model hashes, independent seed+seat coverage, 4-win match completion, guard policy version and nonpromotion. Aggregate report is versioned; statistical improvement must not be claimed from a lone 64-seed exploratory result.

## Safety and rollout

Do not increase the 3,000 decision cap, silently censor legitimate forced/pivot actions, rewrite the original checkpoint/Champion, inflate missing match results or merge Draft PR #6 pending separate ChatGPT independent review and CI. Do not automatically rerun the 128-Match c005 training after a report-only problem: preserve successful `learn-*` artifacts and do an **evaluation-only recovery** if needed, pinned to the original training source SHA. The workflow was changed to **manual-only** after [#43787170970](https://github.com/cueikusuta/p-RL1024/actions/runs/43787170970) queued to prevent duplicate costly jobs on PR synchronization.

## Read-only diagnostic of reward coverage (Issue #5)

`summarizeAcceptedSwitchOpportunities(acceptedRows)` now classifies **authority-accepted voluntary switch rows only** into exclusive reason codes: unknown public opponent types, own low HP/status, public super-effective danger, no legal attack, low attack-pressure, better destination pressure, and match reward cap. It reports counts by seat and the **unchanged** bounded `opportunity-v1` reward. Non-turn mandatory/pivot switches do not enter the denominator. This function cannot certify that arbitrary input JSON is an authority receipt: invoke it only on the trusted Match joint timeline after accepted-receipt, source/engine/seed and four-win completion validation. It does not inspect hidden opponent data and does not alter actor support, reward values, existing checkpoint identity, or archived weights.

This instrumentation **does not retroactively recover reasons** from historical study reports containing only aggregate switch counts: the full accepted observations must exist or a new, explicitly versioned small diagnostic rollout is required. The prior full-state research run #43789378463 remains the numerical reference. Its 4–5% nonzero shaping coverage is a measured fact, not a causal explanation of weak performance. New reason counts must be labeled with their actual source/engine/model SHA, seed and policy version; never silently combine them with earlier training results.

The existing Stage 0 contract workflow explicitly runs `test/rnad-switch-opportunity.test.js`, `test/rnad-switch-loop-guard.test.js` and `test/rnad-switch-training-support.test.js`. These tests check exclusive classifications, two-seat cap accounting, untouched original accepted rows, forced/pivot exclusions and legacy reward behavior. Actual Match-based reason frequencies are a subsequent research measurement, not claimed as completed by these unit tests.
