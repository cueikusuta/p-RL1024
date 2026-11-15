# RNAD-003: semantic v2 / strong AQ teacher / greedy deployment (2026-11-16 verified handoff)

> This entry supersedes earlier "Draft/Selection not learned" statements below. All experiments are **GitHub-hosted `ubuntu-latest`**, never self-hosted. Existing Issue #5 and Draft PR #6 remain open. No main merge or Champion promotion. Sampled-policy and greedy-policy results are NOT interchangeable; likewise Basic, Stage0, and AQ-Base are different opponents.

## Empirical findings (authoritative complete four-win Matches)

- New three-phase neural Actor exists: Draft + Selection + Battle are authority-accepted, seat-projected, trained with separate `rnad-three-phase-features-v1/v2` model contracts; the old battle-only `RNAD-001` remains distinct.
- [Hosted 128-Match public AQ imitation](https://github.com/cueikusuta/p-RL1024/actions/runs/43737981487): 15,222 accepted actions, 4 supervised epochs; **v1 heldout imitation accuracy** Draft 2.2%, Selection 7.1%, Battle 55.0%. Sampled v1 against Basic **3/256**. Merely adding 16 R-NaD Match updates did not establish superiority.
- [Original v2 greedy evaluation](https://github.com/cueikusuta/p-RL1024/actions/runs/43744199284): independent paired 128 seeds (256 complete Matches) against Basic: **27/256 sampled vs 123/256 greedy** from v2 model. The corresponding v1 model was **3/256 sampled vs 27/256 greedy**. [Independent seed-block paired difference verification](https://github.com/cueikusuta/p-RL1024/actions/runs/43747506291) proved a substantial v2 sampled-vs-greedy increase on those frozen seeds; NOT proof against AQ-Base.
- [Phase-balanced semantic v2 teacher study](https://github.com/cueikusuta/p-RL1024/actions/runs/43743931532): 128 frozen-AQ teacher Matches and 4 epochs; 15,222 examples per epoch, phase-balanced CE+value, separate frozen-v2 schema. Heldout teacher agreement Draft 4.40%, Selection 6.59%, Battle 69.51%. Sampled policy vs Basic 41/256; phase losses / public features still require improvement.
- [Real v2 teacher-initialized 16-Match R-NaD](https://github.com/cueikusuta/p-RL1024/actions/runs/43746142851): frozen teacher on fresh paired seeds vs Basic **136/256** (greedy), RNAD16 **134/256** (greedy), RNAD16 directly vs teacher **66/128** (51.56%; not statistically significant). Both training and evaluation used real completed Match authority; exact full-state optimizer/magnet/RNG checkpoint retained. The study marks **candidate_for_128_training_study=true**, which is only a cheap non-regression screen, not a strength confirmation.
- [Exact 16→128 continuation](https://github.com/cueikusuta/p-RL1024/actions/runs/43747882146): fixed two separate CI failures before resumed training: app-scoped `gh api --jq .conclusion` returned an empty field, silently causing previous `success` run to skip training; then checkout-to-source restored runner's default Python and PyTorch 2.5.1 was unavailable. Patched to validate real archived model/checkpoint bytes as the authority, then pin Python 3.11 / Node 22.14. **Continuation is running, NOT yet proven complete at time of this note.** Source SHA is archive-derived and checked before `train(...resume=True)`; zero manual optimizer reset.
- [v2 semantic encoder tests](https://github.com/cueikusuta/p-RL1024/actions/runs/43747882164) and [paired greedy comparison tests](https://github.com/cueikusuta/p-RL1024/actions/runs/43747882169) passed on the tested HEAD.

## Important cautions and next gates

1. **Never claim 128 Match completion from an Actions `success` when its expensive training steps are `skipped`.** Verify each named step, generation-64/-128 checkpoint SHA and archived model `source_sha`. The old skip-only success [#43747506276](https://github.com/cueikusuta/p-RL1024/actions/runs/43747506276) is NOT a trained 128 generation.
2. Compare actual v2 generation-128 vs frozen teacher **on the same held-out seed-pairs and greedily executed moves**; require full scores, SHA, both seats and disjoint train seeds. Its 128-game direct assessment is diagnostic only. For formal small-difference research, 2,048 full Matches (1,024 independent seed blocks), conservative paired CIs, and additional independent opponents are recommended.
3. Strengthen Draft/Selection: 4%-7% exact-action agreement remains poor, though top-1 action accuracy over very large legal candidate sets is not itself directly comparable with Battle accuracy. Study supervised phase-balanced losses, structured/categorical team composition and exposure to frozen stronger teachers. V2 feature IDs are versioned; NEVER change their meaning in-place for existing checkpoints (introduce v3).
4. **Independent review by a DIFFERENT ChatGPT Chat session**, including public/private information, R-NaD/NeuRD/V-trace formulae, all-phase accepted events, greedy deployment vs training-policy interpretation, and checkpoint SHA identity remains mandatory. Do not use same implementer as independent reviewer.
5. No Champion without real >=1,000 independent paired full Matches, required confidence bounds, opponent diversity, latest-HEAD CI and independent review. Archived workflow artifacts expire; not durable production-model storage.

---

# RNAD-001 latest verified status (2026-11-15)

> This section supersedes older chronological notes below. **Only GitHub-hosted `ubuntu-latest` runners** are permitted for RNAD-001; no self-hosted runners. Existing Issue #5 / Draft PR #6 continue; **no Champion promoted or main merge**.

## Executed evidence and interpretation

| Research stage | Real GitHub-hosted evidence | Result |
|---|---|---|
| 16-match learner smoke | [#43681087240](https://github.com/cueikusuta/p-RL1024/actions/runs/43681087240) | 16 complete four-win Matches, 2,629 accepted battle decisions, 16 optimizer updates |
| Pure 256-match RNAD selfplay | [#43681846580](https://github.com/cueikusuta/p-RL1024/actions/runs/43681846580) | 256 Matches, 39,984 accepted decisions, 256 optimizer updates; frozen Stage0 opponent holdout: initial 4/64 wins, trained 9/64; not significant |
| Fixed-Stage0 imitation teacher, then RNAD256 | Archived [#43683127479](https://github.com/cueikusuta/p-RL1024/actions/runs/43683127479) | Source-bound optional *local fixed policy*, **not Metamon** |
| 128 independent seed-pair A/B versus Stage0 | [#43697864312](https://github.com/cueikusuta/p-RL1024/actions/runs/43697864312) | Teacher 123/256 (48.05%), teacher+256 RNAD 115/256 (44.92%), paired gain −3.125 pp, conservative 95% bounds −27.13…+20.88 pp |
| **Direct teacher vs RNAD256** | Same [#43697864312](https://github.com/cueikusuta/p-RL1024/actions/runs/43697864312) | RNAD 69/128 (53.91%) on 64 unseen seed pairs with both seats; 95% paired bounds 36.93%–70.88%, not significant |
| **Low eta/slow magnet ablation** | [#43698077819](https://github.com/cueikusuta/p-RL1024/actions/runs/43698077819) | `eta=.05`, `magnet_period=16`, 64 complete Matches, 6,892 accepted decisions, 64 updates; 32/64 versus teacher (50%) and 27/64 versus earlier 256 RNAD (42.19%), neither significant |

All figures reflect their specific seeds, model opponents and revisions. **Do not rank models from their cross-opponent raw win rates.** Direct head-to-head and identical-seed baselines are deliberately distinct experiments. A few dozen paired seeds cannot establish a Champion.

Current implementation additions: strict `scripts/rnad/compare_warmstart_eval.mjs` validation (score, round, source/model SHA, seed window, both seats), direct neural-to-neural `scripts/rnad/evaluate-direct.mjs`, bounded `research_eta_ablation.py` with a GitHub-hosted-only workflow, and `train.py` full-checkpoint identity binding of `regularization_eta` and `magnet_period`. The latest-HEAD CI must re-verify both ordinary and failed-hyperparameter resume.

**Remaining Goal #5 gates:** learn Draft and Selection (currently Basic), improve categorical Pokémon/action encodings and actual win rate, independently verify original-paper mathematical details, valid external teacher licensing, reliable 2,048/16,384 Match longer-term continuation and durable storage, independent 1,000+ paired Champion confidence bounds, and separate latest-HEAD ChatGPT Chat independent review/P0-P1 clearance. Existing Draft PR #6 stays unmerged until all mandatory Stage gates for integration pass.

---

## Historical notes (chronological; older “pending” claims are superseded above)
# RNAD-001 implementation status — 2026-11-15 (Issue #5 / Draft PR #6)

**Required execution environment: GitHub-hosted GitHub Actions `ubuntu-latest` ONLY. Do not use any self-hosted runner.** This is the latest explicit user instruction, overriding all earlier recommendations.

## Implemented and checked into Draft PR

- Authoritative three-phase Match integration: battle Actor projection/legality; accepted action records bound to single-use Match-minted opaque receipts; completion bound to the real four-win outcome. The producer exports a chronological two-seat timeline only after a complete Match. Generic collectors cannot self-certify receipt authenticity.
- Code provenance: Node checks `source_sha` against the current `git rev-parse HEAD` and `engine_snapshot_sha` against the immutable `provenance/manifest.json`; the Python consumer checks both again.
- Baseline fixed untrained policy: successful Stage0 Node-only four-win smoke was previously recorded on CI #43656271010. **That CI predates the newer authenticated receipt and neural code, so it does not validate them.**
- Mathematical tabular toy: RPS, Matching Pennies and a hidden-type Bayesian matrix game with exact best-response exploitability. Previously verified on CI #43656271010.
- `scripts/rnad/two_player.py`: two-seat V-trace `v-hat`/`Q-hat`, cross-player `r-hat`/`xi` terms, magnet-corrected reward and NeuRD all-legal-logit surrogate. Reference: Perolat et al., arXiv:2206.15378, supplementary p.29 Eq. (1)-(5), p.30 policy/value gradients. Unit tests cover signs, importance ratios, off-seat terminal rewards, bad action probabilities and gradient finiteness. **Latest CI runtime evidence still needed.**
- `scripts/rnad/neural.py`: 32-state / 16-action numeric feature encoder, GRU-128, legal-candidate Policy and Value; ONNX CPU export and Python/Node numerical parity checks.
- `scripts/rnad/infer.mjs`: persistent synchronous pure-JS forward of exported PyTorch GRU weights and a sampled mixed legal policy; no Python process per action. Node remains the authoritative engine.
- `scripts/rnad/export-trajectory.mjs`: completed four-win Match with both accepted seats, immutable SHA, chronological event order and terminal signed reward.
- `scripts/rnad/train.py`: Python CPU self-play rollout, PyTorch V-trace/NeuRD updates, frozen magnet between periodic refreshes, target critic EMA, loss finite checks and one full-checkpoint write per completed Match.
- `scripts/rnad/verify_resume.py`: real Match continuous-two vs one-then-resume-one comparison including trajectory identity and weight/magnet hashes.
- `scripts/rnad/evaluate.mjs`: held-out paired full four-win Match assessment, swapped seats, Wilson 95% confidence interval; deliberately no automatic Champion promotion.
- `.github/workflows/rnad-ci.yml` and `.github/workflows/rnad-train.yml`: dedicated **GitHub-hosted** bounded jobs, Python CPU/ONNX tests, learning smoke, restore and paired evaluation. Training job is serialized and read-only; artifacts have finite retention and are **not durable Champion storage**.

## Not yet demonstrated / mandatory gates before completion

1. Latest neural PyTorch tests (including ONNX + Node parity) pass on GitHub-hosted Ubuntu.
2. Latest runtime accepted receipt and SHA boundary tests pass on GitHub-hosted Ubuntu, including a real complete four-win Match after the changes.
3. Actual RNAD-001 learner 1–16 full-Match smoke runs and its logged normalized Policy/Value/parameter changes are verified.
4. Exact checkpoint resume parity in real game runs is demonstrated with the latest unchanged commit and model.
5. Held-out evaluation with independent seed and swapped seats runs. Full Champion promotion additionally requires **at least 1,000 independent paired full Match evaluations**, 95% CI, and independent review.
6. Neural NeuRD saturation-threshold interpretation and simultaneous Battle resolution chronology need an independent mathematical review against the original equations. The tabular toy tests alone are insufficient proof.
7. Immutable LONG-TERM checkpoint store, continuation on multiple GitHub-hosted jobs, and 256/2,048-scale results are not yet proven. GitHub artifact expiry prevents classifying it as the durable checkpoint store.
8. Independent ChatGPT Chat-session review of the **LATEST** PR head, P0/P1 remediation and successful required CI; only then may main integration be considered.

## How to reproduce on a GitHub-hosted runner

1. In GitHub Actions, run `RNAD-001 Stage 0 contracts` on the existing PR #6 branch.
2. Run `RNAD-001 GitHub-hosted CPU train and paired evaluation` with `matches=1`, `seed=975021`, then inspect the artifact's report and checksum.
3. Run `python -m scripts.rnad.verify_resume --seed 975023` on that *same commit* to verify uninterrupted vs resumed model weights.
4. Do not start large-scale training or promote any Champion until the above and independent-review gates pass.

**No reported workflow success from an older head may be used as proof of the current code.** No PPO+KL surrogate is registered as R-NaD. Prior RL-Linear/RL-Deep and Issue #1 performance branch are unchanged.

## 2026-11-15 hosted-only execution extension (pending runner validation)

User requirement is absolute for this research series: **GitHub-hosted `ubuntu-latest` only**. The old self-hosted recommendation is obsolete.

New work on existing PR #6:

- `scripts/rnad/export_checkpoint.py`: validate SHA, source/engine and completed-match metadata, load PyTorch weights, export frozen evaluation Actor without changing the trained policy.
- `scripts/rnad/release-checkpoint.sh`: versioned GitHub Release asset transport of immutable SHA-256-named completed-match checkpoints. Upload is append-only (no clobber), requires the unique `rnad-001-<16-hex-source>-s<seed>` research tag and a separately serialized authenticated workflow. Download validates archive entry names; the Python loader independently verifies the SHA. GitHub Actions artifacts are only a 30-day recovery cache, not the checkpoint authority.
- `.github/workflows/rnad-scale.yml`: **manual, main-only, post-independent-review** GitHub-hosted 16-Match batches (up to 4 per run, 2,048 overall), one writer globally, source SHA pinned to a reviewed main ancestor, 330-minute limit, immutable GitHub Release checkpoint per completed batch and recovery artifact on failure. This is scale **infrastructure, not evidence that 256 or 2,048 training matches ran**.
- `.github/workflows/rnad-eval.yml`: manual main-only GitHub-hosted read/verify checkpoint and up to 50 independent seed pairs with both candidate seats, immutable per-model/release report upload.
- `scripts/rnad/aggregate-eval.mjs`: batch-report aggregation with exact model/engine/opponent SHA, duplicate/unpaired seed rejection, >=1,000 full matches and an **independent paired-seed Hoeffding 95% confidence lower bound** above 0.5 as *eligibility for independent Champion review only*. No auto-promotion.
- `src/ml/legal-action-adapter.js`: public opponent reveal and scalar fields fail closed if supplied nested arbitrary objects. This is defense in depth; actor observations must still be generated from the seat-scoped Match authority.
- `test/rnad-receipts.test.js`: real Match verifies unforgeable/nonreusable receipt identities and opposite-seat mismatch. `test/rnad-evaluation.test.js` covers duplicate held-out seeds and insufficient strength.
- `scripts/rnad/two_player.py`: local NeuRD proposed-logit clipping gate now explicitly documented as an **approximate neural coordinate update**, not exact nonlinear Adam parameter projection. Independent paper-level review is mandatory.

**Known verified evidence:** Previous old HEAD `7a50e358cdd09d1f9b969119e8b42900cbff3f03` RNAD Stage0 Node CI run `43656271010` succeeded. In the current session a pure JavaScript V8 smoke directly evaluated the checked-in `toy-rnad.mjs` mathematical module: RPS exact exploitability 1.76 → 0.0000313, Matching Pennies 1.68 → 0.000000172, hidden-type Bayesian game 0.8 → 0.0000725; checked-in `aggregate-eval.mjs` rejected duplicate seeds and gave CI95 [0.4393,0.5607] for 1,000 50%-win paired outcomes. A zero-weight GRU forward gate matched an independent Torch CPU calculation to <2e-8. **These checks are not substitutes for current-HEAD GitHub Actions PyTorch/ONNX/full-match CI.**

**Unverified / blocked:** As of this checkpoint, GitHub Actions had not produced a new run after 00:03 UTC despite subsequent PR commits, so no actual hosted neural learner/ONNX parity/checkpoint-resume/smoke/256/2048 results can honestly be claimed. The connected GitHub tool has no new-workflow-dispatch action. A browser attempt to start the GitHub-hosted workflow reached an unsigned-out GitHub session and could not dispatch; user GitHub sign-in/Actions permission is required to execute it. Do **not** imply that any 1–16 match neural training or 1,000-match Champion evaluation ran. **PR remains Draft; main merge forbidden until latest-HEAD CI, separate ChatGPT Chat independent review, and remediation of any P0/P1.**

## 2026-11-15 later continuation: main reconciliation and hosted CI trigger limitation

- Latest main speed work was merged **into the PR branch only**, preserving the already-integrated model/benchmark changes, with an explicit README conflict resolution. The PR was reported by the GitHub API as `mergeable: true`. This was **not a main merge of RNAD**; retain Draft.
- The NeuRD policy loss now multiplies its advantage by the clipped two-player V-trace `is_c` weight and the learner passes the V-trace coefficient, with an explicit regression test that gradient scales at `is_c = 0, 0.5, 1`. Local PyTorch CPU math, checkpoint and numerical gate checks passed. GitHub-hosted current-HEAD CI is still necessary.
- Added a neural training gradient-flow test with two-seat signed terminal reward to ensure model parameters actually change while frozen magnet remains fixed.
- **IMPORTANT GitHub Actions limitation:** `workflow_dispatch` manual launch via the Actions UI requires the workflow YAML to exist on the default branch. The new RNAD workflows currently exist **only in Draft PR #6**, so a logged-in user still might not see a `Run workflow` button before merge. Do **not** prematurely merge the PR to work around this. Instead, with an authenticated GitHub user session, push an ordinary new commit to the existing PR branch (e.g., a documentation-only touch) to generate the normal `pull_request.synchronize` event and run the new RNAD CI and 1-match smoke on GitHub-hosted `ubuntu-latest`. The current connected GitHub App edits did not produce a new workflow run at the latest HEAD. Alternatively use a separately reviewed GitHub Actions CI orchestration path that does not require putting unfinished RNAD code on main. Older run #43656271010 is NOT evidence for newer HEAD.
- **Human login required:** browser automation's GitHub profile was unauthenticated. Do not request passwords in chat. The user should authenticate in the supported browser profile; after successful user login, trigger the above PR synchronize, inspect CI/Artifacts, repair failures and repeat as necessary. Keep large-scale main-only workflow untriggered until CI and independent ChatGPT Chat review clear all P0/P1.
