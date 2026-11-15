# RNAD-003: bounded unattended GitHub-hosted runner queue (2026-11-16)

**User directive:** minimize repeated requests to start research. Use one intentionally bounded research batch rather than repeatedly training identical models or declaring gains from noisy screens.

- Work stays in **Issue #5 / Draft PR #6**, on **GitHub-hosted `ubuntu-latest`** runners only. No self-hosted runners, GITHUB_TOKEN write scope, model promotion, release, or main merge.
- New [`rnad-research-queue-i5.yml`](../../../.github/workflows/rnad-research-queue-i5.yml) registers **36 finite, separately observable research jobs**: **16** four-model-vs-Basic seed shards (4 models × 4 shards), **16** two-mutant-vs-two-standard direct seed shards (2 × 2 × 4), and **4** teacher phase-loss/smoothing configurations. The existing already-running teacher retry #43755359502 and independent study #43755157523 are NOT artificially duplicated.
- Study preparation downloads and verifies immutable model/teacher provenance from successful hosted generation 256 run #43752012559 and 100-candidate screen #43751083532. The four study models are frozen `base128`, `base256`, `c019`, and `c005`. **Each evaluation shard checks out the ORIGINAL v2 training source `f08a9587d6a7033546eab00ffa6479399ca71c2f`** and verifies its engine, model SHA, every accepted completed 4-win Match, both seats, and unique new seed/seat pairs. The new Basic seed window is `1710001-1710256`; direct window is `1720001-1720256`. They are disjoint from the previous holdouts and training, and not used for automatic Champion claims.
- The 4 teacher branches compare `equal` versus `draft-selection` loss weights, each with label smoothing 0 or 0.025, on 24 source-verified AQ teacher Matches per branch, 3 epochs, separate 8-Match imitation holdout and 32 Basic seed pairs. Teacher data training seed window `1730001...`, teacher imitation holdout `1730301...`, and Basic evaluation `1731001...`. This is a **small ablation**, not reliable statistical improvement evidence.
- **Compute cap:** the Basic and direct matrices each use `max-parallel: 2` and per-job 75-minute timeout, while teacher sweeps use `max-parallel: 1` and per-job 140-minute timeout. Jobs have `fail-fast: false` so a local failure does not discard independent valid results. Once the paired shard jobs all succeed, a separate aggregator checks exact seed/seat/model/source coverage and publishes a retained no-Champion research artifact.
- First run [#43756892480](https://github.com/cueikusuta/p-RL1024/actions/runs/43756892480) **started the four teacher jobs**, but source-model preparation halted after all immutable model SHA comparisons when a diagnostic Python `print(json.dumps(...,flush=True))` call used `flush` on the wrong function. This is an orchestration print bug, **not a training or model result**.
- Fixed the two affected print statements and dispatched a second job run **[#43757054212](https://github.com/cueikusuta/p-RL1024/actions/runs/43757054212)** that reuses archived models to queue the **32 missing independent evaluation shards**, skipping teacher jobs so they are not duplicated. The fix was verified because this run's `prepare` job completed successfully and at least the first basic/direct matrix jobs entered execution.
- To prevent costly re-runs on every Draft PR synchronization, the workflow is now `workflow_dispatch` only after those two initial queue starts. A successful matrix job is not proof of model improvement; all final outcomes must be audited and independent ChatGPT Chat review completed before promotion/integration. The first teacher run and second evaluation run are **separate** and each must be checked for actual work and results.

This queue is finite. It does not autonomously invent new training algorithms or silently submit unlimited future jobs. Once the batches finish, the next training decision should use measured winner-vs-winner performance and the pre-registered uncertainty, not mere counts of dispatched jobs.

---

# RNAD-003 follow-up: phase learning, teacher quality and independent A/B (2026-11-16)

Goal #5 / existing Draft PR #6, all GitHub-hosted `ubuntu-latest`, no Champion or main merge. The following is a **research plan/implementation**, not a claim of performance improvement.

1. **Draft/Selection action representation**: `scripts/rnad/feature_diagnostics.py` reads only completed, source-matched public/seat-local v2 teacher examples; counts exact feature-vector collisions and whether the accepted choice shares identical action features with an earlier legal candidate. This diagnoses an information bottleneck before changing the model schema. Earlier v2 vector semantics/checkpoints remain immutable. A future v3 feature schema requires new model weights; existing v2 cannot be relabeled silently.
2. **Teacher / imitation loss**: `train_teacher.py` adds the opt-in `phase_profile=draft-selection` (normalized 2:2:1 Draft:Selection:Battle gradient weights) and `label_smoothing=.025`. The default v2 1:1:1 weights, no smoothing, and existing training behavior remain intact. It can reuse **validated same-source, same-seed authority-accepted teacher traces**, keeping control and intervention identical except for learning loss. Invalid/mismatched source/teacher/schema traces fail closed. The supervised model is **not** R-NaD and cannot be a Champion.
3. **Held-out teacher and Basic outcomes**: [phase-focus study workflow](../../../.github/workflows/rnad-v2-research-next.yml) trains control/focused policies on 64 identical teacher full Matches, tests each against separate unseen 16 teacher Matches for per-phase top1/CE, Basic on the same 64 paired seeds, and direct 64 independent head-to-head seed pairs. Both models are archived with source/model identities, no promotions. Smaller samples are diagnostic.
4. **Independent re-evaluation of normal vs gacha**: The same workflow's separate job restores archived normal v2 generation 128 and 256 (standard [#43752012559](https://github.com/cueikusuta/p-RL1024/actions/runs/43752012559)) and gacha c019/c005 survivors ([#43751083532](https://github.com/cueikusuta/p-RL1024/actions/runs/43751083532)). It binds source, original model and candidate SHA before executing **new** 128 seed-pairs per model vs Basic and new 128 paired seed blocks directly versus each frozen standard model. These seed windows `1601101+` and `1602101+` do not intersect earlier 110..., 130..., 140... research, teacher 151... or candidate selection. Candidate IDs were selected from prior study; no further selection uses these new results as unbiased final Champion evidence.
5. **Avoid overlapping GPU/server and hidden-state side effects**: fixed engine, fixed train code, v2 only, pinned GitHub-hosted CPU; original RNAD-001 and RNAD-002 untouched, no actual game server or private opponent set leakage. Existing saved complete-state optimizer/magnet/RNG checkpoints remain frozen; teacher and genetic candidates use separate initialization, training and seed blocks.

Retain independent reviewer by another ChatGPT Chat session before integration, all current-HEAD contracts green, and full model/score/checkpoint provenance validation. Results remain pending until real training/evaluation steps and archived artifacts are checked. Workflow is manual-only after first bounded launch to avoid costly repeats on routine Draft PR updates.

---

# RNAD-003: ordinary exact-state R-NaD learning continues in parallel (2026-11-16)

**Both research tracks continue independently** under Issue #5 / Draft PR #6:

- **Ordinary (unmutated) R-NaD:** successful generation-128 [GitHub-hosted run #43747882146](https://github.com/cueikusuta/p-RL1024/actions/runs/43747882146) is the immutable origin. Generation-128 directly won 65/128 against the frozen teacher (not significant), and against Basic on the same fresh paired 128 seeds had 130/256 wins compared with the teacher's 129/256 wins. These are diagnostic and not proof of strength improvement.
- The separate [standard continuation run #43752012559](https://github.com/cueikusuta/p-RL1024/actions/runs/43752012559) starts from the **actual generation-128 full optimizer/magnet/scheduler/RNG checkpoint**, not the 100-policy exploration model. It checks out exact original training source `f08a9587d6a7033546eab00ffa6479399ca71c2f`, Python 3.11 / CPU Torch 2.5.1 / Node 22.14.0, original teacher model SHA and strict engine/seed/feature schema checkpoint identity. It resumes 16 accepted complete four-win Matches per batch to 256, retaining checkpoints at 192 and 256.
- Independent evaluation has pre-registered **128 unseen seed pairs** (256 complete Matches) against frozen Basic for both the original 128 and new 256 Actor, plus **128 disjoint seed pairs** (256 Matches) of direct 256 versus 128 with both seats and greedy deployment. All results are checked against the frozen SHA, model bytes, scores, seed/seat completeness, and version. A 55% direct-win diagnostic with no more than 3 percentage-point Basic regression may justify a later 512 generation, but **does not establish statistical significance or trigger Champion promotion**.
- **Evolutionary/gacha exploration** is a separate 100-mutant screen and four fresh 16-Match learners, with entirely disjoint training/evaluation seed windows and new optimizer states. It cannot overwrite or be called a continuation of this standard 128-generation checkpoint.
- Both expensive research workflows are now **manual-only for subsequent starts** to prevent re-running experiments on every Draft PR commit. The already-started hosted runs are independent of later workflow edits; run result and artifact hashes must be verified. Every other QA/CI must remain green, and independent review by a **different ChatGPT Chat session** remains necessary before main merge or Champion promotion.

No result is claimed yet for the new 128→256 standard run until all actual training/evaluation steps complete and the preserved checkpoint bytes are verified.

---

# RNAD-003: population mutation experiment (2026-11-16)

This is an **exploratory evolutionary screen**, not a proven Population Based Training (PBT) continuous optimizer, an exact full-state resume, or a Champion promotion. Source is Issue #5 / Draft PR #6. Only GitHub-hosted Ubuntu CPU may execute research.

- `scripts/rnad/population_probe.py` generates exactly **100 deterministic v2-compatible policies** from the verified generation-128 model artifact of [run #43747882146](https://github.com/cueikusuta/p-RL1024/actions/runs/43747882146): one untouched control, 32 deliberate action-feature coefficient up/down changes, 32 deliberate state-feature coefficient up/down changes, and 35 random small policy-layer perturbations. The original frozen model and optimizer checkpoint bytes are never changed.
- Registered early rejection: 2 independent seed pairs (4 complete four-win Matches) per policy versus Basic, then 8 NEW seed pairs for top 16. All matches execute with both seat assignments and greedy deployment. These are noisy candidate-selection signals, **not significance tests**.
- Only four survivors each get **16 genuine R-NaD accepted full Matches**. These four training jobs start from distinct mutant weights with **new** optimizer/magnet/RNG state. They do **not** claim to resume the original generation-128 optimizer or create a generation-144 model. `train.py` exact state/checkpoint/seed and version contract remains intact; `RNAD-001/002` are untouched.
- Final **32 unseen seed pairs** separately compare each trained candidate against Basic and directly against the original unmutated generation-128 Actor, both seats, greedy. The original Actor is also assessed against Basic with the identical Basic holdout seeds. At most 64 direct games per finalist is a diagnostic comparison, *not* a reliable small-gain Champion gate. When selecting a finalist after seeing holdout results, a further **completely untouched test set** is needed to correct for multiple comparisons.
- The experiment is one bounded, SHA/checkpoint-verified [GitHub-hosted workflow](../../../.github/workflows/rnad-population-study.yml), triggered on introduction of that workflow or manually. It does not automatically rerun for ordinary code changes. Research results must not be claimed until the actual job steps, artifact and score/seed/model identities are verified. A failed or skipped run is not successful training.
- No model auto-promotion, Release upload, `main` merge or conventional PBT optimizer transplant. A full exploit/explore loop with correctly migrated optimizer/magnet/checkpoint state would require separate versioned design and review.

---

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
