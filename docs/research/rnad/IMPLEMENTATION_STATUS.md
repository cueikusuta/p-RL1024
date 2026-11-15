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
