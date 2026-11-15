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
