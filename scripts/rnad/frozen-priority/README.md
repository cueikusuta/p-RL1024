# Frozen public-information priority evaluator

For comparison only, not a training Actor or a production deployment. Run the
installer in a disposable library source checkout pinned to 6a11eee, after
verifying the snapshot with scripts/verify-snapshot.mjs. Copy this directory
into scripts/rnad first. Install takes the JSON overlay path from this PR. It
creates new files with validated source/result hashes and never edits training
files. Then run:

    node scripts/rnad/frozen-priority/evaluate.mjs ACTOR_DIR MODE OPPONENT SEED SEAT

Use off (unchanged), shadow (record a recommendation, but preserve action), or
public-lethal-priority-v1. The explicit active profile can change at most one
accepted command per match, only for a legal priority move that estimates a
KO against the publicly revealed threat. Missing evidence falls back to the
original policy. The opponent can be aq-base or strategy-tactical. No raw
opponent private state, checkpoint, model weights or optimizer is exported.
All recommendations are accepted-action scoped.

The small Node CI runs isolated frozen evaluations against a pinned G421
checkpoint while its original Actions artifact is available. When it expires,
the same exact archived checkpoint can be supplied from the private backup for
local parity using verify-smoke.py. This PR does not enable the profile in the
league's stochastic learning trajectories.
