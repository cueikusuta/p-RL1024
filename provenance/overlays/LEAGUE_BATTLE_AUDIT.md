# B+C league battle audit

The existing B+C generation workflow adds a diagnostic sidecar on both initial
and recovery Actors. The input engines, checkpoint and manifest are verified
against the original frozen source before applying the explicitly recorded
audit patch. The learner, handoff, generation deadline and optimizer are not
changed. This change does not start a generation.

Each finished match records the source/engine SHA, overlay hash, Actions run and
attempt, generation, Actor, opponent and model hashes, seed, seat and score.
Records include accepted draft and selection, the selected action and its
policy-eligible alternatives for selection and battle, pre-decision seat
information, post-submit HP/status, and machine-readable resolution facts.
Draft candidate scores are omitted to bound size; their count is included.
A post-submit is not necessarily a resolution: an action may await the other
seat. Join by match/round/request and keep sequence ordering. A positive
critical-damage event excludes Disguise; a miss is not immunity or Protect.
Both seats in these logs are simulated. No browser/player data, raw RNG
state, checkpoint bytes or model weights are included. Reproduce from the
recorded seed and the original, separately pinned runtime and model bootstrap;
retain that bootstrap separately if future replay is needed.

Audit uploads are separate from accepted learning trajectories (14-day
retention; archive important experiments separately). Each audit match has a
bounded record and byte budget and reports seen, recorded and dropped rows.
Each coverage manifest records verified match hashes plus missing/invalid
matches. For a complete generation, expect exactly 20 shard manifests, each
with 64 matches, for a total of 1,280. Prefer the recovered shard only where
the authority accepted its recovered Actor input. A missing entire shard is
not a zero-event shard. Check `recordedMatches == expectedMatches` and
`counts.dropped == 0` before reading event rates as complete. No sidecar enters
the Learner. The standalone parity workflow exercises real completed matches
but never invokes an optimizer or publishes weights.
