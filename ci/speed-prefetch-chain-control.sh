#!/usr/bin/env bash
set -euo pipefail
role="${1:?}"
[[ "$role" == prepare || "$role" == cleanup ]]
if [[ "$role" == prepare ]]; then
  LEAGUE_WARM_ROUND=1 python3 ci/speed-prefetch-chain-relay.py create
  sleep 2
  # Second draft contains just authenticated readiness, not round-two data.
  # Release gate requires independently verified round-one checkpoint as well.
  LEAGUE_WARM_ROUND=2 python3 ci/speed-prefetch-chain-relay.py create
  LEAGUE_WARM_ROUND=1 python3 ci/speed-prefetch-chain-relay.py await_previous
  LEAGUE_WARM_ROUND=1 python3 ci/speed-prefetch-chain-relay.py cleanup
else
  rc=0
  for round in 1 2; do
    LEAGUE_WARM_ROUND="$round" python3 ci/speed-prefetch-chain-relay.py cleanup || rc=1
  done
  exit "$rc"
fi
