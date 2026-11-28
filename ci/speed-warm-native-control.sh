#!/usr/bin/env bash
set -euo pipefail
role="${1:?}"
[[ "$role" == prepare || "$role" == cleanup ]]
rc=0
for round in 1 2; do
  export LEAGUE_WARM_ROUND="$round"
  if [[ "$role" == cleanup ]]; then
    python3 ci/speed-chain-relay.py cleanup || rc=1
  else
    python3 ci/speed-chain-relay.py create
    if (( round < 2 )); then
      python3 ci/speed-chain-relay.py await_previous
      python3 ci/speed-chain-relay.py cleanup
    fi
  fi
done
exit "$rc"
