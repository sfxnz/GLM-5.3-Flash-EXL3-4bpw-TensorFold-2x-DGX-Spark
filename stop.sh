#!/usr/bin/env bash
# Stops this recipe's pair: rank 0 (HTTP) first, whose SIGTERM ends the session, then rank 1 on the worker. Touches only
# containers that carry LABEL and the NAME_PREFIX0/1 names from recipe.conf. Safe to run when nothing runs.
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=recipe.conf
source "$DIR/recipe.conf"

# stop_ours NAME LABEL: stop (60 s grace) and remove NAME, only when it carries LABEL. Runs here and, over ssh, on the worker.
stop_ours() {
  # A here-string, not a pipe: under pipefail an early-exiting grep -q can fail the pipe.
  if ! grep -Fxq "$1" <<<"$(docker ps -a --filter "label=$2" --format '{{.Names}}')"; then
    echo "no $1 on $(hostname -s)"
    return 0
  fi
  docker stop -t 60 "$1" >/dev/null
  docker rm "$1" >/dev/null
  echo "stopped and removed $1 on $(hostname -s)"
}

stop_ours "${NAME_PREFIX}0" "$LABEL"
script="$(declare -f stop_ours); stop_ours $(printf '%q ' "${NAME_PREFIX}1" "$LABEL")"
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "bash -c $(printf %q "$script")"; then
  echo "error: could not stop ${NAME_PREFIX}1 on $WORKER over ssh; it may still run there" >&2
  exit 1
fi
