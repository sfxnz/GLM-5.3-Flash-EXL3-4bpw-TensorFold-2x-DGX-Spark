#!/usr/bin/env bash
# queue.sh FILE: each line "NAME MODE [VAR=VALUE ...]" boots that config and runs the panel, one at a time.
# Detach: setsid nohup ./queue.sh q1.txt > q1.log 2>&1 < /dev/null &
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while read -r name mode rest; do
  [[ -z $name || $name == \#* ]] && continue
  echo "$(date +%T) boot $name: $rest"
  # shellcheck disable=SC2086
  if ! "$W/boot.sh" "$name" $rest < /dev/null; then echo "$(date +%T) BOOT FAILED $name"; continue; fi
  echo "$(date +%T) panel $name $mode"
  "$W/panel.sh" "$name" "$mode" < /dev/null || echo "$(date +%T) PANEL FAILED $name"
done < "$1"
echo "$(date +%T) QUEUE_DONE $1"
