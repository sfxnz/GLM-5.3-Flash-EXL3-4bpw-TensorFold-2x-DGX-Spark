#!/usr/bin/env bash
# boot.sh NAME [VAR=VALUE ...]: restart the pair with the overrides and record the boot under runs/NAME/.
set -euo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=~/projects/ai-lab/recipes/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-SPARK
NAME=${1:?name}; shift
set -- "${@//\~/ }"  # a queue line is split on spaces: ~ stands for a space inside one value
OUT=$W/runs/$NAME; mkdir -p "$OUT"
printf '%s\n' "$@" > "$W/configs/$NAME.env"
# Drain before stopping: never stop a rank mid-request.
for _ in $(seq 60); do
  r=$(curl -s -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | awk '/^tensorfold:requests_running /{print $2}') || true
  [[ -z ${r:-} || $r == 0 ]] && break; sleep 5
done
mem() {
  echo "== $1 $(date -u +%FT%TZ)"
  for n in head worker; do
    cmd='grep -E "^(MemFree|MemAvailable|Cached|Buffers|SwapFree):" /proc/meminfo | tr -s " " | tr "\n" " "; echo; grep Normal /proc/buddyinfo; grep -E "^(compact_stall|compact_fail|allocstall_normal) " /proc/vmstat | tr "\n" " "; echo'
    if [[ $n == head ]]; then echo "-- head"; bash -c "$cmd"; else echo "-- worker"; ssh -o BatchMode=yes spark2 "$cmd"; fi
  done
}
mem before-stop > "$OUT/mem.txt"
"$R/stop.sh" > "$OUT/stop.txt" 2>&1
sleep "${SETTLE:-5}"
mem after-stop >> "$OUT/mem.txt"
for kv in "$@"; do
  if [[ $kv == PRECACHE=* ]]; then  # stale page cache on the head: that many GiB of another model's files, read once
    find -L ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots -name '*.safetensors' -print0 |
      xargs -0 cat | head -c "$(( ${kv#PRECACHE=} << 30 ))" > /dev/null || true
    mem after-precache >> "$OUT/mem.txt"
  fi
done
if [[ " $* " == *" PAGECACHE=pack "* ]]; then  # the state after a download: both nodes' page cache full of the pack
  snap=$(ls -d ~/.cache/huggingface/hub/models--sfxnz--GLM-5.3-Flash-EXL3-4bpw/snapshots/*)
  ssh -o BatchMode=yes spark2 "cat $snap/*.safetensors > /dev/null" &
  cat "$snap"/*.safetensors > /dev/null; wait
  mem after-pagecache >> "$OUT/mem.txt"
fi
if [[ " $* " == *" DEFRAG=1 "* ]]; then
  { echo "-- head"; python3 "$W/defrag.py"; echo "-- worker"; ssh -o BatchMode=yes spark2 python3 - < "$W/defrag.py"; } > "$OUT/defrag.txt" 2>&1
  mem after-defrag >> "$OUT/mem.txt"
fi
t0=$SECONDS
env "$@" "$R/run.sh" > "$OUT/run.txt" 2>&1 || { tail -30 "$OUT/run.txt"; echo "BOOT FAILED $NAME"; exit 1; }
echo "boot_s $((SECONDS - t0))" >> "$OUT/run.txt"
mem ready >> "$OUT/mem.txt"
quiet='"(GET|POST) /'
docker logs tf-glm53-rank0 2>&1 | grep -Ev "$quiet" > "$OUT/boot.log" || true
ssh -o BatchMode=yes spark2 docker logs tf-glm53-rank1 2>&1 | grep -Ev "$quiet" > "$OUT/boot-rank1.log" || true
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}{{join .Config.Cmd " "}}' tf-glm53-rank0 |
  grep -E '^(TF_|TENSORFOLD_|NCCL_)|serve' > "$OUT/config.txt"
docker inspect --format '{{.Config.Image}} {{.Image}}' tf-glm53-rank0 >> "$OUT/config.txt"
grep -E 'ready after|boot_s' "$OUT/run.txt"; grep -E 'CUDA sm_|serving|windows|warning' "$OUT/boot.log" | cut -c1-600 || true
