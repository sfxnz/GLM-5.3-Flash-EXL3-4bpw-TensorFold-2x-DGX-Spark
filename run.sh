#!/usr/bin/env bash
# GLM-5.3-Flash EXL3 4bpw on two DGX Sparks: TensorFold 1.2.2 at TP=2 over the direct RoCE cable, with vision, MTP drafts
# and thinking. Run it on the head (rank 0, HTTP); it drives the worker (rank 1) over ssh. Settings live in recipe.conf.
# Each step reads the state that exists and acts only on the difference, so a rerun converges.
# VALIDATE_ONLY=1 runs every check read-only, prints what it would do (both docker argvs included) and exits 1 if
# run.sh would refuse.
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=recipe.conf
source "$DIR/recipe.conf"
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
REFUSED=0
WORKER_OK=0
PAIR_STARTING=0
GIB=1073741824

die() { echo "error: $*" >&2; exit 1; }
ok() { echo "ok     $*"; }
quote() { local q; printf -v q '%q ' "$@"; echo "${q% }"; }
refuse() {
  [[ $VALIDATE_ONLY == 1 ]] || die "$*"
  echo "REFUSE $*"
  REFUSED=$((REFUSED + 1))
}
act() {
  local what=$1; shift
  if [[ $VALIDATE_ONLY == 1 ]]; then echo "WOULD  $what: $(quote "$@")"; else echo "...    $what"; "$@"; fi
}

# on head|worker FN [ARGS...]: run one of the self-contained probe_* functions below on that node.
on() {
  local node=$1; shift
  if [[ $node == head ]]; then "$@"; return; fi
  local def=""
  if declare -F "$1" >/dev/null; then def="$(declare -f "$1"); "; fi
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "bash -c $(printf %q "$def$(quote "$@")")"
}

probe_tools() { local t; for t in "$@"; do command -v "$t" >/dev/null || echo "$t"; done; }

# probe_link IFACE HCA,HCA: "state UP|DOWN", "ip ADDR", then "hca NAME ACTIVE|inactive" for each HCA.
probe_link() {
  local h links
  if ip -o link show dev "$1" 2>/dev/null | grep -q ' state UP '; then echo "state UP"; else echo "state DOWN"; fi
  echo "ip $(ip -4 -o addr show dev "$1" 2>/dev/null | awk '{sub(/\/.*/, "", $4); print $4; exit}')"
  links=$(rdma link show 2>/dev/null || true)
  for h in ${2//,/ }; do
    if grep -q "^link $h/1 state ACTIVE" <<<"$links"; then echo "hca $h ACTIVE"; else echo "hca $h inactive"; fi
  done
}

# probe_gpu_containers LABEL: "NAME IMAGE" for each running container that holds a GPU and does not carry LABEL.
probe_gpu_containers() {
  local ids id name image label gpu
  ids=$(docker ps -q) || return 1
  for id in $ids; do
    IFS='|' read -r name image label gpu < <(docker inspect --format \
      '{{.Name}}|{{.Config.Image}}|{{index .Config.Labels "ai-lab.recipe"}}|{{.HostConfig.Runtime}} {{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}}' "$id")
    [[ "ai-lab.recipe=$label" == "$1" ]] && continue
    if [[ $gpu == nvidia* || $gpu == *'"gpu"'* || $gpu == */dev/nvidia* ]]; then echo "${name#/} $image"; fi
  done
}

probe_ours() { docker ps -a --filter "label=$1" --format '{{.Names}} {{.State}}'; }
probe_image() { docker image inspect --format '{{.Id}} {{index .Config.Labels "tensorfold.archive.sha256"}}' "$1" 2>/dev/null || true; }
probe_mem() { free -b | awk '/^Mem:/ {print $7}'; }
probe_ports() { ss -Htln | awk '{n = split($4, a, ":"); print a[n]}' | grep -Fx -f <(printf '%s\n' "$@") | sort -u || true; }

# snapshot_check DIR SHARDS BYTES: prints "complete", or the first thing missing and exits 1.
snapshot_check() {
  local d=$1 f shards n total
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json processor_config.json \
    chat_template.jinja model.safetensors.index.json; do
    [[ -s $d/$f ]] || { echo "no $f in $d"; return 1; }
  done
  shards=$(grep -o '"[^"]*\.safetensors"' "$d/model.safetensors.index.json" | tr -d '"' | sort -u)
  n=$(grep -c . <<<"$shards" || true)
  [[ $n == "$2" ]] || { echo "the index names $n shards, want $2"; return 1; }
  for f in $shards; do [[ -f $d/$f ]] || { echo "no $f"; return 1; }; done
  # shellcheck disable=SC2086
  total=$(cd "$d" && stat -Lc %s $shards | awk '{s += $1} END {printf "%d", s}')
  [[ $total == "$3" ]] || { echo "the shards hold $total bytes, want $3"; return 1; }
  echo complete
}

# rank_argv RANK: the whole `docker run` argv for that rank, one word per line. Both ranks come from here, so their
# serve args are identical (the ranks refuse to serve otherwise) and only TF_GLM_RANK and the name differ.
rank_argv() {
  printf '%s\n' docker run -d --name "$NAME_PREFIX$1" --label "$LABEL" \
    --gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --ulimit core=0 --oom-score-adj 1000 \
    -v "$REPO_DIR:$REPO_DIR:ro" \
    -e "TF_GLM_RANK=$1" -e TF_GLM_WORLD=2 -e "NCCL_COMM_ID=$HEAD_IP:$NCCL_PORT" -e "NCCL_SOCKET_IFNAME=$IFACE" \
    -e "NCCL_IB_HCA=$HCA" -e "TF_ROCE_HCA=$ROCE_HCA" \
    -e "TF_GLM_DENSE_BITS=$DENSE_BITS" -e "TF_GLM_HEAD_BITS=$HEAD_BITS" -e "TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB" \
    "$IMAGE" tensorfold-native serve "$SNAPSHOT" --host 0.0.0.0 --port "$PORT" --name "$SERVED_NAME" \
    --context "$CONTEXT" --parallel "$PARALLEL" --vision --no-update-check
}

logs_tail() {
  echo "--- rank 0 ($(hostname -s)), last 40 lines"
  docker logs --tail 40 "${NAME_PREFIX}0" 2>&1 || true
  echo "--- rank 1 ($WORKER), last 40 lines"
  ssh -o BatchMode=yes "$WORKER" docker logs --tail 40 "${NAME_PREFIX}1" 2>&1 || true
}

endpoint() {
  echo "endpoint  http://$(hostname -s):$PORT/v1"
  echo "try       curl -s http://$(hostname -s):$PORT/v1/chat/completions -H 'content-type: application/json' \\"
  echo "            -d '{\"model\": \"$SERVED_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
  echo "stop      $DIR/stop.sh"
}

running() { # running head|worker RANK
  [[ $(on "$1" docker inspect -f '{{.State.Running}}' "$NAME_PREFIX$2" 2>/dev/null) == true ]]
}

check_config() {
  local v
  [[ $VALIDATE_ONLY =~ ^[01]$ ]] || die "VALIDATE_ONLY=$VALIDATE_ONLY must be 0 or 1"
  [[ $FORCE =~ ^[01]$ ]] || die "FORCE=$FORCE must be 0 or 1"
  for v in PORT CONTEXT PARALLEL NCCL_PORT MEM_GATE_GIB READY_TIMEOUT; do
    [[ ${!v} =~ ^[1-9][0-9]*$ ]] || die "$v=${!v} is not a positive integer"
  done
  for v in DENSE_BITS HEAD_BITS; do
    [[ ${!v} =~ ^(4|6|8|16)$ ]] || die "$v=${!v} must be 4, 6, 8 or 16"
  done
  [[ $MEMORY_RESERVE_GIB =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB is not a number of GiB"
  [[ $HF_CACHE == /* && $HF_CACHE != *[[:space:]]* ]] || die "HF_CACHE=$HF_CACHE must be an absolute path without spaces"
  REPO_DIR=$HF_CACHE/hub/models--${MODEL//\//--}
  SNAPSHOT=$REPO_DIR/snapshots/$REVISION
}

check_pin() {
  if [[ $REVISION =~ ^[0-9a-f]{40}$ && $SHARDS =~ ^[1-9][0-9]*$ && $SHARD_BYTES =~ ^[1-9][0-9]*$ ]]; then
    PINNED=1
  else
    PINNED=0
    refuse "the pack is not pinned: REVISION=$REVISION SHARDS=$SHARDS SHARD_BYTES=$SHARD_BYTES (fill them in recipe.conf once $MODEL is published)"
  fi
}

print_config() {
  local v
  echo "config (recipe.conf, environment wins):"
  for v in $(sed -n 's/^\([A-Z][A-Z0-9_]*\)=.*/\1/p' "$DIR/recipe.conf"); do echo "  $v=${!v}"; done
  echo "  snapshot=$SNAPSHOT"
}

step_reach() {
  local missing
  missing=$(probe_tools docker ssh curl rsync hf ip rdma ss free stat)
  [[ -z $missing ]] || die "the head lacks: $(echo "$missing" | xargs)"
  ok "head $(hostname -s) has docker ssh curl rsync hf ip rdma ss free stat"
  if ! on worker true 2>/dev/null; then refuse "cannot ssh $WORKER (BatchMode)"; return; fi
  WORKER_OK=1
  missing=$(on worker probe_tools docker rsync ip rdma ss free stat)
  if [[ -n $missing ]]; then refuse "$WORKER lacks: $(echo "$missing" | xargs)"; else ok "ssh $WORKER; it has docker rsync ip rdma ss free stat"; fi
}

step_already_up() {
  local h w=""
  h=$(on head probe_ours "$LABEL")
  [[ $WORKER_OK == 1 ]] && w=$(on worker probe_ours "$LABEL")
  if ! grep -Fxq "${NAME_PREFIX}0 running" <<<"$h" || ! grep -Fxq "${NAME_PREFIX}1 running" <<<"$w"; then return 0; fi
  if curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "already serving: ${NAME_PREFIX}0 on $(hostname -s) and ${NAME_PREFIX}1 on $WORKER answer /health"
    endpoint
    exit 0
  fi
  PAIR_STARTING=1
  echo "both ranks run but /health does not answer yet: waiting for them instead of restarting"
}

step_preflight() {
  local node host lines state ip want hca bad gpu
  for node in head worker; do
    [[ $node == worker && $WORKER_OK == 0 ]] && continue
    host=$(hostname -s); want=$HEAD_IP
    [[ $node == worker ]] && { host=$WORKER; want=$WORKER_IP; }
    lines=$(on "$node" probe_link "$IFACE" "$HCA")
    state=$(sed -n 's/^state //p' <<<"$lines"); ip=$(sed -n 's/^ip //p' <<<"$lines")
    if [[ $state != UP || -z $ip ]]; then
      refuse "$IFACE on $host is not up with an IPv4 address (state $state, ip '${ip}')"
    elif [[ -n $want && $want != "$ip" ]]; then
      refuse "$host: ${node^^}_IP=$want, but $IFACE has $ip (NCCL binds to $IFACE)"
    else
      ok "$host $IFACE up at $ip"
    fi
    if [[ $node == head ]]; then HEAD_IP=${HEAD_IP:-$ip}; else WORKER_IP=${WORKER_IP:-$ip}; fi
    bad=$(grep '^hca .* inactive$' <<<"$lines" | cut -d' ' -f2 | xargs || true)
    hca=$(grep '^hca .* ACTIVE$' <<<"$lines" | cut -d' ' -f2 | xargs || true)
    if [[ -n $bad ]]; then refuse "$host: RoCE HCA not ACTIVE: $bad"; else ok "$host RoCE HCAs ACTIVE: $hca"; fi
    if ! gpu=$(on "$node" probe_gpu_containers "$LABEL"); then
      refuse "$host: cannot list containers"
    elif [[ -n $gpu ]]; then
      while read -r g; do refuse "$host runs GPU container $g; this recipe needs both GPUs to itself and never stops it"; done <<<"$gpu"
    else
      ok "$host has no other GPU container"
    fi
  done
  HEAD_IP=${HEAD_IP:-unknown}
  WORKER_IP=${WORKER_IP:-unknown}
}

step_image() {
  local h w
  h=$(on head probe_image "$IMAGE")
  if [[ -n $h && ${h#* } == "$TF_SHA256" ]]; then
    ok "image $IMAGE on head: ${h%% *}, TensorFold archive sha256 ${TF_SHA256:0:12}"
  else
    [[ -n $h ]] && echo "note   $IMAGE on head carries archive sha256 ${h#* }, want $TF_SHA256: rebuilding"
    act "build $IMAGE from docker/" docker build -t "$IMAGE" --build-arg "TF_VERSION=$TF_VERSION" \
      --build-arg "TF_SHA256=$TF_SHA256" "$DIR/docker"
    if [[ $VALIDATE_ONLY == 0 ]]; then
      h=$(on head probe_image "$IMAGE")
      [[ ${h#* } == "$TF_SHA256" ]] || die "the built $IMAGE carries archive sha256 '${h#* }', want $TF_SHA256"
    fi
  fi
  [[ $WORKER_OK == 1 ]] || return 0
  w=$(on worker probe_image "$IMAGE")
  if [[ -n $h && ${w%% *} == "${h%% *}" ]]; then
    ok "image $IMAGE on $WORKER: same id"
  else
    echo "note   $IMAGE on $WORKER: '${w%% *}', head has '${h%% *}'"
    act "copy $IMAGE to $WORKER over $IFACE" bash -c 'docker save "$1" | ssh -o BatchMode=yes "$2" docker load' _ "$IMAGE" "$WORKER_IP"
    if [[ $VALIDATE_ONLY == 0 ]]; then
      w=$(on worker probe_image "$IMAGE")
      [[ ${w%% *} == "${h%% *}" ]] || die "$IMAGE on $WORKER is '${w%% *}' after the copy, want ${h%% *}"
    fi
  fi
}

step_weights() {
  local st inc=() p pats
  if [[ $PINNED == 0 ]]; then echo "skip   weights: the pack is not pinned"; return; fi
  if st=$(on head snapshot_check "$SNAPSHOT" "$SHARDS" "$SHARD_BYTES"); then
    ok "head snapshot complete: $SHARDS shards, $SHARD_BYTES bytes in $SNAPSHOT"
  else
    echo "note   head snapshot: $st"
    read -ra pats <<<"$HF_INCLUDE"
    for p in "${pats[@]}"; do inc+=(--include "$p"); done
    act "download $MODEL at $REVISION" hf download "$MODEL" --revision "$REVISION" --cache-dir "$HF_CACHE/hub" "${inc[@]}"
    if [[ $VALIDATE_ONLY == 0 ]]; then
      st=$(snapshot_check "$SNAPSHOT" "$SHARDS" "$SHARD_BYTES") || die "after the download, head snapshot: $st"
    fi
  fi
  [[ $WORKER_OK == 1 ]] || return 0
  if st=$(on worker snapshot_check "$SNAPSHOT" "$SHARDS" "$SHARD_BYTES"); then
    ok "$WORKER snapshot complete"
  else
    echo "note   $WORKER snapshot: $st"
    act "make the snapshot dir on $WORKER" ssh -o BatchMode=yes "$WORKER" mkdir -p "$SNAPSHOT"
    act "copy the snapshot to $WORKER over $IFACE" rsync -aL --partial --info=progress2 "$SNAPSHOT/" "$WORKER_IP:$SNAPSHOT/"
    if [[ $VALIDATE_ONLY == 0 ]]; then
      st=$(on worker snapshot_check "$SNAPSHOT" "$SHARDS" "$SHARD_BYTES") || die "after the copy, $WORKER snapshot: $st"
    fi
  fi
}

# Anything of ours that is not a running pair (an exited rank, a half-started pair) goes before a fresh start.
step_clear_ours() {
  local h w=""
  h=$(on head probe_ours "$LABEL")
  [[ $WORKER_OK == 1 ]] && w=$(on worker probe_ours "$LABEL")
  if [[ -n $h$w ]]; then
    echo "note   ours left from an earlier run: $(echo "$h" "$w" | xargs)"
    act "stop and remove them" "$DIR/stop.sh"
    CLEARED_OURS=1
  else
    CLEARED_OURS=0
  fi
}

step_ports() {
  local node host busy
  if [[ $VALIDATE_ONLY == 1 && $CLEARED_OURS == 1 ]]; then
    echo "WOULD  check ports $PORT $NCCL_PORT $((NCCL_PORT + 1)) once ours are gone"
    return
  fi
  for node in head worker; do
    [[ $node == worker && $WORKER_OK == 0 ]] && continue
    host=$(hostname -s); [[ $node == worker ]] && host=$WORKER
    busy=$(on "$node" probe_ports "$PORT" "$NCCL_PORT" "$((NCCL_PORT + 1))" | xargs)
    if [[ -n $busy ]]; then
      refuse "$host: port(s) $busy already listen; a stale rank there makes the start fail BindFailed (find it with ss -ltnp)"
    else
      ok "$host ports $PORT $NCCL_PORT $((NCCL_PORT + 1)) free"
    fi
  done
}

step_memory() {
  local node host avail low=""
  for node in head worker; do
    [[ $node == worker && $WORKER_OK == 0 ]] && continue
    host=$(hostname -s); [[ $node == worker ]] && host=$WORKER
    avail=$(on "$node" probe_mem)
    if (( avail < MEM_GATE_GIB * GIB )); then
      low+="$host $(awk -v b="$avail" 'BEGIN {printf "%.1f", b / 2^30}') GiB, "
    else
      ok "$host MemAvailable $(awk -v b="$avail" 'BEGIN {printf "%.1f", b / 2^30}') GiB >= $MEM_GATE_GIB GiB"
    fi
  done
  [[ -n $low ]] || return 0
  echo "--- free -h on $(hostname -s)"; free -h
  if [[ $WORKER_OK == 1 ]]; then echo "--- free -h on $WORKER"; ssh -o BatchMode=yes "$WORKER" free -h; fi
  if [[ $FORCE == 1 ]]; then
    echo "FORCE=1: starting below the ${MEM_GATE_GIB} GiB gate (${low%, })"
  else
    refuse "MemAvailable below the ${MEM_GATE_GIB} GiB gate: ${low%, }. Free memory first (no sudo, no cache drops), or set FORCE=1"
  fi
}

step_start() {
  local a0 a1 i
  mapfile -t a1 < <(rank_argv 1)
  mapfile -t a0 < <(rank_argv 0)
  if [[ $VALIDATE_ONLY == 1 ]]; then
    echo "argv rank1 $WORKER: $(quote "${a1[@]}")"
    echo "argv rank0 $(hostname -s): $(quote "${a0[@]}")"
    return
  fi
  echo "...    start rank 1 on $WORKER"
  on worker "${a1[@]}" >/dev/null
  for i in $(seq 30); do running worker 1 && break; (( i < 30 )) || { logs_tail; "$DIR/stop.sh"; die "rank 1 did not start"; }; sleep 1; done
  echo "...    start rank 0 on $(hostname -s)"
  "${a0[@]}" >/dev/null || { logs_tail; "$DIR/stop.sh"; die "rank 0 did not start"; }
}

step_wait() {
  local t0=$SECONDS next=30
  if [[ $VALIDATE_ONLY == 1 ]]; then
    echo "WOULD  wait up to ${READY_TIMEOUT}s for http://127.0.0.1:$PORT/health, then print rank 0's startup lines"
    return
  fi
  echo "...    wait for /health (about 2 min once the weights sit in page cache; up to ${READY_TIMEOUT}s)"
  until curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    local why=""
    running head 0 || why="rank 0 exited"
    running worker 1 || why="rank 1 exited"
    (( SECONDS - t0 < READY_TIMEOUT )) || why="no /health after ${READY_TIMEOUT}s"
    if [[ -n $why ]]; then
      echo "error: $why" >&2
      logs_tail
      "$DIR/stop.sh"
      exit 1
    fi
    if (( SECONDS - t0 >= next )); then echo "       $((SECONDS - t0))s"; next=$((next + 30)); fi
    sleep 5
  done
  echo "ready after $((SECONDS - t0))s"
  docker logs "${NAME_PREFIX}0" 2>&1 | grep -E '^\[tensorfold\] (warning|CUDA|serving)' || true
  endpoint
}

check_config
echo "== $MODEL, TensorFold $TF_VERSION, TP=2: $(hostname -s) rank 0 (HTTP) + $WORKER rank 1"
[[ $VALIDATE_ONLY == 0 ]] || print_config
check_pin
step_reach
step_already_up
if [[ $PAIR_STARTING == 0 ]]; then
  step_preflight
  step_image
  step_weights
  step_clear_ours
  step_ports
  step_memory
  step_start
fi
step_wait
if [[ $VALIDATE_ONLY == 1 ]]; then
  if (( REFUSED > 0 )); then echo "VALIDATE_ONLY: run.sh would refuse ($REFUSED reasons above)"; exit 1; fi
  echo "VALIDATE_ONLY: every check passes; run.sh would start the pair"
fi
