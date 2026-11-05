#!/usr/bin/env bash
# Prototype two-node TensorFold launcher for the engine probe. usage: tf_pair.sh up|down|logs SNAPSHOT_DIR [serve args...]
set -euo pipefail
act=$1; snap=${2:-}; shift 2 || true
image=${IMAGE:-tf-glm53:1.2.2}
master=10.100.8.1
env_common=(-e TF_GLM_WORLD=2 -e NCCL_COMM_ID=$master:29583 -e NCCL_SOCKET_IFNAME=enp1s0f1np1
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e TF_ROCE_HCA=rocep1s0f1)
run_rank() {
  local host=$1 rank=$2; shift 2
  local cmd=(docker run -d --name tf-glm53-probe-r$rank --label ai-lab.recipe=glm53-tensorfold-probe
    --gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --ulimit core=0
    --oom-score-adj 1000 -v "${snap%/snapshots/*}:${snap%/snapshots/*}:ro" -e TF_GLM_RANK=$rank "${env_common[@]}" $EXTRA_DOCKER
    "$image" tensorfold-native serve "$snap" --host 0.0.0.0 --port 8000 --name glm-5.3-flash --no-update-check "$@")
  if [[ $host == local ]]; then "${cmd[@]}"; else ssh spark2 "$(printf '%q ' "${cmd[@]}")"; fi
}
EXTRA_DOCKER=${EXTRA_DOCKER:-}
case $act in
  up) run_rank spark2 1 "$@"; sleep 2; run_rank local 0 "$@" ;;
  down) docker stop -t 60 tf-glm53-probe-r0 2>/dev/null || true; ssh spark2 'docker stop -t 60 tf-glm53-probe-r1' 2>/dev/null || true
        docker rm tf-glm53-probe-r0 2>/dev/null || true; ssh spark2 'docker rm tf-glm53-probe-r1' 2>/dev/null || true ;;
  logs) docker logs tf-glm53-probe-r0 2>&1 | tail -40; echo ---- rank1; ssh spark2 'docker logs tf-glm53-probe-r1 2>&1 | tail -20' ;;
esac
