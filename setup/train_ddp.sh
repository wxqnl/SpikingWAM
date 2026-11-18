#!/usr/bin/env bash
# Data-parallel training (spikewam/train.py) without torchrun, whose argument parser takes our --run for its
# --run-path: one process per GPU in GPUS, each seeing only its GPU; --batch is the global batch.
# Usage: GPUS="0 5 6 7" MASTER_PORT=29531 setup/train_ddp.sh --run <run> --batch 512 ...
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
read -ra gpus <<< "${GPUS:?set GPUS}"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29531} WORLD_SIZE=${#gpus[@]} LOCAL_RANK=0
cd "$SW_ROOT"
pids=()
for r in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpus[$r]} RANK=$r "$SW_PY" -m spikewam.train "$@" &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
exit $status
