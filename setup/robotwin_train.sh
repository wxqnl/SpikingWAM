#!/usr/bin/env bash
# Train one multi-task RoboTwin 2.0 run on the expert demonstrations of all tasks (setup/robotwin_collect.sh).
# Usage: GPU=2 [STEPS=120000] setup/robotwin_train.sh <run> <model: ann|spike> [--set key=value ...]
# STEPS: training steps (default 60000; 120000 = about 10 passes over the 556k frames, as 60000 on LIBERO's 279k).
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$SW_ROOT"
run=$1 model=$2
shift 2
export CUDA_VISIBLE_DEVICES=${GPU:-7}
mkdir -p logs
steps=${STEPS:-60000}
[ -f "$SW_RUNS/$run/ckpt_$steps.pt" ] || "$SW_PY" -m spikewam.train --run "$run" --model "$model" --bench robotwin \
  --steps "$steps" --data "${RT_DATA:-/mnt/data/user01/spikewam_robotwin/demo_clean}" "$@" > "logs/$run.log" 2>&1
