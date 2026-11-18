#!/usr/bin/env bash
# Train one run, then evaluate it on LIBERO (recording failed episodes) and probe the hand-object offsets.
# Usage: GPU=7 setup/run_and_eval.sh <run> <model: ann|spike> [--set key=value ...]
# Skips any stage whose output already exists, so a rerun resumes.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$SW_ROOT"
run=$1 model=$2
shift 2
export CUDA_VISIBLE_DEVICES=${GPU:-7}
ckpt=$SW_RUNS/$run/ckpt_60000.pt
mkdir -p logs
[ -f "$ckpt" ] || "$SW_PY" -m spikewam.train --run "$run" --model "$model" "$@" > "logs/$run.log" 2>&1
[ -f "$SW_RUNS/$run/eval_ckpt_60000_ema.json" ] || "$SW_PY" -m spikewam.eval_libero --ckpt "$ckpt" \
  --workers "${WORKERS:-32}" --record > "logs/eval_$run.log" 2>&1
[ -f "$SW_RUNS/$run/probe_ckpt_60000_ema.json" ] || "$SW_PY" scripts/probe_offsets.py --ckpt "$ckpt" \
  > "logs/probe_$run.log" 2>&1
grep -A 8 '"suites"' "logs/eval_$run.log" | head -9
