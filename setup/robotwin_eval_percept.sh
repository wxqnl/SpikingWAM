#!/usr/bin/env bash
# RoboTwin closed loop with the spiking front end: one evaluator per task, PER at a time on each GPU of GPUS, in one process
# group, so that stopping this script stops them. Seeds and instructions are those of the state-input run.
# Usage: GPUS="0 1 2" PER=3 CONFIG=demo_clean TEST_NUM=10 PERCEPT=runs/perception/rt160a/ckpt.pt TAG=rt160a [CKPT=runs/<policy>/ckpt_N.pt] setup/robotwin_eval_percept.sh
# Writes runs/robotwin_eval/rt_f2ppt16_big_percept_<TAG>/<config>/<task>.json; a second call resumes (and can raise TEST_NUM).
set -uo pipefail
export RT_ROOT=${RT_ROOT:-/mnt/data/user01/RoboTwin}
export RT_PY=${RT_PY:-/mnt/data/user01/envs/robotwin_sw/bin/python}
export SW_ROOT="${SW_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export SW_MINILM="${SW_MINILM:-$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} CUDA_DEVICE_ORDER=PCI_BUS_ID
export GPUS=${GPUS:-0 1 2} CONFIG=${CONFIG:-demo_clean} TEST_NUM=${TEST_NUM:-100}
export PERCEPT=$SW_ROOT/$PERCEPT CKPT=$SW_ROOT/${CKPT:-runs/rt_f2ppt16_big/ckpt_150000.pt}
export OUT=$SW_ROOT/runs/robotwin_eval/rt_f2ppt16_big_percept_$TAG
PER=${PER:-3}
mkdir -p "$OUT/logs"
cd "$RT_ROOT"
read -ra g <<< "$GPUS"
tasks=${TASKS:-$(ls envs/*.py | xargs -n1 basename | grep -v '^_' | grep -v __init__ | sed 's/\.py$//')}
printf '%s\n' $tasks | xargs -P $((PER * ${#g[@]})) --process-slot-var=SLOT -L1 bash -c '
  read -ra g <<< "$GPUS"
  gpu=${g[$((SLOT % ${#g[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu "$RT_PY" -W ignore "$SW_ROOT/spikewam/robotwin/evaluate.py" --task "$0" --config "$CONFIG" --ckpt "$CKPT" --out "$OUT" --test_num "$TEST_NUM" \
    --instruction_type unseen --k 3 --init_noise -1 --perception "$PERCEPT" >> "$OUT/logs/${CONFIG}_$0.log" 2>&1
  echo "$(date +%X) gpu $gpu $0 exit $?"
'
echo "all done"
