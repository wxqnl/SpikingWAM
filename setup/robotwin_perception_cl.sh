#!/usr/bin/env bash
# Front-end images from the policy's own closed loop (spikewam/robotwin/perception_cl.py): every task in demo_clean and
# demo_randomized, PER processes per GPU, one process group. Training seeds and seen instructions; a finished task is skipped.
# Usage: GPUS="5" PER=10 PERCEPT=runs/perception/rt160h/ckpt.pt CLEAN=10 RANDOMIZED=10 [SEED0=5000 BUDGET=7200 OUT=dir PERCEPT_RIGID=1] setup/robotwin_perception_cl.sh
set -uo pipefail
export RT_ROOT=${RT_ROOT:-/mnt/data/user01/RoboTwin}
export RT_PY=${RT_PY:-/mnt/data/user01/envs/robotwin_sw/bin/python}
export SW_ROOT="${SW_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export SW_MINILM="${SW_MINILM:-$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} CUDA_DEVICE_ORDER=PCI_BUS_ID
export OUT=${OUT:-$SW_ROOT/data/perception_rtcl}
export GPUS=${GPUS:-5} SEED0=${SEED0:-5000} BUDGET=${BUDGET:-7200}
export PERCEPT=$SW_ROOT/$PERCEPT CKPT=$SW_ROOT/${CKPT:-runs/rt_f2ppt16_big/ckpt_150000.pt}
PER=${PER:-10}
mkdir -p "$OUT/logs"
cd "$RT_ROOT"
tasks=$(ls /mnt/data/user01/spikewam_robotwin/demo_clean/*.npz | xargs -n1 basename | sed 's/\.npz$//' | grep -v '_part')
read -ra g <<< "$GPUS"
{
  for t in $tasks; do echo "$t demo_clean ${CLEAN:-10}"; done
  for t in $tasks; do echo "$t demo_randomized ${RANDOMIZED:-10}"; done
} | xargs -P $((PER * ${#g[@]})) --process-slot-var=SLOT -L1 bash -c '
  read -ra g <<< "$GPUS"
  gpu=${g[$((SLOT % ${#g[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu timeout 3h "$RT_PY" -W ignore "$SW_ROOT/spikewam/robotwin/perception_cl.py" --task "$0" --config "$1" --episodes "$2" --ckpt "$CKPT" --perception "$PERCEPT" --out "$OUT" --seed0 "$SEED0" --budget "$BUDGET" > "$OUT/logs/$1_$0.log" 2>&1
  echo "$(date +%X) gpu $gpu $1 $0 exit $?"
'
echo "all done: $(ls "$OUT"/*/meta.npz 2>/dev/null | wc -l) sets"
