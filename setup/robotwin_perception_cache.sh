#!/usr/bin/env bash
# Images for the RoboTwin front end: every task in demo_clean and demo_randomized, PER processes per GPU. A finished task is skipped.
# Usage: GPUS="4 5 6 7" PER=5 CLEAN=20 RANDOMIZED=40 [SEED0=1000 BUDGET=7200 OUT=dir] setup/robotwin_perception_cache.sh
set -uo pipefail
export RT_ROOT=${RT_ROOT:-/mnt/data/user01/RoboTwin}
export RT_PY=${RT_PY:-/mnt/data/user01/envs/robotwin_sw/bin/python}
export SW_ROOT="${SW_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export OUT=${OUT:-$SW_ROOT/data/perception_rt}
export GPUS=${GPUS:-4 5 6 7} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SEED0=${SEED0:-0} BUDGET=${BUDGET:-12600}
PER=${PER:-5}
mkdir -p "$OUT/logs"
cd "$RT_ROOT"
tasks=$(ls /mnt/data/user01/spikewam_robotwin/demo_clean/*.npz | xargs -n1 basename | sed 's/\.npz$//' | grep -v '_part')
read -ra g <<< "$GPUS"
{
  [ "${CLEAN:-20}" -gt 0 ] && for t in $tasks; do echo "$t demo_clean ${CLEAN:-20}"; done
  [ "${RANDOMIZED:-40}" -gt 0 ] && for t in $tasks; do echo "$t demo_randomized ${RANDOMIZED:-40}"; done
} | xargs -P $((PER * ${#g[@]})) --process-slot-var=SLOT -L1 bash -c '
  read -ra g <<< "$GPUS"
  gpu=${g[$((SLOT % ${#g[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu timeout 4h "$RT_PY" -W ignore "$SW_ROOT/spikewam/robotwin/perception_cache.py" --task "$0" --config "$1" --episodes "$2" --out "$OUT" --seed0 "${SEED0:-0}" --budget "${BUDGET:-12600}" > "$OUT/logs/$1_$0.log" 2>&1
  echo "$(date +%X) gpu $gpu $1 $0 exit $?"
'
echo "all done: $(ls "$OUT"/*/meta.npz 2>/dev/null | wc -l) sets"
