#!/usr/bin/env bash
# Collect the RoboTwin 2.0 expert demonstrations with skeletons for every task (one process per task, round robin
# over the GPUs given). Skips tasks whose npz exists.
# Usage: GPUS="2 3 4 5 6 7" EPISODES=50 CONFIG=demo_clean setup/robotwin_collect.sh [task ...]
set -uo pipefail
RT_ROOT=${RT_ROOT:-/mnt/data/user01/RoboTwin}
RT_PY=${RT_PY:-/mnt/data/user01/envs/robotwin_sw/bin/python}
SW_ROOT="${SW_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT=${OUT:-/mnt/data/user01/spikewam_robotwin/${CONFIG:-demo_clean}}
mkdir -p "$OUT/logs"
cd "$RT_ROOT"
tasks=${*:-$(ls envs/*.py | xargs -n1 basename | grep -v '^_' | grep -v __init__ | sed 's/\.py$//')}
read -ra gpus <<< "${GPUS:-2 3 4 5 6 7}"
i=0
for t in $tasks; do
  [ -f "$OUT/$t.npz" ] && continue
  g=${gpus[$((i % ${#gpus[@]}))]}
  i=$((i + 1))
  CUDA_VISIBLE_DEVICES=$g setsid nohup "$RT_PY" -W ignore "$SW_ROOT/spikewam/robotwin/collect.py" --task "$t" \
    --config "${CONFIG:-demo_clean}" --episodes "${EPISODES:-50}" --out "$OUT" > "$OUT/logs/$t.log" 2>&1 < /dev/null &
done
echo "$i collectors started"
