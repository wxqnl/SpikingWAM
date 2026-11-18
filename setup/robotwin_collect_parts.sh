#!/usr/bin/env bash
# Large RoboTwin collections: every task is split into PARTS parts of PER episodes (collect.py --seed0 --tag), run
# MAX at a time over the GPUs given; parts land in $OUT/parts/<task>.p<seed0>.npz (skipped if present) and are merged
# into $OUT/<task>.npz with scripts/robotwin_merge.py once all parts of a task are done. Parts start one at a time
# (a lock held until the new collector has taken its GPU memory) on the GPU with the most free memory, and only when
# that GPU keeps MIN_FREE MiB free after subtracting its RESERVE (e.g. RESERVE="5:26000" keeps 26 GB of GPU 5 for
# evaluations).
# Usage: GPUS="0 6 7" MAX=40 CONFIG=demo_randomized PARTS=5 PER=100 SEED0=10000 STRIDE=15000 MIN_FREE=8000 \
#          RESERVE="5:26000" setup/robotwin_collect_parts.sh [task ...]
# Seeds stay below 100000 (the evaluation seeds). A part starts only when no lock $OUT/locks/<task>.p<seed0> exists (it
# makes one), so several instances can run side by side, and only while fewer than LIMIT collectors run in total.
set -uo pipefail
RT_ROOT=${RT_ROOT:-/mnt/data/user01/RoboTwin}
RT_PY=${RT_PY:-/mnt/data/user01/envs/robotwin_sw/bin/python}
SW_ROOT="${SW_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
SW_PY=${SW_PY:-/home/user01/Minko/spike-new-vla-icml/env/bin/python}
CONFIG=${CONFIG:-demo_randomized}
OUT=${OUT:-/mnt/data/user01/spikewam_robotwin/$CONFIG}
PARTS=${PARTS:-5}
PER=${PER:-100}
SEED0=${SEED0:-10000}
STRIDE=${STRIDE:-15000}
MAX=${MAX:-40}
LIMIT=${LIMIT:-$MAX}
MIN_FREE=${MIN_FREE:-0}
RESERVE=${RESERVE:-}
read -ra gpus <<< "${GPUS:-0 6 7}"
(( SEED0 + PARTS * STRIDE <= 100000 )) || { echo "seeds would reach the evaluation seeds"; exit 1; }
mkdir -p "$OUT/parts" "$OUT/logs" "$OUT/locks"
cd "$RT_ROOT"
tasks=${*:-$(ls envs/*.py | xargs -n1 basename | grep -v '^_' | grep -v __init__ | sed 's/\.py$//')}
GPU_LIST=$(IFS=,; echo "${gpus[*]}")
export RT_PY SW_ROOT CONFIG OUT PER LIMIT GPU_LIST MIN_FREE RESERVE OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
for t in $tasks; do
  for ((p = 0; p < PARTS; p++)); do
    s=$((SEED0 + p * STRIDE))
    [ -f "$OUT/parts/$t.p$s.npz" ] && continue
    echo "$t $s"
  done
done | xargs -P "$MAX" -L 1 bash -c 'mkdir "$OUT/locks/$0.p$1" 2>/dev/null || exit 0
  exec 9> "$OUT/gpu.lock"
  while :; do
    flock 9
    g=$(nvidia-smi -i "$GPU_LIST" --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " -v r="$RESERVE" \
      "BEGIN { n = split(r, a, \",\"); for (i = 1; i <= n; i++) { split(a[i], b, \":\"); res[b[1]] = b[2] } }
       { print \$1, \$2 - res[\$1] }" | sort -k2 -nr | head -1)
    [ "$(pgrep -fc "[c]ollect.py --task")" -lt "$LIMIT" ] && [ "${g#* }" -ge "$MIN_FREE" ] && break
    flock -u 9
    sleep $((20 + RANDOM % 20))
  done
  CUDA_VISIBLE_DEVICES=${g%% *} "$RT_PY" -W ignore "$SW_ROOT/spikewam/robotwin/collect.py" \
  --task "$0" --config "$CONFIG" --episodes "$PER" --seed0 "$1" --tag ".p$1" --out "$OUT/parts" \
  > "$OUT/logs/$0.p$1.log" 2>&1 < /dev/null 9>&- &
  sleep 60
  flock -u 9
  exec 9>&-
  wait'
for t in $tasks; do
  n=$(ls "$OUT"/parts/"$t".p*.npz 2>/dev/null | wc -l)
  [ "$n" = "$PARTS" ] && "$SW_PY" "$SW_ROOT/scripts/robotwin_merge.py" --parts "$OUT/parts" --out "$OUT" --task "$t" \
    --episodes $((PARTS * PER))
done
echo "$(date) collection done"
