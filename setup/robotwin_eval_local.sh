#!/usr/bin/env bash
# setup/robotwin_eval.sh one task at a time on the local GPUs: a task starts on whichever GPU in GPUS has more than
# FREE_MB of free memory, at most MAX evaluators at once (the local machine has two GPUs, not enough for all 50 tasks
# at once). Same environment variables as robotwin_eval.sh (CONFIG, CKPT, K, PAIRED, INSTR).
# Usage: GPUS="6 7" CKPT=ckpt_120000.pt setup/robotwin_eval_local.sh <run> [<run> ...]
set -uo pipefail
SW_ROOT="${SW_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
RT_ROOT=${RT_ROOT:-/mnt/data/user01/RoboTwin}
MAX=${MAX:-40}
FREE_MB=${FREE_MB:-10000}
read -ra gpus <<< "${GPUS:-6 7}"
for t in ${TASKS:-$(ls "$RT_ROOT"/envs/*.py | xargs -n1 basename | grep -v '^_' | grep -v __init__ | sed 's/\.py$//')}; do
  while :; do
    best= most=0
    for g in "${gpus[@]}"; do
      free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      [ "$free" -gt "$most" ] && best=$g most=$free
    done
    [ "$most" -gt "$FREE_MB" ] && [ "$(pgrep -fc '[r]obotwin/evaluate.py')" -lt "$MAX" ] && break
    sleep 30
  done
  GPUS=$best TASKS=$t "$SW_ROOT/setup/robotwin_eval.sh" "$@" > /dev/null
  sleep 20
done
echo "$(date) all tasks started"
