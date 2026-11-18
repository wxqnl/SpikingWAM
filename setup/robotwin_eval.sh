#!/usr/bin/env bash
# Closed-loop RoboTwin 2.0 evaluation (spikewam/robotwin/evaluate.py): one process per (run, task), round robin over
# the GPUs given. Processes of the same task draw the same seeds and instructions, so the runs stay paired.
# Usage: GPUS="1 2 3 4 5 6 7" CONFIG=demo_clean setup/robotwin_eval.sh <run> [<run> ...]
# Writes runs/robotwin_eval/<run>/<config>/<task>.json (resumes); skips tasks whose log shows all seeds done.
# PAIRED=1: one process per task runs all the given runs on each seed (one expert check per seed), writing
# runs/robotwin_eval/<run1>+<run2>/<config>/<task>.json.
# INSTR=seen: instructions of RoboTwin's seen type (the training templates) instead of unseen; the result dir gets
# the suffix _seen.
# CKPT=ckpt_120000.pt: the checkpoint file of each run (default ckpt_60000.pt); <run>:<file> overrides it for one run.
# K=36: plans sample K chunks for the consensus (default 3); results are named <run>_k36, in <run>_k36/.
# RECORD=1: also keep every plan's joints, skeleton and joint targets (evaluate.py --record); the result dir gets the
# suffix _rec.
# INIT_NOISE=3: displace the arm joints by N(0, 3 degrees) before the policy starts (evaluate.py --init_noise); the
# result dir gets the suffix _init3. INIT_NOISE=0: the same initial move without displacement (control), suffix _init0.
# INFER="--flow_steps 1 --fixed_noise": further inference options of evaluate.py; TAG=s1f: result dir suffix _s1f.
set -uo pipefail
RT_ROOT=${RT_ROOT:-/mnt/data/user01/RoboTwin}
RT_PY=${RT_PY:-/mnt/data/user01/envs/robotwin_sw/bin/python}
SW_ROOT="${SW_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG=${CONFIG:-demo_clean}
TEST_NUM=${TEST_NUM:-100}
INSTR=${INSTR:-unseen}
K=${K:-3}
CKPT=${CKPT:-ckpt_60000.pt}
export SW_MINILM="${SW_MINILM:-$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$RT_ROOT"
tasks=${TASKS:-$(ls envs/*.py | xargs -n1 basename | grep -v '^_' | grep -v __init__ | sed 's/\.py$//')}
read -ra gpus <<< "${GPUS:-1 2 3 4 5 6 7}"
if [ "${PAIRED:-0}" = 1 ]; then groups=("$*"); else groups=("$@"); fi
i=0
for group in "${groups[@]}"; do
  read -ra runs <<< "$group"
  ckpts=()
  for run in "${runs[@]}"; do
    if [[ $run == *:* ]]; then ckpts+=("$SW_ROOT/runs/${run%%:*}/${run#*:}"); else ckpts+=("$SW_ROOT/runs/$run/$CKPT"); fi
  done
  names=("${runs[@]%%:*}")
  out=$SW_ROOT/runs/robotwin_eval/$(IFS=+; echo "${names[*]}")
  [ "$K" = 3 ] || out=${out}_k$K
  [ "$INSTR" = unseen ] || out=${out}_$INSTR
  [ "${RECORD:-0}" = 1 ] && out=${out}_rec
  [ -z "${INIT_NOISE:-}" ] || out=${out}_init${INIT_NOISE}
  [ -z "${TAG:-}" ] || out=${out}_$TAG
  mkdir -p "$out/logs"
  for t in $tasks; do
    log=$out/logs/${CONFIG}_$t.log
    [ -f "$log" ] && grep -q "($TEST_NUM/$TEST_NUM," "$log" && continue
    g=${gpus[$((i % ${#gpus[@]}))]}
    i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$g setsid nohup "$RT_PY" -W ignore "$SW_ROOT/spikewam/robotwin/evaluate.py" --task "$t" \
      --config "$CONFIG" --ckpt "${ckpts[@]}" --out "$out" --test_num "$TEST_NUM" \
      --instruction_type "$INSTR" --k "$K" ${INFER:-} $([ "${RECORD:-0}" = 1 ] && echo --record) --init_noise "${INIT_NOISE:--1}" \
      >> "$log" 2>&1 < /dev/null &
  done
done
echo "$i evaluators started"
