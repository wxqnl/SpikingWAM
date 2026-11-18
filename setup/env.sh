# Source from Bash before running the LIBERO training and evaluation commands.
export SW_ROOT="${SW_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export SW_DATA="${SW_DATA:-$SW_ROOT/data}"
export SW_RUNS="${SW_RUNS:-$SW_ROOT/runs}"

# Reuse the original benchmark environment when it is available beside this checkout.
_sw_legacy_root="$(cd -- "$SW_ROOT/../.." && pwd)/spike-new-vla-icml"
if [[ -x "$_sw_legacy_root/env/bin/python" ]]; then
  export SW_PY="${SW_PY:-$_sw_legacy_root/env/bin/python}"
  export SW_BENCH="${SW_BENCH:-$_sw_legacy_root/benchmarks}"
fi
unset _sw_legacy_root

export SW_PY="${SW_PY:-python}"
export SW_BENCH="${SW_BENCH:-$SW_ROOT/benchmarks}"
export SW_LIBERO="${SW_LIBERO:-$SW_BENCH/LIBERO}"
_sw_legacy_demos=/mnt/data/spoil/workspace/DreamerVLA/data/datasets/processed_data
if [[ -d "$_sw_legacy_demos" ]]; then
  export SW_DEMOS="${SW_DEMOS:-$_sw_legacy_demos}"
fi
unset _sw_legacy_demos
export SW_DEMOS="${SW_DEMOS:-$SW_DATA/demos}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$SW_LIBERO/.libero}"
export PYTHONPATH="$SW_ROOT:$SW_LIBERO${PYTHONPATH:+:$PYTHONPATH}"

export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# The LIBERO benchmark's initial states include pickled NumPy arrays.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SW_MINILM="${SW_MINILM:-$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Long LIBERO runs can retain mesh file descriptors across resets.
ulimit -n 65536 2>/dev/null || true
