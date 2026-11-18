# Reproducing SpikingWAM experiments

The examples use Bash on Linux. Run the LIBERO commands from the repository root. Policy and perception training use CUDA. The research environment used Python 3.10, PyTorch 2.7.1, NumPy 1.26.4 and Transformers 4.56.2.

## Environment and inputs

1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support for your machine. Install NumPy, Transformers, h5py, Pillow and SciPy in the same environment.
2. Follow the installation and asset instructions for the benchmark you want to run. LIBERO and LIBERO-Plus both provide the `libero` Python module. Use a separate interpreter or give the relevant checkout priority on `PYTHONPATH`. [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) · [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
3. Cache [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and point `SW_MINILM` to its `snapshots` directory. For the spiking encoder, set `SW_SPIKE_TEXT` to a directory containing `spike_minilm_d16/model.pt`. The implementation and distillation procedure are in [`spike_text.py`](../spikewam/spike_text.py).
4. For LIBERO training, provide the regenerated HDF5 demonstrations with flattened MuJoCo states and end-effector observations. The extractor expects `<suite>_no_noops_t_256/<task>_demo.hdf5`. [OpenVLA dataset tooling](https://github.com/openvla/openvla)

```bash
export SW_LIBERO=/path/to/LIBERO
export SW_DEMOS=/path/to/libero-hdf5-demos
export SW_SPIKE_TEXT=/path/to/spiking-text-weights
source setup/env.sh
```

Set `SW_PY` before sourcing if you are using a specific interpreter. Set `SW_DATA` and `SW_RUNS` to redirect generated data and checkpoints. To use a benchmark configuration outside its checkout, set `LIBERO_CONFIG_PATH` to the directory containing its `config.yaml`.

## Train a LIBERO policy

Extract simulator-state skeletons from the training demonstrations, then train the world and action branches together. The checkpoint also stores the exponential moving average weights used by the evaluator.

```bash
"$SW_PY" -m scripts.extract_skeletons --workers 8
CUDA_VISIBLE_DEVICES=0 "$SW_PY" -m spikewam.train \
  --run libero_spikewam --model spike --bench libero \
  --steps 60000 --batch 48 \
  --set frame=both geo_attn=1 fine_in=cur text_pool=mean text_encoder=spikeminilm16
```

The independent camera front end trains on rendered demonstration caches. [`perception_cache.py`](../scripts/perception_cache.py) describes the cache format and domain-randomization options. [`train_perception.py`](../scripts/train_perception.py) fits the object-keypoint model. The reported LIBERO experiment used the two-camera `dr4M` front end with 160-pixel inputs, widths `48,96,192,384`, 256 output and hidden channels, and 200,000 training steps.

## Evaluate LIBERO and LIBERO-Plus

Set `POLICY` and `FRONTEND` to checkpoints on your machine. Keep the policy's `stats.json` beside its checkpoint for action and skeleton normalization. The reported LIBERO-family results use the same checkpoints and a one-step readout with fixed episode noise.

```bash
POLICY="$SW_RUNS/libero_spikewam/ckpt_60000.pt"
FRONTEND=/path/to/libero-perception/ckpt.pt

"$SW_PY" -m spikewam.eval_libero \
  --ckpt "$POLICY" --perception "$FRONTEND" \
  --weights ema --k 1 --flow_steps 1 --fixed_noise --workers 8
```

For LIBERO-Plus, supply an instructions JSON with an `instructions` mapping from variant task names to instructions. Some variant names carry nuisance suffixes that should be absent from the instruction text. Install the benchmark's rendering and image-corruption dependencies in the interpreter used for this run. Set `SW_LIBERO_PLUS_CONFIG` to use a configuration directory outside the checkout.

```bash
export SW_LIBERO_PLUS=/path/to/LIBERO-plus
source setup/libero_plus_env.sh

"$SW_PY" -m spikewam.eval_libero_plus \
  --ckpt "$POLICY" --perception "$FRONTEND" \
  --instructions /path/to/libero_plus_instructions.json \
  --weights ema --k 1 --flow_steps 1 --fixed_noise --workers 8
```

LIBERO writes a per-suite evaluation JSON beside the policy checkpoint. LIBERO-Plus writes an episode JSONL and a summary JSON in a run-specific subdirectory and resumes from the episode records.

The reported LIBERO run used 40 workers. Its evaluation seeds depend on how episodes are assigned to workers, so changing the worker count can change the result.

## RoboTwin 2.0

Install RoboTwin and its assets, then run the collector and evaluator from the RoboTwin checkout with its Python environment active. Set the language weight paths described above and keep the policy's `stats.json` beside its checkpoint. The following example evaluates one task. Run across all 50 tasks with both `demo_clean` and `demo_randomized` to cover Easy and Hard.

```bash
SPIKEWAM=/absolute/path/to/SpikingWAM
RT_POLICY=/path/to/robotwin-policy/ckpt.pt
RT_FRONTEND=/path/to/robotwin-perception/ckpt.pt

python "$SPIKEWAM/spikewam/robotwin/evaluate.py" \
  --task place_a2b_left --config demo_clean \
  --ckpt "$RT_POLICY" --perception "$RT_FRONTEND" \
  --out "$SPIKEWAM/runs/robotwin_eval" --test_num 100 \
  --instruction_type unseen --k 3 --flow_steps 10 --consensus medoid
```

The reported policy was trained on 50 clean and approximately 500 randomized expert demonstrations per task. Use [`collect.py`](../spikewam/robotwin/collect.py) to generate skeleton demonstrations, then pass their directory to `spikewam.train --bench robotwin --data`. The evaluation front end estimates object keypoints using three camera views. Its policy uses measured robot joints and forward kinematics for robot keypoints.
