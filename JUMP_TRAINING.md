# Training the Open Duck Mini Excited-Hop Policy

This guide describes how to train the one-shot excited-hop policy on another
computer, evaluate the exported candidates, and bring only the best deployable
policy back into Git.

The jump task is a residual policy: the authored motion in
`playground/open_duck_mini_v2/jump_motion.py` supplies the nominal crouch,
takeoff, head tick, and landing poses, while PPO learns the corrections needed
to make the motion robust in MuJoCo. The policy contract is:

- robot: Open Duck Mini v2;
- observation: 104 floating-point values;
- action: 14 normalized actuator corrections;
- control rate: 50 Hz; and
- episode length: 150 control steps (3 seconds).

An ONNX policy is coupled to the observation layout, jump reference, robot
model, and action scale in the commit that trained it. Record the Git commit
and do not evaluate or deploy the policy against unrelated code without
retesting it.

## 1. Prepare the training computer

A CUDA-capable Linux computer is strongly recommended. The project currently
installs the CUDA 12 build of JAX. Confirm that a compatible NVIDIA driver is
working before installing the Python environment:

```bash
nvidia-smi
```

Clone the repository and check out the exact commit that contains the jump
reference you want to train:

```bash
git clone <repository-url> Open_Duck_Playground
cd Open_Duck_Playground
git checkout <commit-or-branch>
```

Install `uv` if necessary, then create the environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Verify that JAX sees the GPU:

```bash
uv run python -c "import jax; print(jax.devices())"
```

The result should contain a CUDA device. Do not begin a long run if it lists
only `CpuDevice` unless CPU training is intentional.

## 2. Verify the reference before training

Run the fast interface tests:

```bash
uv run python -m unittest tests.test_jump
```

Run the quantitative scripted-reference evaluation:

```bash
uv run python scripts/evaluate_jump.py --controller scripted --episodes 5
```

Every episode should report `"success": true`. You can also inspect the motion
visually:

```bash
uv run playground/open_duck_mini_v2/jump_infer.py --controller scripted
```

Press `j` or Space to request the hop. Fix a failing reference before spending
time on PPO training.

## 3. Run a short training smoke test

The default PPO configuration uses 8,192 parallel environments and is intended
for a reasonably large GPU. Start with a short run to catch CUDA, memory, ONNX
export, and checkpoint-writing problems:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --env jump \
  --task flat_terrain \
  --output_dir checkpoints/jump_smoke \
  --num_timesteps 1000000 \
  --num_envs 1024 \
  --num_evals 2
```

If that runs out of GPU memory, reduce `--num_envs` to 512 or 256. For the full
run, use the largest stable value that fits the GPU. Values divisible by 256
fit the current PPO batch configuration cleanly.

The first JAX compilation may take several minutes. A slow first evaluation is
not evidence that the full run will proceed at the same speed.

## 4. Run the full training job

The baseline run is 150 million environment steps:

```bash
mkdir -p checkpoints/jump

uv run playground/open_duck_mini_v2/runner.py \
  --env jump \
  --task flat_terrain \
  --output_dir checkpoints/jump \
  --num_timesteps 150000000 \
  2>&1 | tee checkpoints/jump/train.log
```

Add `--num_envs <count>` if the default 8,192 environments do not fit. Reducing
`--num_evals` also reduces evaluation overhead, but produces fewer candidate
checkpoints.

Monitor the run from another terminal:

```bash
uv run tensorboard --logdir checkpoints/jump --host 0.0.0.0 --port 6006
```

If port 6006 is exposed beyond a trusted network, protect it with SSH
forwarding or another access-control layer.

Each evaluation save produces two artifacts in `checkpoints/jump/`:

- `<timestamp>_<step>/` is an Orbax training checkpoint that can resume PPO;
- `<timestamp>_<step>.onnx` is the inference policy to evaluate and deploy.

The whole `checkpoints/` tree is intentionally ignored by Git. It may contain
many optimizer checkpoints and event logs; do not force-add the entire tree.

## 5. Resume an interrupted run

Resume from an Orbax checkpoint directory, not from its neighboring ONNX file:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --env jump \
  --task flat_terrain \
  --output_dir checkpoints/jump_resumed \
  --restore_checkpoint_path checkpoints/jump/<timestamp>_<step> \
  --num_timesteps 150000000
```

Use a new output directory so the resumed candidates are easy to distinguish
from the original run.

## 6. Select the best exported policy

Do not select a policy using training step or TensorBoard reward alone. Evaluate
every ONNX candidate using the same inference path used by deployment:

```bash
mkdir -p checkpoints/jump/evaluations

for model in checkpoints/jump/*.onnx; do
  name=$(basename "${model%.onnx}")
  uv run python scripts/evaluate_jump.py \
    --controller onnx \
    --onnx_model_path "$model" \
    --episodes 5 \
    | sed -n '/^{/,$p' \
    > "checkpoints/jump/evaluations/${name}.json"
done
```

Review the results:

```bash
for result in checkpoints/jump/evaluations/*.json; do
  echo "$result"
  python -m json.tool "$result"
done
```

The built-in success gate requires all of the following:

- at least 4 cm of trunk-height gain;
- at least three consecutive 2 ms physics steps with both feet airborne;
- a detected landing;
- minimum upright score above 0.9;
- less than 5 cm of horizontal displacement; and
- at least 0.4 seconds of stable upright contact after landing.

Prefer a model that passes every evaluation. Among passing models, prioritize:

1. larger safety margin in `min_upright`;
2. lower `horizontal_displacement_m`;
3. longer `stable_hold_s`; and
4. adequate rather than extreme height and airtime.

A larger or longer jump is not automatically better. This task is intended to
produce a short, controlled expression that returns safely to standing.

Preview the leading candidates before choosing one:

```bash
uv run playground/open_duck_mini_v2/jump_infer.py \
  --controller onnx \
  --onnx_model_path checkpoints/jump/<candidate>.onnx
```

Press `j` or Space several times and look for foot scuffing, delayed recovery,
head oscillation, or slow drift that is not obvious from a single score.

## 7. Commit only the selected ONNX policy

Copy the winning model to a stable, descriptive location outside the checkpoint
tree:

```bash
mkdir -p playground/open_duck_mini_v2/models
cp checkpoints/jump/<best-candidate>.onnx \
  playground/open_duck_mini_v2/models/excited_hop.onnx
```

Record its identity before committing it:

```bash
sha256sum playground/open_duck_mini_v2/models/excited_hop.onnx
git rev-parse HEAD
```

The repository ignores all `*.onnx` files, so intentionally force-add this one
named release artifact. Add the associated documentation or evaluation JSON
normally:

```bash
git add -f playground/open_duck_mini_v2/models/excited_hop.onnx
git add JUMP_TRAINING.md
git status --short
git diff --cached --stat
git commit -m "Add trained excited-hop policy"
```

At roughly one megabyte, the current policy export is small enough for regular
Git. If future policies become substantially larger, configure Git LFS before
adding them.

Keep the original ignored training directory until the selected ONNX policy has
been copied, hashed, tested from its final path, committed, and backed up. The
ONNX export supports inference but cannot resume training; retain the matching
Orbax checkpoint separately if future fine-tuning may be useful.

## Final release check

After copying the winner, evaluate the exact file that will be committed:

```bash
uv run python scripts/evaluate_jump.py \
  --controller onnx \
  --onnx_model_path playground/open_duck_mini_v2/models/excited_hop.onnx \
  --episodes 5
```

Also confirm its interface:

```bash
uv run python - <<'PY'
import onnxruntime as ort

path = "playground/open_duck_mini_v2/models/excited_hop.onnx"
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print("inputs:", [(value.name, value.shape) for value in session.get_inputs()])
print("outputs:", [(value.name, value.shape) for value in session.get_outputs()])
PY
```

The expected shapes are input `[1, 104]` and output `[1, 14]`.
