# Unified stand, walk, and animation policy

`motion_tracking` trains one 50 Hz asymmetric PPO actor for stable standing,
velocity-conditioned walking, and upright animation tracking. The deployable
actor observes 205 floats and outputs 14 joint-position residuals. Its first
101 fields are the existing walking observation; four 26-float future
references at 0, 0.1, 0.2, and 0.4 seconds form the remaining 104 fields. The
critic is privileged and is never exported.

## Warm-start the walking actor

```bash
uv run python scripts/onnx_to_orbax_warmstart.py \
  --onnx ../Open_Duck_Mini/BEST_WALK_ONNX.onnx \
  --output checkpoints/unified_walk_warmstart \
  --target-observation-size 205 \
  --privileged-observation-size 417 \
  --target-hidden-sizes 512 256 128
```

This copies the existing 101 first-layer rows, zero-initializes the appended
104 reference rows, preserves the rest of the `[512, 256, 128]` Swish actor,
and initializes a new critic.

## Three-stage curriculum

Run from the repository root. Start at 8,192 environments on the RTX 5090; if
memory is insufficient, repeat mechanically with `--num_envs 4096`.

```bash
# 1. Nominal stand/locomotion warm-start, about 100M steps.
uv run python playground/open_duck_mini_v2/runner.py \
  --env motion_tracking --unified-stage locomotion --task flat_terrain \
  --restore_checkpoint_path checkpoints/unified_walk_warmstart \
  --output_dir checkpoints/unified_stage1 --num_timesteps 100000000 \
  --num_evals 21 --video --video-interval-steps 5000000

# 2. Mixed upright motion and learned 0.4 s transitions, about 150M steps.
uv run python playground/open_duck_mini_v2/runner.py \
  --env motion_tracking --unified-stage mixed --task flat_terrain \
  --restore_checkpoint_path checkpoints/unified_stage1/<checkpoint> \
  --output_dir checkpoints/unified_stage2 --num_timesteps 150000000

# 3. Pushes, latency, noise, bias, random dynamics, roughness/backlash, about 100M steps.
uv run python playground/open_duck_mini_v2/runner.py \
  --env motion_tracking --unified-stage sim2real --task rough_terrain_backlash \
  --restore_checkpoint_path checkpoints/unified_stage2/<checkpoint> \
  --output_dir checkpoints/unified_stage3 --num_timesteps 100000000
```

Use `--motion-bundle <path>` to train against a different validated
`open_duck.motion_bundle.v1` file. The default checked-in bundle contains 297
clips: the starter/holdout motions, their train-only augmentations, and 240
direct-frame gait clips.

## Export and evaluation gates

Every 205-input export embeds `open_duck.policy_manifest.v1`, including the
exact observation layout, normalization, joint order, reference offsets,
action scale, control rate, and compatible bundle schema. Verify numerical
parity before rollout:

```bash
uv run python scripts/verify_onnx_parity.py \
  --onnx checkpoints/unified_stage3/<actor>.onnx \
  --checkpoint checkpoints/unified_stage3/<checkpoint>

uv run python scripts/evaluate_motion_matrix.py \
  --onnx checkpoints/unified_stage3/<actor>.onnx \
  --episodes 1000 --steps 1000 \
  --output checkpoints/unified_stage3/evaluation_matrix.json

uv run python scripts/evaluate_motion_tracking.py \
  --onnx checkpoints/unified_stage3/<actor>.onnx \
  --scenario nominal --split holdout --episodes 1000
```

Select a checkpoint from these rollout metrics, not training reward alone.
Promotion still requires the staged bench, tether, soft-floor, individual
motion, repeated-transition, and ten-minute mixed hardware tests in the
project plan. Jump/get-up policies remain separate.

## Periodic progress videos

`--video` follows Persona's fixed-interval training-video pattern without
putting cameras in thousands of training environments. At eligible Brax
evaluation callbacks, it runs a separate deterministic one-environment
showcase and writes an annotated MP4 to:

```text
<output_dir>/videos/motion_showcase_<environment_step>.mp4
```

The default showcase records three seconds each of `stand`, a fixed forward
gait, `head_nod`, and `bow`. Keeping the seed and clips fixed makes progress
directly comparable. The last frame and MP4 path are also added to TensorBoard.
Use `--video-motions`, `--video-length`, and `--video-interval-steps` to change
the contents or cadence. Brax invokes checkpoint/video callbacks only at eval
boundaries, so set `--num_evals` finely enough; 21 evaluations over 100M steps
gives roughly one callback every 5M steps. `--video-strict` makes rendering
failure stop the run instead of printing a warning.

## Resume an interrupted stage

Restore the newest completed checkpoint directory and set the intended global
stage target. The runner reads the step suffix, offsets TensorBoard,
checkpoint, ONNX, and video names, and trains only the remaining steps:

```bash
uv run python playground/open_duck_mini_v2/runner.py \
  --env motion_tracking --unified-stage locomotion --task flat_terrain \
  --restore_checkpoint_path checkpoints/unified_stage1/<checkpoint_step> \
  --target-total-timesteps 100000000 \
  --output_dir checkpoints/unified_stage1 --video \
  --video-interval-steps 5000000 --video-length 150
```

These callback checkpoints preserve actor, critic, and observation statistics,
but not optimizer momentum or simulator episode state. Resumption therefore
continues the learned policy safely but is not bit-exact.
