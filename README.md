# Open Duck Playground

# Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Training

If you want to use the [imitation reward](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf), you can generate reference motion with [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

Then copy `polynomial_coefficients.pkl` in `playground/<robot>/data/`

You'll also have to set `USE_IMITATION_REWARD=True` in it's `joystick.py` file

Run: 

```bash
uv run playground/<robot>/runner.py 
```

## Tensorboard

```bash
uv run tensorboard --logdir=<yourlogdir>
```

# Inference 

Infer mujoco

(for now this is specific to open_duck_mini_v2)

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_.onnx>
```

Add a deliberately excited little walk on top of the learned gait with:

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py \
  -o <path_to_.onnx> \
  --fixed-command 0.20 0 0 \
  --expressive-walk
```

This adds a two-beat head bounce, a small spring in both knees, and a quick
head roll/yaw that makes the rigid antennae appear to wiggle. The flourish
automatically fades out when the movement command returns to zero. Tune it
with `--head-bobble-amplitude`, `--antenna-wiggle-amplitude`, and
`--step-bounce-amplitude`. Press `j` or Space in the MuJoCo viewer to stop,
settle, perform the excited hop, and return to walking. In Xbox mode, press B
to request the same sequence; A remains the stop/dead-man control.

## Playful high-step policy

`--playful-walk` fine-tunes the normal walking task toward 45 mm foot
clearance and one alternating 65 mm accent step every three gait cycles. The
accent cue reuses the existing head-roll command slot, so the 101-observation,
14-action walking network remains compatible with existing policies.

An exported ONNX actor cannot directly resume PPO because it does not contain a
critic or optimizer state. Reconstruct a compatible warm start from the actor
weights and exact observation normalization with:

```bash
JAX_PLATFORMS=cpu uv run scripts/onnx_to_orbax_warmstart.py \
  --onnx ../Open_Duck_Mini/BEST_WALK_ONNX.onnx \
  --output checkpoints/warmstart_best_walk
```

Then fine-tune it with the conservative playful-walk PPO settings:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --env joystick --task flat_terrain --playful-walk \
  --restore_checkpoint_path checkpoints/warmstart_best_walk \
  --output_dir checkpoints/playful_walk \
  --num_timesteps 100000000 --num_evals 20
```

The selected 70.3M-step policy reached the highest evaluation reward in the
100M-step run and is checked in as
`playground/open_duck_mini_v2/models/playful_high_step_walk.onnx`. When running
it, provide the same accent cue:

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py \
  -o playground/open_duck_mini_v2/models/playful_high_step_walk.onnx \
  --fixed-command 0.20 0 0 --playful-policy
```

## Excited-hop policy

The jump environment trains a short, one-shot excited hop with a 104-element
observation and 14 actuator outputs. Its reference includes a small head nod,
a brief airborne phase, and an absorbed landing. The scripted controller is
useful for checking the exact reference without a trained checkpoint:

See [JUMP_TRAINING.md](JUMP_TRAINING.md) for the complete remote-machine
training, checkpoint evaluation, policy selection, and release workflow.

```bash
uv run playground/open_duck_mini_v2/jump_infer.py --controller scripted
```

Press `j` or Space in the MuJoCo viewer to request one jump. Train the PPO
version with the same runner used by the walking tasks:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --env jump --task flat_terrain \
  --output_dir checkpoints/jump \
  --num_timesteps 150000000
```

Run a trained jump ONNX policy with:

```bash
uv run playground/open_duck_mini_v2/jump_infer.py \
  --controller onnx --onnx_model_path checkpoints/jump/<checkpoint>.onnx
```

## Face-down get-up policy

The get-up evaluation resets the robot face-down, follows a four-second reference
that folds the legs, plants both feet, rises through a crouch, and then holds
the normal standing pose.  The torso and head use inset collision proxies so
fall recovery has real floor contacts; these proxies remain clear of the floor
during normal walking and jumping.

During training, 70% of environments use reference-state initialization at a
random point in the animation so PPO learns the crouch and stand before
stitching the full recovery together.  The evaluation environment always uses
the requested phase-zero face-down reset.

Preview the kinematic reference animation (press `g` or Space to replay it):

```bash
uv run playground/open_duck_mini_v2/getup_infer.py --controller reference
```

Train the residual PPO policy from face-down resets:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --env getup --task flat_terrain \
  --output_dir checkpoints/getup \
  --num_timesteps 150000000
```

Train the goal-only policy with the balanced-foot reward, corrective leg cost,
and late-stage reset curriculum:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --env getup --task flat_terrain --goal-only-getup \
  --getup-ablation combined \
  --output_dir checkpoints/getup_goal_only \
  --num_timesteps 150000000
```

The `balanced` and `corrective` ablations respectively disable the corrective
cost, or keep the cost while using face-down resets only.  Compare exported
policies with the strict two-second standing evaluator:

```bash
uv run python scripts/evaluate_getup.py \
  --onnx-model-path checkpoints/getup_goal_only/<checkpoint>.onnx \
  --episodes 64
```

For GPU training, build the locked Docker environment and keep checkpoints in
a host-mounted directory:

```bash
docker build \
  --build-arg USER_ID="$(id -u)" \
  --build-arg GROUP_ID="$(id -g)" \
  -t open-duck-getup:latest .

GPU_ID=1 \
OUTPUT_DIR="$PWD/checkpoints/getup" \
scripts/docker_train_getup.sh
```

Pass normal runner options after the script name.  For example, a checkpoint
already below `OUTPUT_DIR` can be resumed with
`--restore_checkpoint_path /runs/<checkpoint-directory>`.

Run an exported get-up policy in MuJoCo:

```bash
uv run playground/open_duck_mini_v2/getup_infer.py \
  --controller onnx --onnx_model_path checkpoints/getup/<checkpoint>.onnx
```

# Documentation

## Project structure : 

```
.
├── pyproject.toml
├── README.md
├── playground
│   ├── common
│   │   ├── export_onnx.py
│   │   ├── onnx_infer.py
│   │   ├── poly_reference_motion.py
│   │   ├── randomize.py
│   │   ├── rewards.py
│   │   └── runner.py
│   ├── open_duck_mini_v2
│   │   ├── base.py
│   │   ├── data
│   │   │   └── polynomial_coefficients.pkl
│   │   ├── joystick.py
│   │   ├── mujoco_infer.py
│   │   ├── constants.py
│   │   ├── runner.py
│   │   └── xmls
│   │       ├── assets
│   │       ├── open_duck_mini_v2_no_head.xml
│   │       ├── open_duck_mini_v2.xml
│   │       ├── scene_mjx_flat_terrain.xml
│   │       ├── scene_mjx_rough_terrain.xml
│   │       └── scene.xml
```

## Adding a new robot

Create a new directory in `playground` named after `<your robot>`. You can copy the `open_duck_mini_v2` directory as a starting point.

You will need to:
- Edit `base.py`: Mainly renaming stuff to match you robot's name
- Edit `constants.py`: specify the names of some important geoms, sensors etc
  - In your `mjcf`, you'll probably have to add some sites, name some bodies/geoms and add the sensors. Look at how we did it for `open_duck_mini_v2`
- Add your `mjcf` assets in `xmls`. 
- Edit `joystick.py` : to choose the rewards you are interested in
  - Note: for now there is still some hard coded values etc. We'll improve things on the way
- Edit `runner.py`



# Notes

Inspired from https://github.com/kscalelabs/mujoco_playground


## Current win

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```
