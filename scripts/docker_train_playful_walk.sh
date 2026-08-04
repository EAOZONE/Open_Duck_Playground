#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-open-duck-playful-walk:latest}"
container_name="${CONTAINER_NAME:-open-duck-playful-walk}"
gpu_id="${GPU_ID:-1}"
output_dir="${OUTPUT_DIR:-${PWD}/checkpoints/playful_walk}"
warmstart_dir="${WARMSTART_DIR:-${PWD}/checkpoints/warmstart_best_walk}"
num_timesteps="${NUM_TIMESTEPS:-100000000}"
num_evals="${NUM_EVALS:-20}"

mkdir -p "${output_dir}"
test -d "${warmstart_dir}"

docker run --rm \
  --name "${container_name}" \
  --gpus "device=${gpu_id}" \
  --ipc=host \
  --volume "${output_dir}:/runs" \
  --volume "${warmstart_dir}:/warmstart:ro" \
  "${image_name}" \
  playground/open_duck_mini_v2/runner.py \
  --env joystick \
  --task flat_terrain \
  --playful-walk \
  --output_dir /runs \
  --restore_checkpoint_path /warmstart \
  --num_timesteps "${num_timesteps}" \
  --num_evals "${num_evals}" \
  "$@"
