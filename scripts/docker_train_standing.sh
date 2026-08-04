#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-open-duck-standing:latest}"
container_name="${CONTAINER_NAME:-open-duck-standing}"
gpu_id="${GPU_ID:-1}"
output_dir="${OUTPUT_DIR:-${PWD}/checkpoints/standing}"
num_timesteps="${NUM_TIMESTEPS:-100000000}"
num_evals="${NUM_EVALS:-20}"

mkdir -p "${output_dir}"

docker run --rm \
  --name "${container_name}" \
  --gpus "device=${gpu_id}" \
  --ipc=host \
  --volume "${output_dir}:/runs" \
  "${image_name}" \
  playground/open_duck_mini_v2/runner.py \
  --env standing \
  --task flat_terrain \
  --output_dir /runs \
  --num_timesteps "${num_timesteps}" \
  --num_evals "${num_evals}" \
  "$@"
