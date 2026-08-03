# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/trainer \
    JAX_PLATFORMS=cuda \
    MUJOCO_GL=egl \
    PATH=/workspace/.venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# MuJoCo loads these libraries even for headless MJX training.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        g++ \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglfw3 \
        libx11-6 \
        libxcursor1 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${GROUP_ID}" trainer \
    && useradd --create-home --uid "${USER_ID}" --gid "${GROUP_ID}" trainer \
    && install -d --owner="${USER_ID}" --group="${GROUP_ID}" /workspace

WORKDIR /workspace

# Keep dependency installation cached when only source files change.
COPY --chown=${USER_ID}:${GROUP_ID} pyproject.toml uv.lock README.md ./
USER trainer
RUN uv sync --frozen --no-install-project --no-dev
# Open Duck uses only the checked-in robot assets.  The upstream package checks
# for this directory at import time and otherwise downloads the 1.7 GB generic
# MuJoCo Menagerie, which this training task does not need.
RUN mkdir -p .venv/lib/python3.13/site-packages/mujoco_playground/external_deps/mujoco_menagerie

COPY --chown=${USER_ID}:${GROUP_ID} . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev"]
CMD ["playground/open_duck_mini_v2/runner.py", "--env", "getup", "--task", "flat_terrain", "--output_dir", "/runs", "--num_timesteps", "150000000", "--num_evals", "15"]
