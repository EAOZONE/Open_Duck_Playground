"""Periodic deterministic rollout videos for Brax training callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jp
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class VideoResult:
    path: Path
    preview: np.ndarray
    frame_count: int


class TrainingVideoRecorder:
    """Record a fixed motion showcase without wrapping the training env."""

    def __init__(
        self,
        env,
        output_dir: str | Path,
        *,
        interval_steps: int,
        rollout_steps: int,
        motions: Sequence[str],
        fps: float = 50.0,
        width: int = 480,
        height: int = 360,
        camera: str | None = None,
    ) -> None:
        if interval_steps <= 0 or rollout_steps <= 0:
            raise ValueError("video interval and rollout length must be positive")
        if not motions:
            raise ValueError("at least one showcase motion is required")
        unknown = [name for name in motions if name not in env.motion_names]
        if unknown:
            raise ValueError(f"unknown video motions: {', '.join(unknown)}")
        self.env = env
        self.output_dir = Path(output_dir) / "videos"
        self.interval_steps = int(interval_steps)
        self.rollout_steps = int(rollout_steps)
        self.motions = tuple(motions)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.camera = camera
        self.last_recorded_step: int | None = None

    def should_record(self, current_step: int) -> bool:
        return self.last_recorded_step is None or (
            current_step - self.last_recorded_step >= self.interval_steps
        )

    def _force_motion(self, state, name: str):
        motion_index = self.env.motion_names.index(name)
        stand_index = self.env.motion_names.index("stand")
        state.info["motion_index"] = jp.asarray(motion_index)
        state.info["motion_time"] = jp.asarray(0.0)
        state.info["motion_age"] = jp.asarray(0.0)
        state.info["blend_time"] = jp.asarray(0.0)
        state.info["blend_from"] = self.env._raw_reference(
            jp.asarray(stand_index), jp.asarray(0.0)
        )
        state.info["encoder_bias"] = jp.zeros(14)
        command, phase = self.env._command_and_phase(state.info)
        state.info["command"] = command
        state.info["imitation_phase"] = phase
        contact = self.env._contacts(state.data)
        return state.replace(obs=self.env._get_obs(state.data, state.info, contact))

    @staticmethod
    def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((0, 0), label)
        width = box[2] - box[0]
        height = box[3] - box[1]
        draw.rounded_rectangle((8, 8, width + 24, height + 20), radius=5, fill=(0, 0, 0))
        draw.text((16, 12), label, fill=(255, 255, 255))
        return np.asarray(image)

    def record(self, current_step: int, make_policy: Callable, params) -> VideoResult | None:
        if not self.should_record(current_step):
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        policy = jax.jit(
            make_policy((params[0], params[1].policy), deterministic=True)
        )
        step_fn = jax.jit(self.env.step)
        all_frames: list[np.ndarray] = []

        for motion_number, motion_name in enumerate(self.motions):
            state = self.env.reset(jax.random.PRNGKey(10_000 + motion_number))
            state = self._force_motion(state, motion_name)
            trajectory = [state]
            rng = jax.random.PRNGKey(20_000 + motion_number)
            for _ in range(self.rollout_steps - 1):
                rng, action_rng = jax.random.split(rng)
                action, _ = policy(state.obs, action_rng)
                state = step_fn(state, action)
                trajectory.append(state)
                if bool(state.done):
                    break
            frames = self.env.render(
                trajectory,
                height=self.height,
                width=self.width,
                camera=self.camera,
            )
            for frame_index, frame in enumerate(frames):
                all_frames.append(
                    self._annotate(
                        np.asarray(frame),
                        f"step {current_step:,}  |  {motion_name}  |  "
                        f"t={frame_index / self.fps:.2f}s",
                    )
                )

        path = self.output_dir / f"motion_showcase_{current_step:012d}.mp4"
        media.write_video(path, all_frames, fps=self.fps)
        self.last_recorded_step = int(current_step)
        return VideoResult(path=path, preview=all_frames[-1], frame_count=len(all_frames))
