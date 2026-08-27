"""Read and sample versioned Open Duck direct-frame motion bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


BUNDLE_SCHEMA = "open_duck.motion_bundle.v1"
REFERENCE_OFFSETS_SECONDS = (0.0, 0.1, 0.2, 0.4)
REFERENCE_WIDTH = 26
LEGACY_OBSERVATION_SIZE = 101
UNIFIED_OBSERVATION_SIZE = 205
CONTROLLED_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
JOINT_LIMITS = np.deg2rad(
    np.asarray(
        [
            (-30, 30), (-25, 25), (-70, 30), (-90, 90), (-90, 90),
            (-20, 65), (-45, 45), (-160, 160), (-30, 30),
            (-30, 30), (-25, 25), (-30, 70), (-90, 90), (-90, 90),
        ],
        dtype=np.float32,
    )
)


def observation_layout() -> list[dict[str, object]]:
    """Machine-readable actor layout embedded in every unified ONNX."""

    return [
        {"name": "gyro", "start": 0, "width": 3},
        {"name": "accelerometer", "start": 3, "width": 3},
        {"name": "command_local_vx_vy_yaw_rate", "start": 6, "width": 3},
        {"name": "current_head_targets", "start": 9, "width": 4},
        {"name": "joint_position_from_home", "start": 13, "width": 14},
        {"name": "joint_velocity_scaled_0.05", "start": 27, "width": 14},
        {"name": "action_t_minus_1", "start": 41, "width": 14},
        {"name": "action_t_minus_2", "start": 55, "width": 14},
        {"name": "action_t_minus_3", "start": 69, "width": 14},
        {"name": "previous_motor_targets", "start": 83, "width": 14},
        {"name": "foot_contacts", "start": 97, "width": 2},
        {"name": "motion_phase_cos_sin", "start": 99, "width": 2},
        {
            "name": "future_reference_samples",
            "start": 101,
            "width": 104,
            "offsets_seconds": list(REFERENCE_OFFSETS_SECONDS),
            "sample_fields": [
                ["joint_targets", 14], ["root_height", 1], ["body_up_vector", 3],
                ["local_linear_velocity", 3], ["local_angular_velocity", 3],
                ["foot_contacts", 2],
            ],
        },
    ]


class MotionBundleError(ValueError):
    pass


@dataclass(frozen=True)
class MotionClip:
    name: str
    kind: str
    split: str
    loop: bool
    fps: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_height: np.ndarray
    projected_gravity: np.ndarray
    local_linear_velocity: np.ndarray
    local_angular_velocity: np.ndarray
    foot_contacts: np.ndarray
    antenna_position: np.ndarray

    @property
    def frame_count(self) -> int:
        return len(self.joint_position)

    @property
    def duration(self) -> float:
        return (self.frame_count - 1) / self.fps

    def _frame(self, time_s: float) -> float:
        frame = max(0.0, float(time_s) * self.fps)
        return frame % max(self.frame_count - 1, 1) if self.loop else min(frame, self.frame_count - 1)

    def sample(self, time_s: float) -> dict[str, np.ndarray | float]:
        frame = self._frame(time_s)
        left = int(np.floor(frame))
        right = min(left + 1, self.frame_count - 1)
        alpha = np.float32(frame - left)

        def linear(values: np.ndarray) -> np.ndarray:
            return ((1 - alpha) * values[left] + alpha * values[right]).astype(np.float32)

        contact_index = left if alpha < 0.5 else right
        return {
            "joint_position": linear(self.joint_position),
            "joint_velocity": linear(self.joint_velocity),
            "root_height": float(linear(self.root_height[:, None])[0]),
            "projected_gravity": linear(self.projected_gravity),
            "local_linear_velocity": linear(self.local_linear_velocity),
            "local_angular_velocity": linear(self.local_angular_velocity),
            "foot_contacts": self.foot_contacts[contact_index].astype(np.float32),
            "antenna_position": linear(self.antenna_position),
        }

    def reference_vector(self, time_s: float) -> np.ndarray:
        value = self.sample(time_s)
        return np.concatenate(
            [
                value["joint_position"],
                np.asarray([value["root_height"]], dtype=np.float32),
                value["projected_gravity"],
                value["local_linear_velocity"],
                value["local_angular_velocity"],
                value["foot_contacts"],
            ]
        ).astype(np.float32)

    def reference_window(
        self, time_s: float, offsets: Sequence[float] = REFERENCE_OFFSETS_SECONDS
    ) -> np.ndarray:
        return np.concatenate([self.reference_vector(time_s + offset) for offset in offsets])

    def phase_features(self, time_s: float) -> np.ndarray:
        phase = self._frame(time_s) / max(self.frame_count - 1, 1)
        return np.asarray([np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)], dtype=np.float32)


class MotionBundle:
    def __init__(self, clips: Sequence[MotionClip], manifest: dict):
        self.clips = tuple(clips)
        self.manifest = dict(manifest)
        self.by_name = {clip.name: clip for clip in self.clips}
        if len(self.by_name) != len(self.clips):
            raise MotionBundleError("motion names must be unique")

    @classmethod
    def load(cls, path: str | Path) -> "MotionBundle":
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest"].item()))
            if manifest.get("schema") != BUNDLE_SCHEMA:
                raise MotionBundleError(f"expected bundle schema {BUNDLE_SCHEMA!r}")
            if tuple(manifest.get("joint_names", ())) != CONTROLLED_JOINTS:
                raise MotionBundleError("bundle joint order does not match the 14-actuator contract")
            if int(manifest.get("reference_width", -1)) != REFERENCE_WIDTH:
                raise MotionBundleError("bundle reference width must be 26")
            if float(manifest.get("fps", 0.0)) != 50.0:
                raise MotionBundleError("bundle control rate must be 50 Hz")
            offsets = np.asarray(data["offsets"], dtype=np.int64)
            names = np.asarray(data["names"])
            kinds = np.asarray(data["kinds"])
            splits = np.asarray(data["splits"]) if "splits" in data else np.full(len(names), "train")
            loops = np.asarray(data["loops"], dtype=np.bool_)
            if offsets.shape != (len(names) + 1,) or offsets[0] != 0:
                raise MotionBundleError("invalid clip offsets")

            arrays = {
                name: np.asarray(data[name], dtype=np.float32)
                for name in (
                    "joint_position", "joint_velocity", "root_height", "projected_gravity",
                    "local_linear_velocity", "local_angular_velocity", "foot_contacts", "antenna_position",
                )
            }
            total = int(offsets[-1])
            if any(len(value) != total for value in arrays.values()):
                raise MotionBundleError("bundle arrays have inconsistent frame counts")
            expected_tails = {
                "joint_position": (14,), "joint_velocity": (14,), "root_height": (),
                "projected_gravity": (3,), "local_linear_velocity": (3,),
                "local_angular_velocity": (3,), "foot_contacts": (2,),
                "antenna_position": (2,),
            }
            for field, tail in expected_tails.items():
                if arrays[field].shape[1:] != tail or not np.all(np.isfinite(arrays[field])):
                    raise MotionBundleError(f"bundle field {field!r} is invalid")
            up_norm = np.linalg.norm(arrays["projected_gravity"], axis=1)
            if not np.allclose(up_norm, 1.0, atol=1.0e-3, rtol=0.0):
                raise MotionBundleError("bundle body-up vectors must be normalized")
            if np.max(np.abs(arrays["joint_velocity"])) > 4.0 + 1.0e-4:
                raise MotionBundleError("bundle exceeds the 4 rad/s target velocity limit")
            if np.any(arrays["joint_position"] < JOINT_LIMITS[:, 0] - 1.0e-4) or np.any(
                arrays["joint_position"] > JOINT_LIMITS[:, 1] + 1.0e-4
            ):
                raise MotionBundleError("bundle contains a target outside logical joint limits")
            clips = []
            for index, name in enumerate(names):
                frame_slice = slice(int(offsets[index]), int(offsets[index + 1]))
                if frame_slice.stop - frame_slice.start < 2:
                    raise MotionBundleError(f"clip {name!s} has fewer than two frames")
                clips.append(
                    MotionClip(
                        name=str(name), kind=str(kinds[index]), split=str(splits[index]), loop=bool(loops[index]),
                        fps=float(manifest["fps"]),
                        **{key: value[frame_slice].copy() for key, value in arrays.items()},
                    )
                )
        return cls(clips, manifest)

    def get(self, name: str) -> MotionClip:
        try:
            return self.by_name[name]
        except KeyError as exc:
            raise MotionBundleError(f"unknown motion {name!r}; available: {', '.join(sorted(self.by_name))}") from exc

    def padded_arrays(self) -> dict[str, np.ndarray]:
        """Return fixed-shape arrays suitable for conversion to JAX constants."""

        maximum = max(clip.frame_count for clip in self.clips)
        result: dict[str, np.ndarray] = {
            "lengths": np.asarray([clip.frame_count for clip in self.clips], dtype=np.int32),
            "loops": np.asarray([clip.loop for clip in self.clips], dtype=np.bool_),
            "kinds": np.asarray([{"stand": 0, "locomotion": 1, "animation": 2}.get(clip.kind, 2) for clip in self.clips], dtype=np.int32),
            "splits": np.asarray([0 if clip.split == "train" else 1 for clip in self.clips], dtype=np.int32),
        }
        fields = (
            "joint_position", "joint_velocity", "root_height", "projected_gravity",
            "local_linear_velocity", "local_angular_velocity", "foot_contacts", "antenna_position",
        )
        for field in fields:
            examples = [getattr(clip, field) for clip in self.clips]
            padded = np.zeros((len(self.clips), maximum) + examples[0].shape[1:], dtype=np.float32)
            for index, values in enumerate(examples):
                padded[index, : len(values)] = values
                padded[index, len(values) :] = values[-1]
            result[field] = padded
        return result


def make_policy_manifest(
    observation_mean: np.ndarray | None = None,
    observation_std: np.ndarray | None = None,
) -> dict:
    manifest = {
        "schema": "open_duck.policy_manifest.v1",
        "control_frequency_hz": 50,
        "action_scale": 0.25,
        "action_size": 14,
        "observation_size": UNIFIED_OBSERVATION_SIZE,
        "legacy_prefix_size": LEGACY_OBSERVATION_SIZE,
        "reference_offsets_seconds": list(REFERENCE_OFFSETS_SECONDS),
        "reference_width": REFERENCE_WIDTH,
        "joint_names": list(CONTROLLED_JOINTS),
        "motion_bundle_schema": BUNDLE_SCHEMA,
        "observation_layout": observation_layout(),
        "action_interpretation": "joint_position_residual_from_home",
    }
    if observation_mean is not None:
        manifest["observation_mean"] = np.asarray(observation_mean, dtype=np.float32).tolist()
    if observation_std is not None:
        manifest["observation_std"] = np.asarray(observation_std, dtype=np.float32).tolist()
    return manifest


def validate_policy_manifest(manifest: dict) -> None:
    expected = make_policy_manifest()
    for key in (
        "schema", "control_frequency_hz", "action_scale", "action_size", "observation_size",
        "legacy_prefix_size", "reference_offsets_seconds", "reference_width", "joint_names",
        "motion_bundle_schema", "observation_layout", "action_interpretation",
    ):
        if manifest.get(key) != expected[key]:
            raise MotionBundleError(f"policy manifest field {key!r} is incompatible")
