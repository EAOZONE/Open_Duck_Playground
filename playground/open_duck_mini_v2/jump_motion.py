"""Shared trajectory and one-shot controller for Open Duck jumps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


JUMP_DURATION = 2.0
# The nominal Open Duck hop is shorter than the nominal 0.50 s ballistic
# window.  Start absorption when the model actually approaches the floor.
LANDING_START = 0.75
RECOVERY_START = 1.05
JUMP_PHASES = {
    "crouch": (0.0, 0.30),
    "takeoff": (0.30, 0.55),
    "flight": (0.55, 1.05),
    "landing": (LANDING_START, 1.05),
    "recovery": (RECOVERY_START, 2.00),
}

# Actuator order is the order in open_duck_mini_v2.xml.  Head offsets remain
# zero so the scripted and learned jump controllers keep the head at home.
CROUCH_OFFSETS = np.array(
    [0.0, 0.0, 0.12, 0.45, -0.18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.12, 0.45, 0.18],
    dtype=np.float32,
)
TAKEOFF_OFFSETS = np.array(
    [0.0, 0.0, -0.18, -0.55, 0.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18, -0.55, -0.22],
    dtype=np.float32,
)
FLIGHT_OFFSETS = np.array(
    [0.0, 0.0, 0.05, 0.25, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 0.25, 0.05],
    dtype=np.float32,
)


def _smoothstep(value: Any, xp: Any) -> Any:
    value = xp.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _lerp(start: Any, end: Any, value: Any, xp: Any) -> Any:
    return start + (end - start) * _smoothstep(value, xp)


def trajectory_pose(time_s: Any, home_pose: Any, xp: Any = np) -> Any:
    """Return the desired 14-actuator pose at ``time_s`` seconds.

    The implementation uses array operations rather than Python branching so
    it can be used both by NumPy inference code and by a JAX environment.
    """

    home_pose = xp.asarray(home_pose)
    crouch = home_pose + xp.asarray(CROUCH_OFFSETS, dtype=home_pose.dtype)
    takeoff = home_pose + xp.asarray(TAKEOFF_OFFSETS, dtype=home_pose.dtype)
    flight = home_pose + xp.asarray(FLIGHT_OFFSETS, dtype=home_pose.dtype)

    t = xp.asarray(time_s)
    home_to_crouch = _lerp(home_pose, crouch, t / 0.15, xp)
    crouch_hold = crouch
    crouch_to_takeoff = _lerp(crouch, takeoff, (t - 0.30) / 0.25, xp)
    takeoff_to_flight = _lerp(takeoff, flight, (t - 0.55) / 0.50, xp)
    flight_to_crouch = _lerp(flight, crouch, (t - LANDING_START) / 0.30, xp)
    crouch_to_home = _lerp(crouch, home_pose, (t - RECOVERY_START) / 0.95, xp)

    pose = xp.where((t[..., None] < 0.15), home_to_crouch, crouch_hold)
    pose = xp.where((t[..., None] >= 0.30), crouch_to_takeoff, pose)
    pose = xp.where((t[..., None] >= 0.55), takeoff_to_flight, pose)
    pose = xp.where((t[..., None] >= LANDING_START), flight_to_crouch, pose)
    pose = xp.where((t[..., None] >= RECOVERY_START), crouch_to_home, pose)
    pose = xp.where((t[..., None] >= JUMP_DURATION), home_pose, pose)
    return pose


def clip_pose(pose: Any, lower: Any, upper: Any, xp: Any = np) -> Any:
    """Clip a trajectory pose to actuator position limits."""

    return xp.clip(pose, xp.asarray(lower), xp.asarray(upper))


def phase_features(time_s: Any, xp: Any = np) -> Any:
    """Return ``[active, sin(phase), cos(phase)]`` for an observation."""

    t = xp.asarray(time_s)
    normalized = xp.clip(t / JUMP_DURATION, 0.0, 1.0)
    active = (t < JUMP_DURATION).astype(xp.asarray(normalized).dtype)
    angle = normalized * 2.0 * xp.pi
    return xp.stack([active, xp.sin(angle), xp.cos(angle)], axis=-1)


@dataclass
class JumpMotionController:
    """A deterministic one-shot jump phase clock and target generator."""

    elapsed_s: float = JUMP_DURATION

    @property
    def active(self) -> bool:
        return self.elapsed_s < JUMP_DURATION

    def request_jump(self) -> bool:
        """Arm one jump; return whether a new jump was accepted."""

        if self.active:
            return False
        self.elapsed_s = 0.0
        return True

    def reset(self) -> None:
        self.elapsed_s = JUMP_DURATION

    def advance(self, dt: float) -> None:
        if self.active:
            self.elapsed_s = min(JUMP_DURATION, self.elapsed_s + float(dt))

    def target(
        self,
        home_pose: np.ndarray,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.active:
            target = np.asarray(home_pose, dtype=np.float32).copy()
        else:
            target = np.asarray(trajectory_pose(self.elapsed_s, home_pose), dtype=np.float32)
        if lower is not None and upper is not None:
            target = np.clip(target, lower, upper)
        return target

    def features(self) -> np.ndarray:
        return np.asarray(phase_features(self.elapsed_s), dtype=np.float32)
