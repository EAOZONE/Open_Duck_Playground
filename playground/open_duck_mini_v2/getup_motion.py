"""Face-down get-up reference motion shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


GETUP_DURATION = 7.0
TUCK_REACHED = 1.40
PLANT_REACHED = 2.80
CROUCH_REACHED = 4.80
STAND_REACHED = 6.00
GETUP_PHASES = {
    "tuck": (0.0, TUCK_REACHED),
    "plant": (TUCK_REACHED, PLANT_REACHED),
    "rise": (PLANT_REACHED, CROUCH_REACHED),
    "stand": (CROUCH_REACHED, STAND_REACHED),
    "hold": (STAND_REACHED, GETUP_DURATION),
}

# MuJoCo quaternions are [w, x, y, z].  A perfectly horizontal 90 degree
# pitch starts above the contact surface and drops several centimetres before
# the controller can act.  This slightly shallower, still face-down attitude
# is the settled contact pose measured in MuJoCo (forward-z < -0.99).
FACE_DOWN_QUAT = np.array(
    [0.7480, 0.0, 0.6637, 0.0], dtype=np.float32
)
FACE_DOWN_QUAT /= np.linalg.norm(FACE_DOWN_QUAT)
UPRIGHT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
PRONE_ROOT_HEIGHT = 0.054
STANDING_ROOT_HEIGHT = 0.172

# Actuator order follows open_duck_mini_v2.xml.  Both hip-yaw joints steer in
# the same direction during the roll so the knees travel together.  Hip pitch
# and roll remain mirrored and both knees use matching flexion; this avoids the
# scissoring motion in which one leg extended while the other folded.  Long,
# quintic blends keep the coordinated turn from kicking or snapping.
TUCK_OFFSETS = np.array(
    [
        0.378, 0.147, 0.380, 0.132, -0.266,
        -0.200, -0.300, 0.250, -0.150,
        0.383, -0.135, -0.385, 0.121, -0.254,
    ],
    dtype=np.float32,
)
PLANT_OFFSETS = np.array(
    [
        0.448, 0.067, 0.480, 0.082, -0.366,
        -0.100, -0.150, 0.150, -0.080,
        0.453, -0.055, -0.485, 0.071, -0.354,
    ],
    dtype=np.float32,
)
CROUCH_OFFSETS = np.array(
    [
        0.148, 0.027, 0.330, 0.132, -0.216,
        0.100, -0.100, 0.0, 0.0,
        0.153, -0.015, -0.335, 0.121, -0.204,
    ],
    dtype=np.float32,
)

_KEYFRAME_TIMES = np.array(
    [0.0, TUCK_REACHED, PLANT_REACHED, CROUCH_REACHED, STAND_REACHED],
    dtype=np.float32,
)

_ROOT_POSITION_TIMES = np.array(
    [0.0, 0.70, 1.00, TUCK_REACHED, PLANT_REACHED, 3.70, 4.00, 4.50, CROUCH_REACHED, STAND_REACHED, GETUP_DURATION],
    dtype=np.float32,
)

_ROOT_POSITION_KEYFRAMES = np.array(
    [
        [0.000, 0.000, PRONE_ROOT_HEIGHT],
        [0.013, -0.028, 0.158],
        [0.018, -0.040, 0.176],
        [0.025, -0.055, 0.175],
        [0.030, -0.085, 0.170],
        [0.023, -0.067, 0.200],
        [0.021, -0.062, 0.207],
        [0.016, -0.050, 0.173],
        [0.012, -0.040, 0.157],
        [0.000, 0.000, STANDING_ROOT_HEIGHT],
        [0.000, 0.000, STANDING_ROOT_HEIGHT],
    ],
    dtype=np.float32,
)

_ROOT_QUATERNION_KEYFRAMES = np.array(
    [
        FACE_DOWN_QUAT,
        [0.9141, 0.0885, 0.3434, 0.1968],
        [0.8783, 0.0488, 0.2176, 0.4228],
        [0.9850, 0.0000, 0.0500, 0.1650],
        UPRIGHT_QUAT,
    ],
    dtype=np.float32,
)
_ROOT_QUATERNION_KEYFRAMES /= np.linalg.norm(
    _ROOT_QUATERNION_KEYFRAMES, axis=1, keepdims=True
)


def _smoothstep(value: Any, xp: Any) -> Any:
    """Quintic smootherstep with zero velocity and acceleration at each end."""

    value = xp.clip(value, 0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _lerp(start: Any, end: Any, value: Any, xp: Any) -> Any:
    return start + (end - start) * _smoothstep(value, xp)


def _quat_slerp(start: Any, end: Any, value: Any, xp: Any) -> Any:
    """Shortest-path quaternion interpolation for NumPy and JAX arrays."""

    value = _smoothstep(value, xp)[..., None]
    dot = xp.sum(start * end, axis=-1, keepdims=True)
    end = xp.where(dot < 0.0, -end, end)
    dot = xp.clip(xp.abs(dot), 0.0, 1.0)
    angle = xp.arccos(dot)
    sin_angle = xp.sin(angle)
    safe_sin = xp.where(sin_angle > 1.0e-6, sin_angle, 1.0)
    result = (
        xp.sin((1.0 - value) * angle) / safe_sin * start
        + xp.sin(value * angle) / safe_sin * end
    )
    linear = start + value * (end - start)
    result = xp.where(sin_angle > 1.0e-6, result, linear)
    return result / xp.maximum(xp.linalg.norm(result, axis=-1, keepdims=True), 1.0e-6)


def _piecewise_lerp(
    time_s: Any,
    values: Any,
    xp: Any,
    quaternion: bool = False,
    keyframe_times: Any = _KEYFRAME_TIMES,
) -> Any:
    """Interpolate ordered keyframes with a stationary join at every pose."""

    values = xp.asarray(values)
    t = xp.asarray(time_s, dtype=values.dtype)
    times = xp.asarray(keyframe_times, dtype=values.dtype)
    result = values[0]
    for index in range(len(keyframe_times) - 1):
        amount = (t - times[index]) / (times[index + 1] - times[index])
        if quaternion:
            candidate = _quat_slerp(
                values[index], values[index + 1], amount, xp
            )
        else:
            candidate = _lerp(
                values[index], values[index + 1], amount[..., None], xp
            )
        result = xp.where((t >= times[index])[..., None], candidate, result)
    return result


def trajectory_pose(time_s: Any, home_pose: Any, xp: Any = np) -> Any:
    """Return the desired 14-actuator pose at ``time_s``."""

    home = xp.asarray(home_pose)
    offsets = xp.asarray(
        np.stack(
            [
                np.zeros_like(TUCK_OFFSETS),
                TUCK_OFFSETS,
                PLANT_OFFSETS,
                CROUCH_OFFSETS,
                np.zeros_like(TUCK_OFFSETS),
            ]
        ),
        dtype=home.dtype,
    )
    return _piecewise_lerp(time_s, home + offsets, xp)


def root_trajectory(
    time_s: Any, start_xy: Any | None = None, xp: Any = np
) -> tuple[Any, Any]:
    """Return reference root position and orientation for imitation rewards."""

    t = xp.asarray(time_s)
    if start_xy is None:
        start_xy = xp.zeros(2, dtype=t.dtype)
    else:
        start_xy = xp.asarray(start_xy)

    pos = _piecewise_lerp(
        t,
        _ROOT_POSITION_KEYFRAMES,
        xp,
        keyframe_times=_ROOT_POSITION_TIMES,
    )
    quat = _piecewise_lerp(
        t, _ROOT_QUATERNION_KEYFRAMES, xp, quaternion=True
    )
    if hasattr(pos, "at"):
        pos = pos.at[..., :2].add(start_xy)
    else:
        pos = np.array(pos, copy=True)
        pos[..., :2] += np.asarray(start_xy)
    return pos, quat


def clip_pose(pose: Any, lower: Any, upper: Any, xp: Any = np) -> Any:
    return xp.clip(pose, xp.asarray(lower), xp.asarray(upper))


def phase_features(time_s: Any, xp: Any = np) -> Any:
    """Return ``[active, sin(phase), cos(phase)]`` for policy observations."""

    t = xp.asarray(time_s)
    normalized = xp.clip(t / GETUP_DURATION, 0.0, 1.0)
    active = (t < GETUP_DURATION).astype(xp.asarray(normalized).dtype)
    angle = normalized * 2.0 * xp.pi
    return xp.stack([active, xp.sin(angle), xp.cos(angle)], axis=-1)


@dataclass
class GetUpMotionController:
    """One-shot clock for previewing or controlling the get-up motion."""

    elapsed_s: float = GETUP_DURATION

    @property
    def active(self) -> bool:
        return self.elapsed_s < GETUP_DURATION

    def request_getup(self) -> bool:
        if self.active:
            return False
        self.elapsed_s = 0.0
        return True

    def reset(self) -> None:
        self.elapsed_s = GETUP_DURATION

    def advance(self, dt: float) -> None:
        if self.active:
            self.elapsed_s = min(GETUP_DURATION, self.elapsed_s + float(dt))

    def target(
        self,
        home_pose: np.ndarray,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> np.ndarray:
        target = np.asarray(
            trajectory_pose(self.elapsed_s, home_pose), dtype=np.float32
        )
        if lower is not None and upper is not None:
            target = np.clip(target, lower, upper)
        return target

    def root_target(self, start_xy: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = root_trajectory(self.elapsed_s, start_xy)
        return np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)

    def features(self) -> np.ndarray:
        return np.asarray(phase_features(self.elapsed_s), dtype=np.float32)
