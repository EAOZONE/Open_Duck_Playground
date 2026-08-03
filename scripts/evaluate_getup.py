"""Headlessly evaluate a goal-only get-up ONNX policy in MuJoCo."""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from playground.open_duck_mini_v2.getup import (
    LEFT_FOOT_STUCK_QPOS,
    RIGHT_FOOT_STUCK_QPOS,
    STANDING_FOOT_NORMAL_MIN,
)
from playground.open_duck_mini_v2.getup_infer import GetUpInfer
from playground.open_duck_mini_v2.getup_motion import GETUP_DURATION


def _multiply_axis_angle(
    quaternion: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    delta = np.empty(4)
    result = np.empty(4)
    mujoco.mju_axisAngle2Quat(delta, axis, angle)
    mujoco.mju_mulQuat(result, delta, quaternion)
    return result


def reset_episode(
    infer: GetUpInfer,
    rng: np.random.Generator,
    reset: str,
    randomized: bool,
) -> None:
    infer.motion.elapsed_s = GETUP_DURATION
    infer.request_getup()
    if reset == "right-stuck":
        infer.data.qpos[:] = RIGHT_FOOT_STUCK_QPOS
    elif reset == "left-stuck":
        infer.data.qpos[:] = LEFT_FOOT_STUCK_QPOS
    elif reset == "near-standing":
        infer.data.qpos[:] = infer.model.keyframe("home").qpos

    if randomized:
        infer.data.qpos[:2] += rng.uniform(-0.015, 0.015, size=2)
        quaternion = infer.data.qpos[3:7].copy()
        quaternion = _multiply_axis_angle(
            quaternion, np.array([0.0, 0.0, 1.0]), rng.uniform(-0.12, 0.12)
        )
        quaternion = _multiply_axis_angle(
            quaternion, np.array([0.0, 1.0, 0.0]), rng.uniform(-0.08, 0.08)
        )
        quaternion = _multiply_axis_angle(
            quaternion, np.array([1.0, 0.0, 0.0]), rng.uniform(-0.06, 0.06)
        )
        infer.data.qpos[3:7] = quaternion
        qpos_noise = np.zeros(infer.model.nu)
        for index, name in enumerate(infer.base.actuator_names):
            if "_hip" in name:
                qpos_noise[index] = 0.025
            elif "_knee" in name:
                qpos_noise[index] = 0.04
            elif "_ankle" in name:
                qpos_noise[index] = 0.05
        joint_qpos = infer.base.get_actuator_joints_qpos(infer.data.qpos)
        infer.base.set_actuator_joints_qpos(
            joint_qpos + rng.uniform(-1.0, 1.0, infer.model.nu) * qpos_noise,
            infer.data.qpos,
        )
        infer.data.qvel[:6] = rng.uniform(-0.03, 0.03, size=6)

    infer.data.ctrl[:] = infer.base.get_actuator_joints_qpos(infer.data.qpos)
    infer.motor_targets = infer.data.ctrl.copy()
    infer.prev_motor_targets = infer.data.ctrl.copy()
    mujoco.mj_forward(infer.model, infer.data)


def _body_on_floor(infer: GetUpInfer) -> bool:
    floor = infer.model.geom("floor").id
    body_geoms = {
        infer.model.geom("trunk_collision").id,
        infer.model.geom("head_collision").id,
    }
    for contact in infer.data.contact:
        if (contact.geom1 == floor and contact.geom2 in body_geoms) or (
            contact.geom2 == floor and contact.geom1 in body_geoms
        ):
            return True
    return False


def evaluate_once(
    infer: GetUpInfer,
    rng: np.random.Generator,
    *,
    reset: str = "face-down",
    randomized: bool = True,
    duration_s: float = 8.0,
    required_hold_s: float = 2.0,
) -> dict[str, float | bool | list[float]]:
    reset_episode(infer, rng, reset, randomized)
    foot_ids = [infer.model.site(name).id for name in ("left_foot", "right_foot")]
    trunk_body = infer.model.body("trunk_assembly").id
    linvel_sensor = infer.model.sensor("local_linvel")
    linvel_address = int(infer.model.sensor_adr[linvel_sensor.id])
    linvel_dimension = int(infer.model.sensor_dim[linvel_sensor.id])
    required_steps = int(round(required_hold_s / infer.sim_dt))
    standing_streak = 0
    max_standing_streak = 0
    current_streak_start_s: float | None = None
    best_streak_start_s: float | None = None
    body_contact_during_hold = False

    for step in range(int(round(duration_s / infer.sim_dt))):
        mujoco.mj_step(infer.model, infer.data)
        if step % infer.decimation == infer.decimation - 1:
            infer._update_dynamic_target()

        foot_normal_z = np.array(
            [infer.data.site_xmat[index].reshape(3, 3)[2, 2] for index in foot_ids]
        )
        contacts = np.asarray(infer.base.get_feet_contacts(infer.data), dtype=bool)
        upright = float(infer.data.xmat[trunk_body].reshape(3, 3)[2, 2])
        local_speed = float(
            np.linalg.norm(
                infer.data.sensordata[
                    linvel_address : linvel_address + linvel_dimension
                ]
            )
        )
        body_contact = _body_on_floor(infer)
        valid_standing = bool(
            upright > 0.95
            and 0.16 <= infer.data.qpos[2] <= 0.20
            and contacts.all()
            and (foot_normal_z > STANDING_FOOT_NORMAL_MIN).all()
            and local_speed < 0.15
            and not body_contact
        )
        if valid_standing:
            if standing_streak == 0:
                current_streak_start_s = step * infer.sim_dt
            standing_streak += 1
            body_contact_during_hold |= body_contact
        else:
            standing_streak = 0
            current_streak_start_s = None
        if standing_streak > max_standing_streak:
            max_standing_streak = standing_streak
            best_streak_start_s = current_streak_start_s

    final_normal_z = np.array(
        [infer.data.site_xmat[index].reshape(3, 3)[2, 2] for index in foot_ids]
    )
    final_tilt_deg = np.rad2deg(np.arccos(np.clip(final_normal_z, -1.0, 1.0)))
    hold_s = max_standing_streak * infer.sim_dt
    success = bool(
        max_standing_streak >= required_steps
        and best_streak_start_s is not None
        and best_streak_start_s <= duration_s - required_hold_s
        and not body_contact_during_hold
    )
    return {
        "success": success,
        "first_standing_s": (
            float(best_streak_start_s) if best_streak_start_s is not None else -1.0
        ),
        "stable_hold_s": float(hold_s),
        "root_height_m": float(infer.data.qpos[2]),
        "final_foot_tilt_deg": final_tilt_deg.tolist(),
        "worst_final_foot_tilt_deg": float(np.max(final_tilt_deg)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model-path", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reset",
        choices=("face-down", "right-stuck", "left-stuck", "near-standing"),
        default="face-down",
    )
    parser.add_argument("--exact", action="store_true")
    parser.add_argument(
        "--model-path",
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    args = parser.parse_args()
    infer = GetUpInfer(
        args.model_path,
        controller="onnx",
        onnx_model_path=args.onnx_model_path,
        use_reference_motion=False,
    )
    rng = np.random.default_rng(args.seed)
    episodes = [
        evaluate_once(
            infer,
            rng,
            reset=args.reset,
            randomized=not args.exact,
        )
        for _ in range(args.episodes)
    ]
    print(
        json.dumps(
            {
                "summary": {
                    "episodes": len(episodes),
                    "success_rate": float(
                        np.mean([episode["success"] for episode in episodes])
                    ),
                    "mean_worst_final_foot_tilt_deg": float(
                        np.mean(
                            [
                                episode["worst_final_foot_tilt_deg"]
                                for episode in episodes
                            ]
                        )
                    ),
                    "max_worst_final_foot_tilt_deg": float(
                        np.max(
                            [
                                episode["worst_final_foot_tilt_deg"]
                                for episode in episodes
                            ]
                        )
                    ),
                },
                "episodes": episodes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
