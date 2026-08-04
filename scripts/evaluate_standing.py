"""Headlessly rank standing ONNX policies by survival, drift, and foot motion."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from playground.open_duck_mini_v2 import constants
from playground.open_duck_mini_v2.mujoco_infer import MjInfer


DEFAULT_MODEL = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
DEFAULT_REFERENCE = (
    "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
)


def evaluate_policy(
    onnx_path: Path,
    *,
    duration: float,
    initial_forward_velocity: float,
) -> dict[str, float | bool]:
    infer = MjInfer(
        model_path=DEFAULT_MODEL,
        reference_data=DEFAULT_REFERENCE,
        onnx_model_path=str(onnx_path),
        standing=True,
    )
    infer.data.qvel[infer._floating_base_qvel_addr] = initial_forward_velocity
    mujoco.mj_forward(infer.model, infer.data)

    root_start = infer.data.qpos[infer._floating_base_qpos_addr :][:2].copy()
    previous_root = root_start.copy()
    foot_ids = np.asarray(
        [mujoco.mj_name2id(infer.model, mujoco.mjtObj.mjOBJ_SITE, name)
         for name in constants.FEET_SITES]
    )
    initial_foot_z = infer.data.site_xpos[foot_ids, 2].copy()
    previous_foot_xy = infer.data.site_xpos[foot_ids, :2].copy()
    root_path = 0.0
    foot_path = 0.0
    max_foot_lift = 0.0
    elapsed = 0.0

    total_steps = round(duration / infer.sim_dt)
    for step in range(total_steps):
        mujoco.mj_step(infer.model, infer.data)
        elapsed += infer.sim_dt
        if (step + 1) % infer.decimation != 0:
            continue

        obs = infer.get_obs(infer.data, infer.commands).astype(np.float32)
        action = infer._active_locomotion_policy().infer(obs)
        infer.last_last_last_action = infer.last_last_action.copy()
        infer.last_last_action = infer.last_action.copy()
        infer.last_action = action.copy()
        target = infer.default_actuator + action * infer.action_scale
        control_dt = infer.sim_dt * infer.decimation
        target = np.clip(
            target,
            infer.prev_motor_targets - infer.max_motor_velocity * control_dt,
            infer.prev_motor_targets + infer.max_motor_velocity * control_dt,
        )
        infer.motor_targets = np.clip(
            target,
            infer.model.actuator_ctrlrange[:, 0],
            infer.model.actuator_ctrlrange[:, 1],
        )
        infer.prev_motor_targets = infer.motor_targets.copy()
        infer.data.ctrl = infer.motor_targets

        root_xy = infer.data.qpos[infer._floating_base_qpos_addr :][:2].copy()
        feet_xy = infer.data.site_xpos[foot_ids, :2].copy()
        root_path += float(np.linalg.norm(root_xy - previous_root))
        foot_path += float(np.linalg.norm(feet_xy - previous_foot_xy, axis=1).sum())
        max_foot_lift = max(
            max_foot_lift,
            float(np.max(infer.data.site_xpos[foot_ids, 2] - initial_foot_z)),
        )
        previous_root = root_xy
        previous_foot_xy = feet_xy

        root_height = infer.data.qpos[infer._floating_base_qpos_addr + 2]
        if root_height < 0.15 or not np.isfinite(infer.data.qpos).all():
            break

    displacement = float(np.linalg.norm(previous_root - root_start))
    survived = elapsed >= duration - infer.sim_dt
    return {
        "survived": survived,
        "elapsed": elapsed,
        "displacement": displacement,
        "root_path": root_path,
        "foot_path": foot_path,
        "max_foot_lift": max_foot_lift,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--shove", type=float, default=0.25)
    args = parser.parse_args()

    ranked = []
    for path in args.models:
        nominal = evaluate_policy(
            path, duration=args.duration, initial_forward_velocity=0.0
        )
        shoved = evaluate_policy(
            path, duration=args.duration, initial_forward_velocity=args.shove
        )
        score = (
            1000.0 * (float(nominal["survived"]) + float(shoved["survived"]))
            - 100.0
            * (float(nominal["displacement"]) + float(shoved["displacement"]))
            - 10.0 * (float(nominal["foot_path"]) + float(shoved["foot_path"]))
            - 100.0
            * (float(nominal["max_foot_lift"]) + float(shoved["max_foot_lift"]))
        )
        ranked.append((score, path, nominal, shoved))

    print(
        "score\tmodel\tnominal_xy\tshoved_xy\tnominal_foot_path\t"
        "shoved_foot_path\tmax_foot_lift\tsurvived"
    )
    for score, path, nominal, shoved in sorted(ranked, reverse=True):
        print(
            f"{score:.3f}\t{path.name}\t"
            f"{nominal['displacement']:.4f}\t{shoved['displacement']:.4f}\t"
            f"{nominal['foot_path']:.4f}\t{shoved['foot_path']:.4f}\t"
            f"{max(nominal['max_foot_lift'], shoved['max_foot_lift']):.4f}\t"
            f"{nominal['survived'] and shoved['survived']}"
        )


if __name__ == "__main__":
    main()
