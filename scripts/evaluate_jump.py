"""Evaluate scripted or ONNX jump rollouts in the MuJoCo model."""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from playground.open_duck_mini_v2.jump_infer import JumpInfer


def evaluate_once(infer: JumpInfer, duration_s: float = 3.0) -> dict[str, float | bool]:
    infer.motion.reset()
    infer.data.qpos[:] = infer.model.keyframe("home").qpos
    infer.data.qvel[:] = 0.0
    infer.data.ctrl[:] = infer.default_actuator
    mujoco.mj_forward(infer.model, infer.data)
    infer.prev_motor_targets = infer.default_actuator.copy()
    infer.motor_targets = infer.default_actuator.copy()
    infer.last_action[:] = 0.0
    infer.last_last_action[:] = 0.0
    infer.last_last_last_action[:] = 0.0
    infer.request_jump()

    trunk_site = infer.model.site("trunk").id
    trunk_body = infer.model.body("trunk_assembly").id
    start_height = float(infer.data.site_xpos[trunk_site, 2])
    start_xy = infer.data.site_xpos[trunk_site, :2].copy()
    max_height = start_height
    max_air_steps = 0
    air_steps = 0
    landing_step = None
    stable_steps = 0
    min_upright = 1.0

    total_steps = int(duration_s / infer.sim_dt)
    for step in range(total_steps):
        mujoco.mj_step(infer.model, infer.data)
        trunk_up = float(infer.data.xmat[trunk_body].reshape(3, 3)[2, 2])
        min_upright = min(min_upright, trunk_up)
        max_height = max(max_height, float(infer.data.site_xpos[trunk_site, 2]))
        contacts = infer.base.get_feet_contacts(infer.data)
        if not any(contacts):
            air_steps += 1
        else:
            if air_steps >= 3 and landing_step is None:
                landing_step = step
            max_air_steps = max(max_air_steps, air_steps)
            air_steps = 0

        if landing_step is not None and all(contacts) and trunk_up > 0.9:
            stable_steps += 1
        if step % infer.decimation == infer.decimation - 1:
            infer._update_target()

    end_xy = infer.data.site_xpos[trunk_site, :2]
    horizontal_displacement = float(np.linalg.norm(end_xy - start_xy))
    return {
        "height_delta_m": max_height - start_height,
        "max_air_steps": float(max_air_steps),
        "landed": landing_step is not None,
        "min_upright": min_upright,
        "horizontal_displacement_m": horizontal_displacement,
        "stable_hold_s": stable_steps * infer.sim_dt,
        "success": bool(
            max_height - start_height >= 0.04
            and max_air_steps >= 3
            and landing_step is not None
            and min_upright > 0.9
            and horizontal_displacement < 0.05
            and stable_steps * infer.sim_dt >= 0.4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("scripted", "onnx"), default="scripted")
    parser.add_argument("--onnx_model_path", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--model_path",
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    args = parser.parse_args()
    infer = JumpInfer(
        args.model_path,
        controller=args.controller,
        onnx_model_path=args.onnx_model_path,
    )
    results = [evaluate_once(infer) for _ in range(args.episodes)]
    print(json.dumps({"episodes": results}, indent=2))


if __name__ == "__main__":
    main()
