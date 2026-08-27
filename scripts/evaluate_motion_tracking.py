"""Headlessly score a unified ONNX policy over mixed motion transitions."""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jp
import numpy as np

from playground.common.onnx_infer import OnnxInfer
from playground.open_duck_mini_v2.motion_tracking import MotionTracking, default_config


SCENARIO_TASK = {
    "nominal": "flat_terrain",
    "backlash": "flat_terrain_backlash",
    "rough_terrain": "rough_terrain",
    "latency": "flat_terrain",
    "push": "flat_terrain",
    "parameter_extreme": "rough_terrain_backlash",
}


def evaluate(
    onnx_path: str,
    episodes: int,
    steps: int,
    scenario: str,
    motion_names: list[str] | None = None,
) -> dict[str, object]:
    config = default_config()
    config.noise_config.level = 1.0 if scenario in ("latency", "parameter_extreme") else 0.0
    config.push_config.enable = scenario in ("push", "parameter_extreme")
    if scenario not in ("latency", "parameter_extreme"):
        config.noise_config.action_max_delay = 1
        config.noise_config.imu_max_delay = 1
    task = SCENARIO_TASK[scenario]
    env = MotionTracking(task=task, config=config)
    if scenario == "parameter_extreme":
        model = env.mjx_model
        gain = model.actuator_gainprm.at[:, 0].set(model.actuator_gainprm[:, 0] * 1.1)
        bias = model.actuator_biasprm.at[:, 1].set(model.actuator_biasprm[:, 1] * 1.1)
        env._mjx_model = model.tree_replace(
            {
                "body_mass": model.body_mass * 1.1,
                "geom_friction": model.geom_friction.at[0, 0].set(0.5),
                "body_ipos": model.body_ipos.at[env._torso_body_id].add(jp.asarray([0.05, 0.05, 0.05])),
                "actuator_gainprm": gain,
                "actuator_biasprm": bias,
            }
        )
    policy = OnnxInfer(onnx_path, awd=True)
    step_fn = jax.jit(env.step)
    falls = 0
    transitions = 0
    joint_squared_error = 0.0
    orientation_errors = []
    samples = 0
    inference_times = []
    per_motion: dict[str, dict[str, float]] = {}
    for episode in range(episodes):
        state = env.reset(jax.random.PRNGKey(episode))
        episode_motion = None
        if motion_names:
            episode_motion = motion_names[episode % len(motion_names)]
            motion_index = env.motion_names.index(episode_motion)
            state.info["motion_index"] = jp.asarray(motion_index)
            state.info["motion_time"] = jp.asarray(0.0)
            state.info["motion_age"] = jp.asarray(0.0)
            state.info["blend_time"] = jp.asarray(0.0)
            stand_index = env.motion_names.index("stand")
            state.info["blend_from"] = env._raw_reference(
                jp.asarray(stand_index), jp.asarray(0.0)
            )
            command, phase = env._command_and_phase(state.info)
            state.info["command"] = command
            state.info["imitation_phase"] = phase
            state = state.replace(obs=env._get_obs(state.data, state.info, env._contacts(state.data)))
            per_motion.setdefault(
                episode_motion,
                {"episodes": 0.0, "falls": 0.0, "joint_squared_error": 0.0, "samples": 0.0},
            )["episodes"] += 1
        previous_motion = int(state.info["motion_index"])
        episode_steps = steps
        if episode_motion and not bool(env._motion["loops"][previous_motion]):
            frames = int(env._motion["lengths"][previous_motion])
            episode_steps = min(
                steps, int(np.ceil((frames - 1) / env._motion_fps / env.dt)) + 20
            )
        for _ in range(episode_steps):
            observation = np.asarray(state.obs["state"], dtype=np.float32)
            started = time.perf_counter()
            action = policy.infer(observation)
            inference_times.append(time.perf_counter() - started)
            state = step_fn(state, jax.numpy.asarray(action))
            current_motion = int(state.info["motion_index"])
            transitions += current_motion != previous_motion
            previous_motion = current_motion
            reference = np.asarray(env._reference_vector(state.info))
            joints = np.asarray(env.get_actuator_joints_qpos(state.data.qpos))
            joint_squared_error += float(np.mean(np.square(joints - reference[:14])))
            if episode_motion:
                per_motion[episode_motion]["joint_squared_error"] += float(
                    np.mean(np.square(joints - reference[:14]))
                )
                per_motion[episode_motion]["samples"] += 1
            # ``upvector`` is already the IMU frame's positive Z axis in world
            # coordinates, matching the motion bundle's body-up reference.
            actual_up = np.asarray(env.get_gravity(state.data))
            cosine = np.clip(np.dot(actual_up, reference[15:18]), -1.0, 1.0)
            orientation_errors.append(float(np.degrees(np.arccos(cosine))))
            samples += 1
            if bool(state.done):
                falls += 1
                if episode_motion:
                    per_motion[episode_motion]["falls"] += 1
                break
    motion_summary = {
        name: {
            "completion_rate": 1.0 - values["falls"] / max(values["episodes"], 1.0),
            "joint_rmse_rad": float(
                np.sqrt(values["joint_squared_error"] / max(values["samples"], 1.0))
            ),
        }
        for name, values in per_motion.items()
    }
    return {
        "scenario": scenario,
        "episodes": episodes,
        "falls": falls,
        "completion_rate": 1.0 - falls / episodes,
        "transitions": transitions,
        "joint_rmse_rad": float(np.sqrt(joint_squared_error / max(samples, 1))),
        "mean_orientation_error_deg": float(np.mean(orientation_errors)),
        "p99_inference_ms": float(np.percentile(inference_times, 99) * 1000.0),
        "per_motion": motion_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_TASK), default="nominal")
    parser.add_argument("--motion", action="append", help="Force and score a named clip; repeatable.")
    parser.add_argument(
        "--split", choices=("train", "holdout", "all"), default=None,
        help="Evaluate every clip in a bundle split (or all clips).",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    motion_names = args.motion
    if args.split:
        probe = MotionTracking(config=default_config())
        split_values = np.asarray(probe._motion["splits"])
        motion_names = [
            name for index, name in enumerate(probe.motion_names)
            if args.split == "all" or int(split_values[index]) == (0 if args.split == "train" else 1)
        ]
    result = evaluate(args.onnx, args.episodes, args.steps, args.scenario, motion_names)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
