"""Preview the get-up animation or run a learned ONNX recovery policy."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from playground.common.onnx_infer import OnnxInfer
from playground.open_duck_mini_v2.getup_motion import GetUpMotionController
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase


class GetUpInfer:
    def __init__(
        self,
        model_path: str,
        controller: str = "reference",
        onnx_model_path: str | None = None,
        use_reference_motion: bool = True,
    ) -> None:
        self.base = MJInferBase(model_path)
        self.model = self.base.model
        self.data = self.base.data
        self.default_actuator = self.base.default_actuator.copy()
        self.actuator_lowers = self.model.actuator_ctrlrange[:, 0].copy()
        self.actuator_uppers = self.model.actuator_ctrlrange[:, 1].copy()
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.action_scale = 1.0
        self.max_motor_velocity = 5.24
        self.sim_dt = 0.002
        self.decimation = 10
        self.controller_name = controller
        self.use_reference_motion = use_reference_motion
        self.motion = GetUpMotionController()
        self.start_xy = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(self.base.num_dofs, dtype=np.float32)
        self.last_last_action = np.zeros_like(self.last_action)
        self.last_last_last_action = np.zeros_like(self.last_action)
        self.imitation_phase = np.zeros(2, dtype=np.float32)
        self.policy = None
        if controller == "onnx":
            if not onnx_model_path:
                raise ValueError("--onnx_model_path is required with --controller onnx")
            self.policy = OnnxInfer(onnx_model_path, awd=True)
        self.request_getup()

    def _set_reference_state(self) -> None:
        root_pos, root_quat = self.motion.root_target(self.start_xy)
        base_qpos = self.base.get_floating_base_qpos(self.data.qpos).copy()
        base_qpos[:3] = root_pos
        base_qpos[3:7] = root_quat
        self.base.set_floating_base_qpos(base_qpos, self.data.qpos)
        target = self.motion.target(
            self.default_actuator, self.actuator_lowers, self.actuator_uppers
        )
        self.base.set_actuator_joints_qpos(target, self.data.qpos)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = target
        self.motor_targets = target.copy()
        self.prev_motor_targets = target.copy()
        mujoco.mj_forward(self.model, self.data)

    def request_getup(self) -> bool:
        accepted = self.motion.request_getup()
        if accepted:
            self.start_xy = self.data.qpos[:2].copy()
            self.last_action[:] = 0.0
            self.last_last_action[:] = 0.0
            self.last_last_last_action[:] = 0.0
            # Every replay begins from the exact face-down training reset.
            self._set_reference_state()
            print("Face-down get-up armed")
        return accepted

    def get_obs(self) -> np.ndarray:
        gyro = self.base.get_gyro(self.data)
        accelerometer = self.base.get_accelerometer(self.data)
        joint_angles = self.base.get_actuator_joints_qpos(self.data.qpos)
        joint_vel = self.base.get_actuator_joints_qvel(self.data.qvel)
        contacts = np.asarray(self.base.get_feet_contacts(self.data), dtype=np.float32)
        return np.concatenate(
            [
                gyro,
                accelerometer,
                np.zeros(7, dtype=np.float32),
                joint_angles - self.default_actuator,
                joint_vel * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                self.imitation_phase,
                (
                    self.motion.features()
                    if self.use_reference_motion
                    else np.zeros(3, dtype=np.float32)
                ),
            ]
        ).astype(np.float32)

    def _update_dynamic_target(self) -> None:
        reference = (
            self.motion.target(
                self.default_actuator,
                self.actuator_lowers,
                self.actuator_uppers,
            )
            if self.use_reference_motion
            else self.default_actuator
        )
        if self.controller_name == "onnx":
            action = np.asarray(self.policy.infer(self.get_obs()), dtype=np.float32)
            self.last_last_last_action = self.last_last_action.copy()
            self.last_last_action = self.last_action.copy()
            self.last_action = action.copy()
            target = reference + action * self.action_scale
        else:
            target = reference
            self.last_last_last_action = self.last_last_action.copy()
            self.last_last_action = self.last_action.copy()
            self.last_action = np.clip(
                (target - self.default_actuator) / self.action_scale, -1.0, 1.0
            )
        target = np.clip(target, self.actuator_lowers, self.actuator_uppers)
        self.motor_targets = np.clip(
            target,
            self.prev_motor_targets
            - self.max_motor_velocity * self.sim_dt * self.decimation,
            self.prev_motor_targets
            + self.max_motor_velocity * self.sim_dt * self.decimation,
        )
        self.prev_motor_targets = self.motor_targets.copy()
        self.data.ctrl[:] = self.motor_targets
        self.motion.advance(self.sim_dt * self.decimation)

    def key_callback(self, keycode: int) -> None:
        if keycode in (71, 32):  # g or space
            self.request_getup()

    def run(self) -> None:
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback,
        ) as viewer:
            counter = 0
            while viewer.is_running():
                step_start = time.time()
                if self.controller_name == "reference":
                    self._set_reference_state()
                    self.motion.advance(self.sim_dt)
                else:
                    mujoco.mj_step(self.model, self.data)
                    counter += 1
                    if counter % self.decimation == 0:
                        self._update_dynamic_target()
                viewer.sync()
                remaining = self.model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller", choices=("reference", "scripted", "onnx"), default="reference"
    )
    parser.add_argument("--onnx_model_path", default=None)
    parser.add_argument(
        "--goal-only",
        action="store_true",
        help="Use a goal-only policy with no animation target or phase signal.",
    )
    parser.add_argument(
        "--model_path",
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    args = parser.parse_args()
    GetUpInfer(
        model_path=args.model_path,
        controller=args.controller,
        onnx_model_path=args.onnx_model_path,
        use_reference_motion=not args.goal_only,
    ).run()


if __name__ == "__main__":
    main()
