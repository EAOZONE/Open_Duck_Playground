"""Interactive MuJoCo inference for scripted and learned Open Duck hops."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from playground.common.onnx_infer import OnnxInfer
from playground.open_duck_mini_v2.jump_motion import JumpMotionController
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase


class JumpInfer:
    """Run a one-shot jump controller in the MuJoCo viewer."""

    def __init__(
        self,
        model_path: str,
        controller: str = "scripted",
        onnx_model_path: str | None = None,
    ) -> None:
        self.base = MJInferBase(model_path)
        self.model = self.base.model
        self.data = self.base.data
        self.default_actuator = self.base.default_actuator.copy()
        self.actuator_lowers = self.model.actuator_ctrlrange[:, 0].copy()
        self.actuator_uppers = self.model.actuator_ctrlrange[:, 1].copy()
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.action_scale = 0.35
        self.max_motor_velocity = 5.24
        self.sim_dt = 0.002
        self.decimation = 10
        self.controller_name = controller
        self.motion = JumpMotionController()
        self.last_action = np.zeros(self.base.num_dofs, dtype=np.float32)
        self.last_last_action = np.zeros_like(self.last_action)
        self.last_last_last_action = np.zeros_like(self.last_action)
        self.imitation_phase = np.zeros(2, dtype=np.float32)
        self.policy = None
        if controller == "onnx":
            if not onnx_model_path:
                raise ValueError("--onnx_model_path is required with --controller onnx")
            self.policy = OnnxInfer(onnx_model_path, awd=True)

    def request_jump(self) -> bool:
        """Start one jump if the previous cycle has finished."""

        accepted = self.motion.request_jump()
        if accepted:
            print("Jump armed")
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
                self.motion.features(),
            ]
        ).astype(np.float32)

    def _target_from_controller(self) -> np.ndarray:
        if self.controller_name == "scripted":
            return self.motion.target(
                self.default_actuator, self.actuator_lowers, self.actuator_uppers
            )

        obs = self.get_obs()
        action = np.asarray(self.policy.infer(obs), dtype=np.float32)
        action[5:9] = 0.0
        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()
        reference = self.motion.target(
            self.default_actuator, self.actuator_lowers, self.actuator_uppers
        )
        return reference + action * self.action_scale

    def _update_target(self) -> None:
        target = self._target_from_controller()
        self.motor_targets = np.clip(
            target,
            self.prev_motor_targets - self.max_motor_velocity * self.sim_dt * self.decimation,
            self.prev_motor_targets + self.max_motor_velocity * self.sim_dt * self.decimation,
        )
        self.prev_motor_targets = self.motor_targets.copy()
        self.data.ctrl[:] = self.motor_targets

        if self.controller_name == "scripted":
            self.last_last_last_action = self.last_last_action.copy()
            self.last_last_action = self.last_action.copy()
            self.last_action = np.clip(
                (self.motor_targets - self.default_actuator) / self.action_scale,
                -1.0,
                1.0,
            )
        self.motion.advance(self.sim_dt * self.decimation)

    def key_callback(self, keycode: int) -> None:
        if keycode in (74, 32):  # j or space
            self.request_jump()

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
                mujoco.mj_step(self.model, self.data)
                counter += 1
                if counter % self.decimation == 0:
                    self._update_target()
                viewer.sync()
                remaining = self.model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller", choices=("scripted", "onnx"), default="scripted"
    )
    parser.add_argument("--onnx_model_path", default=None)
    parser.add_argument(
        "--model_path",
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    args = parser.parse_args()
    JumpInfer(
        model_path=args.model_path,
        controller=args.controller,
        onnx_model_path=args.onnx_model_path,
    ).run()


if __name__ == "__main__":
    main()
