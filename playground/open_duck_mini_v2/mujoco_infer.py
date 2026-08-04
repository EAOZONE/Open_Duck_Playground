import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase
from playground.open_duck_mini_v2.jump_motion import JumpMotionController
from playground.open_duck_mini_v2 import playful_walk

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    KEY_JUMP = 74
    KEY_PLAYFUL_WALK = 80
    KEY_STAND = 83
    KEY_NORMAL_WALK = 87
    COMMANDS_RANGE_X = [-0.20, 0.20]
    COMMANDS_RANGE_Y = [-0.2, 0.2]
    COMMANDS_RANGE_THETA = [-1.0, 1.0]
    # The useful walking gait is near the top of the trained forward-command
    # range. The final clip below keeps an overdriven stick in-distribution,
    # while making about 80% forward stick reach the trained walking command.
    XBOX_FORWARD_GAIN = 1.25
    DEFAULT_LATCHED_FORWARD_SPEED = 0.20
    DEFAULT_HEAD_BOBBLE_AMPLITUDE = 0.12
    DEFAULT_ANTENNA_WIGGLE_AMPLITUDE = 0.10
    DEFAULT_STEP_BOUNCE_AMPLITUDE = 0.018
    HOP_SETTLE_DURATION = 0.4
    HOP_RECOVERY_DURATION = 0.3
    XBOX_HOP_BUTTON = 1  # B on the Linux SDL mapping used by this project.

    def __init__(
        self,
        model_path: str,
        reference_data: str,
        onnx_model_path: str,
        standing: bool,
        stand_onnx_model_path: str | None = None,
        policy_only: bool = False,
        xbox_controller: bool = False,
        fixed_command: tuple[float, float, float] | None = None,
        latched_walk: bool = False,
        latched_forward_speed: float = DEFAULT_LATCHED_FORWARD_SPEED,
        expressive_walk: bool = False,
        playful_policy: bool = False,
        head_bobble_amplitude: float = DEFAULT_HEAD_BOBBLE_AMPLITUDE,
        antenna_wiggle_amplitude: float = DEFAULT_ANTENNA_WIGGLE_AMPLITUDE,
        step_bounce_amplitude: float = DEFAULT_STEP_BOUNCE_AMPLITUDE,
    ):
        super().__init__(model_path)

        self.standing = standing
        self.policy_only = policy_only
        self.use_xbox_controller = xbox_controller
        self.fixed_command = fixed_command
        self.latched_walk = latched_walk
        self.latched_forward_speed = float(latched_forward_speed)
        self.expressive_walk = expressive_walk
        self.playful_policy = playful_policy
        self.playful_mode = bool(playful_policy)
        self.keyboard_gait_mode = "standing" if standing else "normal"
        self.head_bobble_amplitude = float(head_bobble_amplitude)
        self.antenna_wiggle_amplitude = float(antenna_wiggle_amplitude)
        self.step_bounce_amplitude = float(step_bounce_amplitude)
        self._latched_forward_command = 0.0
        self.head_control_mode = self.standing
        self._pygame = None
        self._xbox = None
        self._last_controller_command = None
        self._last_controller_report = 0.0
        self._hop_button_was_pressed = False

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.hop = JumpMotionController()
        self.hop_requested = False
        self.motion_mode = "walking"
        self.settle_elapsed = 0.0

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing and not self.policy_only:
            self.PRM = PolyReferenceMotion(reference_data)

        self.walk_policy = OnnxInfer(onnx_model_path, awd=True)
        self.stand_policy = (
            OnnxInfer(stand_onnx_model_path, awd=True)
            if stand_onnx_model_path is not None
            else (self.walk_policy if standing else None)
        )
        # Retain the old attribute for small external scripts that inspect it.
        self.policy = self.walk_policy

        self.COMMANDS_RANGE_X = [-0.20, 0.20]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.gait_cycle = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        if not self.use_xbox_controller:
            stand_status = "ready" if self.stand_policy is not None else "not loaded"
            print(
                "Keyboard modes: W normal walk, P playful walk, "
                f"S stand ({stand_status}), J jump"
            )
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def _apply_expressive_style(self) -> None:
        """Layer a bounded, excited character motion over the learned gait."""
        if not self.expressive_walk or self.standing or self.policy_only:
            return

        # Fade the flourish out with the movement command so the robot does
        # not keep bobbling after it stops.  Full forward walk gets the full
        # effect, while low-speed corrections retain a quieter version.
        command_scale = max(
            abs(float(self.commands[0])) / self.COMMANDS_RANGE_X[1],
            abs(float(self.commands[1])) / self.COMMANDS_RANGE_Y[1],
            abs(float(self.commands[2])) / self.COMMANDS_RANGE_THETA[1],
        )
        activity = float(np.clip(command_scale, 0.0, 1.0))
        if activity <= 0.05:
            return
        activity = float(np.clip((activity - 0.05) / 0.45, 0.0, 1.0))
        activity = activity * activity * (3.0 - 2.0 * activity)

        # The gait phase is [cos(theta), sin(theta)].  Its second harmonic
        # produces one perk/bounce per footfall rather than only one per full
        # left-right gait cycle.
        phase_cos = float(self.imitation_phase[0])
        phase_sin = float(self.imitation_phase[1])
        phase_sin_2 = 2.0 * phase_sin * phase_cos
        phase_cos_2 = phase_cos * phase_cos - phase_sin * phase_sin

        nod = self.head_bobble_amplitude * activity * (
            0.35 * phase_sin - 0.65 * phase_cos_2
        )
        self.motor_targets[5] += 0.50 * nod  # neck pitch
        self.motor_targets[6] += nod  # head pitch

        # The antennae are rigid meshes, not independently actuated joints.
        # A quick head roll plus a gentler yaw makes the pair visibly wiggle
        # without adding new dynamics or destabilizing the torso.
        wiggle = self.antenna_wiggle_amplitude * activity
        self.motor_targets[7] += 0.35 * wiggle * phase_sin  # head yaw
        self.motor_targets[8] += wiggle * (
            0.75 * phase_sin_2 + 0.25 * phase_cos
        )  # head roll

        # A small symmetric knee extension at each footfall makes the walk
        # springy without altering the policy's left/right foot timing.
        rise = 0.5 * (1.0 - phase_cos_2)
        bounce = self.step_bounce_amplitude * activity * rise
        self.motor_targets[3] -= bounce
        self.motor_targets[12] -= bounce
        self.motor_targets[4] += 0.35 * bounce
        self.motor_targets[13] += 0.35 * bounce

    def _update_playful_policy_cue(self, moving: bool) -> None:
        """Set the style cue consumed by a playful-walk policy."""
        if not self.playful_policy:
            return
        cue = (
            playful_walk.skip_cue_for_cycle(self.gait_cycle, np)
            if moving and self.playful_mode
            else 0.0
        )
        self.commands[playful_walk.SKIP_COMMAND_INDEX] = float(cue)

    def _set_keyboard_gait_mode(self, mode: str) -> bool:
        """Select a persistent forward, playful-forward, or standing mode."""
        if mode not in ("normal", "playful", "standing"):
            raise ValueError(f"Unknown keyboard gait mode: {mode}")
        if mode == "standing" and self.stand_policy is None:
            print(
                "Standing policy is not loaded; pass --stand-onnx-model-path "
                "to enable S"
            )
            return False

        moving = mode != "standing"
        speed = self.DEFAULT_LATCHED_FORWARD_SPEED if moving else 0.0
        changed = mode != self.keyboard_gait_mode
        self.keyboard_gait_mode = mode
        self.playful_mode = mode == "playful"
        self.fixed_command = (speed, 0.0, 0.0)
        self._latched_forward_command = speed
        self.commands[:3] = list(self.fixed_command)
        self.commands[3:] = [0.0, 0.0, 0.0, 0.0]
        if changed:
            self._reset_policy_history()
            self.imitation_i = 0
            self.imitation_phase = np.zeros(2)
        self._update_playful_policy_cue(moving)

        suffix = "" if mode != "playful" or self.playful_policy else " (cue disabled)"
        print(f"Keyboard mode: {mode}{suffix}")
        return True

    def _active_locomotion_policy(self):
        """Return the learned controller selected by the current keyboard mode."""
        if self.keyboard_gait_mode == "standing":
            if self.stand_policy is None:
                raise RuntimeError("Standing mode selected without a standing policy")
            return self.stand_policy
        return self.walk_policy

    def request_hop(self) -> bool:
        """Queue one hop when the walking controller is available."""
        if self.motion_mode != "walking" or self.hop_requested:
            return False
        self.hop_requested = True
        return True

    def _zero_movement_command(self) -> None:
        """Stop translation and yaw while preserving the command layout."""
        self.commands[:3] = [0.0, 0.0, 0.0]

    def _reset_policy_history(self) -> None:
        """Discard stale walking actions before returning from a hop."""
        self.last_action[:] = 0.0
        self.last_last_action[:] = 0.0
        self.last_last_last_action[:] = 0.0

    def _prepare_motion_step(self, dt: float) -> None:
        """Advance non-pose state transitions before computing a target."""
        if self.motion_mode == "walking" and self.hop_requested:
            self.hop_requested = False
            self.motion_mode = "settling"
            self.settle_elapsed = 0.0
            self._latched_forward_command = 0.0
            print("Hop requested: stopping before takeoff")

        if self.motion_mode != "walking":
            self._zero_movement_command()

        if self.motion_mode == "settling":
            self.settle_elapsed += float(dt)
            if self.settle_elapsed >= self.HOP_SETTLE_DURATION:
                # A stale active clock should not normally be possible, but
                # resetting it here makes repeated walk/hop cycles robust.
                if not self.hop.request_jump():
                    self.hop.reset()
                    self.hop.request_jump()
                self.motion_mode = "hopping"
                self.settle_elapsed = 0.0
                print("Hop started")
        elif self.motion_mode == "recovering":
            self.settle_elapsed += float(dt)
            if self.settle_elapsed >= self.HOP_RECOVERY_DURATION:
                self._reset_policy_history()
                self.prev_motor_targets = self.default_actuator.copy()
                self.motion_mode = "walking"
                self.settle_elapsed = 0.0
                print("Hop complete: walking controller restored")

    def _target_for_motion_mode(
        self, walking_target: np.ndarray | None, dt: float
    ) -> np.ndarray:
        """Select walking, hop, or recovery targets for this control tick."""
        if self.motion_mode == "hopping":
            target = self.hop.target(
                self.default_actuator,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
            self.hop.advance(dt)
            if not self.hop.active:
                self.motion_mode = "recovering"
                self.settle_elapsed = 0.0
                print("Hop landed: recovering")
            return target

        if self.motion_mode == "recovering":
            return self.default_actuator.copy()

        if walking_target is None:
            raise RuntimeError(
                f"Walking target is required while motion mode is {self.motion_mode!r}"
            )
        return walking_target

    def _update_hop_button(self, buttons: list[int]) -> None:
        """Queue a hop on the rising edge of the Xbox B button."""
        pressed = (
            len(buttons) > self.XBOX_HOP_BUTTON
            and bool(buttons[self.XBOX_HOP_BUTTON])
        )
        if pressed and not self._hop_button_was_pressed:
            self.request_hop()
        self._hop_button_was_pressed = pressed

    @staticmethod
    def _deadzone(value: float, threshold: float = 0.1) -> float:
        """Remove stick drift and rescale the useful range to [-1, 1]."""
        if abs(value) <= threshold:
            return 0.0
        sign = 1.0 if value > 0.0 else -1.0
        return sign * (abs(value) - threshold) / (1.0 - threshold)

    def _init_xbox_controller(self) -> None:
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            pygame.quit()
            raise RuntimeError(
                "No Xbox controller detected. Connect it and verify it appears "
                "under /dev/input before retrying."
            )

        self._pygame = pygame
        self._xbox = pygame.joystick.Joystick(0)
        self._xbox.init()
        pygame.display.set_mode((320, 80))
        pygame.display.set_caption("Open Duck Xbox Controller")
        print(
            f"Xbox controller: {self._xbox.get_name()} "
            f"({self._xbox.get_numaxes()} axes, {self._xbox.get_numbuttons()} buttons)"
        )

    @classmethod
    def _xbox_command_from_input(
        cls,
        axes: list[float],
        buttons: list[int],
        hat: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """Convert SDL's raw Xbox values into the trained 3-D command."""

        def axis(index: int) -> float:
            if index >= len(axes):
                return 0.0
            return cls._deadzone(float(axes[index]))

        # SDL's Xbox mapping for the controller used on Linux is:
        # left stick Y = 1 and right stick X = 3.
        left_y = axis(1)
        right_x = axis(3)

        # On this Generic X-Box pad, physically pushing the left stick forward
        # reports a positive SDL Y value.
        lin_vel_x = left_y * cls.COMMANDS_RANGE_X[1] * cls.XBOX_FORWARD_GAIN

        command = np.array(
            [lin_vel_x, 0.0, right_x * cls.COMMANDS_RANGE_THETA[1]],
            dtype=np.float32,
        )
        command[0] = np.clip(command[0], *cls.COMMANDS_RANGE_X)
        command[1] = np.clip(command[1], *cls.COMMANDS_RANGE_Y)
        command[2] = np.clip(command[2], *cls.COMMANDS_RANGE_THETA)

        # A is an explicit stop/dead-man command.
        if buttons and buttons[0]:
            command[:] = 0.0
        return command

    def _update_xbox_command(self) -> None:
        """Map an Xbox controller to the joystick command observation."""
        self._pygame.event.pump()

        axes = [
            self._xbox.get_axis(i) for i in range(self._xbox.get_numaxes())
        ]
        buttons = [
            self._xbox.get_button(i) for i in range(self._xbox.get_numbuttons())
        ]
        self._update_hop_button(buttons)
        hat = self._xbox.get_hat(0) if self._xbox.get_numhats() else (0, 0)
        command = self._xbox_command_from_input(axes, buttons, hat)
        if self.fixed_command is not None:
            command = np.asarray(self.fixed_command, dtype=np.float32)
        elif self.latched_walk:
            # A forward/backward stick flick selects a full-speed walk and
            # keeps it active after the stick returns to center. A stops.
            if buttons and buttons[0]:
                self._latched_forward_command = 0.0
            elif command[0] > 0.01:
                self._latched_forward_command = self.latched_forward_speed
            elif command[0] < -0.01:
                self._latched_forward_command = -self.latched_forward_speed
            command[0] = self._latched_forward_command
        else:
            # Normal Xbox mode is straight-walk only; do not use right-stick
            # yaw unless latched mode was explicitly requested.
            command[2] = 0.0
        self.commands[:3] = command.tolist()

        # This policy was trained with zero head commands for the basic walk
        # experiment. The head remains neutral while using the controller.
        self.commands[3:] = [0.0, 0.0, 0.0, 0.0]

        # Make it obvious that input is reaching the policy.  This reports
        # only changes, so it does not flood the terminal during inference.
        now = time.monotonic()
        command_tuple = tuple(float(value) for value in np.round(command, 3))
        if (
            command_tuple != self._last_controller_command
            and now - self._last_controller_report > 0.1
        ):
            print(f"Xbox command: {command_tuple}")
            self._last_controller_command = command_tuple
            self._last_controller_report = now

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        # Keep this consistent with the current training environment.  The
        # JAX .at[...] expression there is not assigned, so no offset is used.

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        # if not self.standing:
        # ref = self.PRM.get_reference_motion(*command[:3], self.imitation_i)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                # gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                # ref if not self.standing else np.array([]),
                # [self.imitation_i]
                self.imitation_phase,
            ]
        )

        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        if keycode in (self.KEY_JUMP, 32):  # J or Space
            self.request_hop()
            return
        if keycode == self.KEY_NORMAL_WALK:
            self._set_keyboard_gait_mode("normal")
            return
        if keycode == self.KEY_PLAYFUL_WALK:
            self._set_keyboard_gait_mode("playful")
            return
        if keycode == self.KEY_STAND:
            self._set_keyboard_gait_mode("standing")
            return
        # Do not let a viewer key event clear or replace a live controller
        # command. Controller input is sampled every policy tick instead.
        if self.use_xbox_controller:
            if keycode == 59:  # m
                self.phase_frequency_factor -= 0.1
            return
        if keycode == 72:  # h
            self.head_control_mode = not self.head_control_mode
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if not self.head_control_mode:
            if keycode == 265:  # arrow up
                lin_vel_x = self.COMMANDS_RANGE_X[1]
            if keycode == 264:  # arrow down
                lin_vel_x = self.COMMANDS_RANGE_X[0]
            if keycode == 263:  # arrow left
                lin_vel_y = self.COMMANDS_RANGE_Y[1]
            if keycode == 262:  # arrow right
                lin_vel_y = self.COMMANDS_RANGE_Y[0]
            if keycode == 81:  # a
                ang_vel = self.COMMANDS_RANGE_THETA[1]
            if keycode == 69:  # e
                ang_vel = self.COMMANDS_RANGE_THETA[0]
            if keycode == 59:  # m
                self.phase_frequency_factor -= 0.1
        else:
            neck_pitch = 0
            head_pitch = 0
            head_yaw = 0
            head_roll = 0
            if keycode == 265:  # arrow up
                head_pitch = self.NECK_PITCH_RANGE[1]
            if keycode == 264:  # arrow down
                head_pitch = self.NECK_PITCH_RANGE[0]
            if keycode == 263:  # arrow left
                head_yaw = self.HEAD_YAW_RANGE[1]
            if keycode == 262:  # arrow right
                head_yaw = self.HEAD_YAW_RANGE[0]
            if keycode == 81:  # a
                head_roll = self.HEAD_ROLL_RANGE[1]
            if keycode == 69:  # e
                head_roll = self.HEAD_ROLL_RANGE[0]

            self.commands[3] = neck_pitch
            self.commands[4] = head_pitch
            self.commands[5] = head_yaw
            self.commands[6] = head_roll

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def run(self):
        if self.use_xbox_controller:
            self._init_xbox_controller()
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        control_dt = self.sim_dt * self.decimation
                        if self.use_xbox_controller:
                            self._update_xbox_command()
                        elif self.fixed_command is not None:
                            self.commands[:3] = list(self.fixed_command)
                            self.commands[3:] = [0.0, 0.0, 0.0, 0.0]

                        self._prepare_motion_step(control_dt)

                        if (
                            self.motion_mode in ("walking", "settling")
                            and not self.standing
                            and not self.policy_only
                            and self.keyboard_gait_mode != "standing"
                        ):
                            previous_i = self.imitation_i
                            self.imitation_i += 1.0 * self.phase_frequency_factor
                            self.imitation_i = (
                                self.imitation_i % self.PRM.nb_steps_in_period
                            )
                            if self.imitation_i < previous_i:
                                self.gait_cycle += 1
                            # print(self.PRM.nb_steps_in_period)
                            # exit()
                            self.imitation_phase = np.array(
                                [
                                    np.cos(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                    np.sin(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                ]
                            )

                        moving = np.linalg.norm(self.commands[:3]) > 0.01
                        self._update_playful_policy_cue(bool(moving))

                        walking_target = None
                        if self.motion_mode in ("walking", "settling"):
                            obs = self.get_obs(
                                self.data,
                                self.commands,
                            )
                            self.saved_obs.append(obs)
                            action = self._active_locomotion_policy().infer(obs)

                            # self.action_filter.push(action)
                            # action = self.action_filter.get_filtered_action()

                            self.last_last_last_action = self.last_last_action.copy()
                            self.last_last_action = self.last_action.copy()
                            self.last_action = action.copy()

                            self.motor_targets = (
                                self.default_actuator + action * self.action_scale
                            )
                            if self.motion_mode == "walking":
                                self._apply_expressive_style()
                            walking_target = self.motor_targets.copy()

                        self.motor_targets = self._target_for_motion_mode(
                            walking_target, control_dt
                        )

                        # The style layer is intentionally small, but keep
                        # the final target inside the physical joint limits
                        # even if a policy action is already near an edge.
                        self.motor_targets = np.clip(
                            self.motor_targets,
                            self.model.actuator_ctrlrange[:, 0],
                            self.model.actuator_ctrlrange[:, 1],
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * control_dt,
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * control_dt,
                            )

                        self.prev_motor_targets = self.motor_targets.copy()

                        # head_targets = self.commands[3:]
                        # self.motor_targets[5:9] = head_targets
                        self.data.ctrl = self.motor_targets.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))
        finally:
            if self._pygame is not None:
                self._pygame.quit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument(
        "--stand-onnx-model-path",
        type=str,
        help=(
            "Load a separate 101-observation standing policy and select it "
            "when S is pressed."
        ),
    )
    parser.add_argument(
        "--no-imitation-reward",
        action="store_true",
        help="Keep gait-phase inputs at zero for a policy trained without imitation.",
    )
    parser.add_argument(
        "--xbox-controller",
        action="store_true",
        help="Use the first connected Xbox controller for movement commands.",
    )
    parser.add_argument(
        "--fixed-command",
        nargs=3,
        type=float,
        metavar=("X", "Y", "THETA"),
        help="Use a fixed trained command instead of the Xbox input.",
    )
    parser.add_argument(
        "--latched-walk",
        action="store_true",
        help="Latch forward/reverse walking until A is pressed; right stick turns.",
    )
    parser.add_argument(
        "--latched-forward-speed",
        type=float,
        default=MjInfer.DEFAULT_LATCHED_FORWARD_SPEED,
        help="Forward command used by latched walk mode (default: 0.20).",
    )
    parser.add_argument(
        "--expressive-walk",
        action="store_true",
        help="Add an excited two-beat head bobble, antenna wiggle, and springy step.",
    )
    parser.add_argument(
        "--playful-policy",
        action="store_true",
        help=(
            "Drive the reserved alternating accent-step cue expected by a "
            "policy trained with --playful-walk."
        ),
    )
    parser.add_argument(
        "--head-bobble-amplitude",
        type=float,
        default=MjInfer.DEFAULT_HEAD_BOBBLE_AMPLITUDE,
        help="Head/neck bobble amplitude in radians (default: 0.12).",
    )
    parser.add_argument(
        "--antenna-wiggle-amplitude",
        type=float,
        default=MjInfer.DEFAULT_ANTENNA_WIGGLE_AMPLITUDE,
        help="Head roll used to visually wiggle the rigid antennae (default: 0.10).",
    )
    parser.add_argument(
        "--step-bounce-amplitude",
        type=float,
        default=MjInfer.DEFAULT_STEP_BOUNCE_AMPLITUDE,
        help="Symmetric knee bounce amplitude in radians (default: 0.018).",
    )

    args = parser.parse_args()

    mjinfer = MjInfer(
        model_path=args.model_path,
        reference_data=args.reference_data,
        onnx_model_path=args.onnx_model_path,
        standing=args.standing,
        stand_onnx_model_path=args.stand_onnx_model_path,
        policy_only=args.no_imitation_reward,
        xbox_controller=args.xbox_controller,
        fixed_command=tuple(args.fixed_command) if args.fixed_command else None,
        latched_walk=args.latched_walk,
        latched_forward_speed=args.latched_forward_speed,
        expressive_walk=args.expressive_walk,
        playful_policy=args.playful_policy,
        head_bobble_amplitude=args.head_bobble_amplitude,
        antenna_wiggle_amplitude=args.antenna_wiggle_amplitude,
        step_bounce_amplitude=args.step_bounce_amplitude,
    )
    mjinfer.run()
