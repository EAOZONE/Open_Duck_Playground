"""MuJoCo PPO task for recovering to standing from a face-down fall."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from playground.common.rewards import cost_action_rate, cost_torques
from . import jump
from .getup_motion import (
    GETUP_DURATION,
    PRONE_ROOT_HEIGHT,
    PLANT_REACHED,
    STAND_REACHED,
    STANDING_ROOT_HEIGHT,
    clip_pose,
    phase_features,
    root_trajectory,
    trajectory_pose,
)


USE_MOTOR_SPEED_LIMITS = True
STANDING_FOOT_NORMAL_MIN = float(np.cos(np.deg2rad(15.0)))
STANDING_HOLD_STEPS = 25

# A physically settled late-stage failure from the goal-only policy.  The
# torso is upright and raised, the left sole is flat, and the right sole is
# trapped on its edge.  Its reflected counterpart exercises the same recovery
# with the opposite leg.  These are reset curriculum states only: they are not
# exposed to the actor and do not introduce an animation or phase clock.
RIGHT_FOOT_STUCK_QPOS = np.array(
    [
        -0.02228310,
        -0.04414041,
        0.16983342,
        0.93483958,
        -0.09751029,
        -0.01571871,
        0.34105663,
        -0.18418314,
        -0.15787081,
        -0.45750098,
        1.30663253,
        -0.85950728,
        -0.21516810,
        -0.17853980,
        -0.12323098,
        0.10465479,
        0.20232278,
        -0.15703518,
        1.09364886,
        0.93871786,
        -0.76786770,
    ],
    dtype=np.float32,
)
LEFT_FOOT_STUCK_QPOS = np.array(
    [
        -0.02228310,
        0.04414041,
        0.16983342,
        0.93483958,
        0.09751029,
        -0.01571871,
        -0.34105663,
        -0.20232278,
        0.15703518,
        -1.09364886,
        0.93871786,
        -0.76786770,
        -0.21516810,
        -0.17853980,
        0.12323098,
        -0.10465479,
        0.18418314,
        0.15787081,
        0.45750098,
        1.30663253,
        -0.85950728,
    ],
    dtype=np.float32,
)


def foot_flatness_scores(foot_normal_z: jax.Array) -> jax.Array:
    """Returns dense per-foot scores that reach one at the 15 degree goal."""
    return jp.square(
        jp.clip(
            (foot_normal_z + 1.0) / (STANDING_FOOT_NORMAL_MIN + 1.0),
            0.0,
            1.0,
        )
    )


def leg_reposition_cost(
    foot_normal_z: jax.Array,
    knee_pos: jax.Array,
    knee_target: jax.Array,
    hip_pitch_pos: jax.Array,
    hip_pitch_target: jax.Array,
) -> jax.Array:
    """Penalizes the extended/displaced leg that is trapping a tilted foot."""
    knee_extension = jp.square(jp.clip((knee_target - 0.15 - knee_pos) / 0.6, 0.0, 1.0))
    hip_pitch_error = jp.square(
        jp.clip(
            (jp.abs(hip_pitch_pos - hip_pitch_target) - 0.15) / 0.45,
            0.0,
            1.0,
        )
    )
    tilt_gate = jp.clip(
        (STANDING_FOOT_NORMAL_MIN - foot_normal_z) / (STANDING_FOOT_NORMAL_MIN - 0.5),
        0.0,
        1.0,
    )
    return jp.mean(tilt_gate * (knee_extension + 0.5 * hip_pitch_error))


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        # Seven seconds for the motion plus one second to prove the final
        # stance is stable.
        episode_length=400,
        action_repeat=1,
        # Get-up contacts are less repeatable than a hop, so the residual
        # needs enough authority to substantially revise a reference pose.
        action_scale=1.0,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,
        # The dense standing terms intentionally exceed the shared task's
        # 100-point cap.  Keep their ordering signal intact until the sparse
        # sustained-standing bonus becomes active.
        reward_clip=200.0,
        # Goal-only mode keeps the face-down reset and standing objective but
        # removes the time-indexed animation, phase input, and moving target.
        use_reference_motion=True,
        # Goal-only training reset mix: [face-down, stuck-foot, near-stand].
        # Evaluation overrides this with [1, 0, 0].
        goal_only_reset_mix=[0.5, 0.25, 0.25],
        use_worst_foot_flatness=True,
        use_leg_reposition_cost=True,
        # The standalone environment and evaluation always start face-down.
        # The runner enables this only on its training copy to teach later
        # reference segments before stitching the whole maneuver together.
        reference_state_init_probability=0.0,
        noise_config=config_dict.create(
            level=0.15,
            action_min_delay=0,
            action_max_delay=2,
            imu_min_delay=0,
            imu_max_delay=2,
            scales=config_dict.create(
                hip_pos=0.025,
                knee_pos=0.04,
                ankle_pos=0.05,
                joint_vel=1.5,
                gravity=0.04,
                accelerometer=0.03,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                pose_tracking=1.0,
                orientation_tracking=6.0,
                height_tracking=8.0,
                upright=15.0,
                standing_height=60.0,
                standing=120.0,
                feet_contact=2.0,
                feet_flat=60.0,
                leg_reposition=-30.0,
                body_contact=-8.0,
                angular_velocity=-0.1,
                horizontal_drift=-1.0,
                torques=-1.0e-3,
                action_rate=-0.04,
                alive=0.1,
            )
        ),
    )


class GetUp(jump.Jump):
    """Learn a reference-guided face-down get-up and stable final stance."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(task=task, config=config, config_overrides=config_overrides)
        self._body_contact_geom_id = np.array(
            [
                self._mj_model.geom("trunk_collision").id,
                self._mj_model.geom("head_collision").id,
            ]
        )
        self._knee_actuator_idx = jp.array(
            [
                self.actuator_names.index("left_knee"),
                self.actuator_names.index("right_knee"),
            ]
        )
        self._hip_pitch_actuator_idx = jp.array(
            [
                self.actuator_names.index("left_hip_pitch"),
                self.actuator_names.index("right_hip_pitch"),
            ]
        )
        if self.mjx_model.nq != RIGHT_FOOT_STUCK_QPOS.shape[0]:
            raise ValueError("Get-up curriculum qpos does not match the loaded model")
        reset_mix = np.asarray(self._config.goal_only_reset_mix, dtype=float)
        if (
            reset_mix.shape != (3,)
            or np.any(reset_mix < 0.0)
            or not np.isclose(reset_mix.sum(), 1.0)
        ):
            raise ValueError(
                "goal_only_reset_mix must contain three nonnegative values "
                "that sum to one"
            )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng,
            xy_rng,
            roll_rng,
            pitch_rng,
            yaw_rng,
            phase_gate_rng,
            phase_rng,
            joint_rng,
            vel_rng,
            reset_mode_rng,
            stuck_side_rng,
        ) = jax.random.split(rng, 11)
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        dxy = jax.random.uniform(xy_rng, (2,), minval=-0.015, maxval=0.015)
        # Bias reference-state initialization toward the difficult low-crouch
        # and rise stages while retaining 30% complete face-down attempts.
        sampled_time = STAND_REACHED * jp.sqrt(
            jax.random.uniform(phase_rng, (), minval=0.0, maxval=1.0)
        )
        use_reference_state = jax.random.uniform(phase_gate_rng, ()) < (
            self._config.reference_state_init_probability
        )
        initial_time = jp.where(use_reference_state, sampled_time, 0.0)
        target_root_pos, target_root_quat = root_trajectory(initial_time, dxy, jp)
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[:2].set(dxy)
        base_qpos = base_qpos.at[2].set(target_root_pos[2])
        quat = target_root_quat
        roll = jax.random.uniform(roll_rng, (), minval=-0.06, maxval=0.06)
        pitch = jax.random.uniform(pitch_rng, (), minval=-0.08, maxval=0.08)
        yaw = jax.random.uniform(yaw_rng, (), minval=-0.12, maxval=0.12)
        quat = math.quat_mul(
            math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw), quat
        )
        quat = math.quat_mul(
            math.axis_angle_to_quat(jp.array([0.0, 1.0, 0.0]), pitch), quat
        )
        quat = math.quat_mul(
            math.axis_angle_to_quat(jp.array([1.0, 0.0, 0.0]), roll), quat
        )
        base_qpos = base_qpos.at[3:7].set(quat)
        qpos = self.set_floating_base_qpos(base_qpos, qpos)

        initial_pose = (
            trajectory_pose(initial_time, self._default_actuator, jp)
            if self._config.use_reference_motion
            else self._default_actuator
        )
        if self._config.use_reference_motion:
            initial_pose = (
                initial_pose
                + jax.random.uniform(
                    joint_rng, (self._actuators,), minval=-1.0, maxval=1.0
                )
                * self._qpos_noise_scale
            )
            qpos = self.set_actuator_joints_qpos(initial_pose, qpos)

        # In goal-only mode, mix complete face-down attempts with late-stage
        # correction states.  Reference-guided training retains its existing
        # reference-state initialization unchanged.
        reset_mode = jp.array(0, dtype=jp.int32)
        if not self._config.use_reference_motion:
            reset_mix = jp.asarray(self._config.goal_only_reset_mix)
            reset_draw = jax.random.uniform(reset_mode_rng, ())
            use_stuck_state = (reset_draw >= reset_mix[0]) & (
                reset_draw < reset_mix[0] + reset_mix[1]
            )
            use_near_standing = reset_draw >= reset_mix[0] + reset_mix[1]

            use_left_stuck = jax.random.bernoulli(stuck_side_rng)
            stuck_qpos = jp.where(
                use_left_stuck,
                jp.asarray(LEFT_FOOT_STUCK_QPOS),
                jp.asarray(RIGHT_FOOT_STUCK_QPOS),
            )
            stuck_qpos = stuck_qpos.at[:2].add(dxy)

            near_qpos = self._init_q
            near_base_qpos = self.get_floating_base_qpos(near_qpos)
            near_base_qpos = near_base_qpos.at[:2].set(dxy)
            near_quat = near_base_qpos[3:7]
            near_quat = math.quat_mul(
                math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw),
                near_quat,
            )
            near_quat = math.quat_mul(
                math.axis_angle_to_quat(jp.array([0.0, 1.0, 0.0]), pitch),
                near_quat,
            )
            near_quat = math.quat_mul(
                math.axis_angle_to_quat(jp.array([1.0, 0.0, 0.0]), roll),
                near_quat,
            )
            near_base_qpos = near_base_qpos.at[3:7].set(near_quat)
            near_qpos = self.set_floating_base_qpos(near_base_qpos, near_qpos)

            qpos = jp.where(use_stuck_state, stuck_qpos, qpos)
            qpos = jp.where(use_near_standing, near_qpos, qpos)
            initial_pose = self.get_actuator_joints_qpos(qpos)
            initial_pose = (
                initial_pose
                + jax.random.uniform(
                    joint_rng,
                    (self._actuators,),
                    minval=-1.0,
                    maxval=1.0,
                )
                * self._qpos_noise_scale
            )
            qpos = self.set_actuator_joints_qpos(initial_pose, qpos)
            reset_mode = jp.where(
                use_stuck_state,
                jp.array(1, dtype=jp.int32),
                reset_mode,
            )
            reset_mode = jp.where(
                use_near_standing,
                jp.array(2, dtype=jp.int32),
                reset_mode,
            )

        qvel = self.set_floating_base_qvel(
            jax.random.uniform(vel_rng, (6,), minval=-0.03, maxval=0.03), qvel
        )

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=initial_pose)
        contact = self._contacts(data)
        getup_phase = (
            phase_features(initial_time, jp)
            if self._config.use_reference_motion
            else jp.zeros(3)
        )
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "getup_time": initial_time,
            "root_start_xy": self.get_floating_base_qpos(qpos)[:2],
            "success_steps": jp.array(0, dtype=jp.int32),
            "standing_streak": jp.array(0, dtype=jp.int32),
            "max_standing_streak": jp.array(0, dtype=jp.int32),
            "reset_mode": reset_mode,
            "last_contact": contact,
            "last_act": jp.zeros(self._actuators),
            "last_last_act": jp.zeros(self._actuators),
            "last_last_last_act": jp.zeros(self._actuators),
            "motor_targets": initial_pose,
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            "imitation_phase": jp.zeros(2),
            # Jump's observation builder is intentionally reused so get-up
            # exports the same 104-value hardware observation contract.
            "jump_phase": getup_phase,
            "airborne_steps": jp.array(0, dtype=jp.int32),
            "ever_airborne": jp.array(False),
            "max_root_height": data.qpos[self._floating_base_height_addr],
        }

        metrics = {}
        for name, scale in self._config.reward_config.scales.items():
            metrics[("reward/" if scale > 0 else "cost/") + name] = jp.zeros(())
        metrics["success_steps"] = jp.zeros(())
        metrics["max_standing_streak"] = jp.zeros(())
        metrics["upright"] = jp.zeros(())
        metrics["root_height"] = jp.zeros(())

        obs = self._get_obs(data, info, contact)
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.zeros(()),
            done=jp.zeros(()),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["rng"], delay_rng = jax.random.split(state.info["rng"])
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            delay_rng,
            (),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        delayed_action = action_history.reshape((-1, self._actuators))[action_idx]

        if self._config.use_reference_motion:
            reference_pose = clip_pose(
                trajectory_pose(state.info["getup_time"], self._default_actuator, jp),
                self._actuator_lowers,
                self._actuator_uppers,
                jp,
            )
        else:
            reference_pose = self._default_actuator
        motor_targets = reference_pose + delayed_action * self._config.action_scale
        motor_targets = jp.clip(
            motor_targets, self._actuator_lowers, self._actuator_uppers
        )
        if USE_MOTOR_SPEED_LIMITS:
            previous = state.info["motor_targets"]
            motor_targets = jp.clip(
                motor_targets,
                previous - self._config.max_motor_velocity * self.dt,
                previous + self._config.max_motor_velocity * self.dt,
            )
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        contact = self._contacts(data)
        state.info["last_contact"] = contact
        state.info["getup_time"] = jp.minimum(
            GETUP_DURATION, state.info["getup_time"] + self.dt
        )
        state.info["jump_phase"] = (
            phase_features(state.info["getup_time"], jp)
            if self._config.use_reference_motion
            else jp.zeros(3)
        )
        root_height = data.qpos[self._floating_base_height_addr]
        state.info["max_root_height"] = jp.maximum(
            state.info["max_root_height"], root_height
        )

        upright = self.get_gravity(data)[-1]
        foot_normal_z = self._foot_normal_z(data)
        local_speed = jp.linalg.norm(self.get_local_linvel(data))
        phase_ready = (
            state.info["getup_time"] >= STAND_REACHED
            if self._config.use_reference_motion
            else jp.array(True)
        )
        standing_now = (
            phase_ready
            & (upright > 0.9)
            & (root_height > 0.13)
            & jp.all(contact)
            & jp.all(foot_normal_z > STANDING_FOOT_NORMAL_MIN)
            & (local_speed < 0.2)
        )
        standing_streak = jp.where(standing_now, state.info["standing_streak"] + 1, 0)
        state.info["standing_streak"] = standing_streak
        state.info["max_standing_streak"] = jp.maximum(
            state.info["max_standing_streak"], standing_streak
        )
        success = standing_streak >= STANDING_HOLD_STEPS
        state.info["success_steps"] += success.astype(jp.int32)

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)
        rewards = self._get_reward(data, delayed_action, state.info, contact, success)
        scaled_rewards = {
            key: value * self._config.reward_config.scales[key]
            for key, value in rewards.items()
        }
        reward = (
            jp.clip(
                sum(scaled_rewards.values()),
                -self._config.reward_clip,
                self._config.reward_clip,
            )
            * self.dt
        )

        state.info["step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        for key, value in scaled_rewards.items():
            scale = self._config.reward_config.scales[key]
            state.metrics[("reward/" if scale > 0 else "cost/") + key] = (
                value if scale > 0 else -value
            )
        state.metrics["success_steps"] = state.info["success_steps"].astype(jp.float32)
        state.metrics["max_standing_streak"] = state.info["max_standing_streak"].astype(
            jp.float32
        )
        state.metrics["upright"] = upright
        state.metrics["root_height"] = root_height
        return state.replace(
            data=data, obs=obs, reward=reward, done=done.astype(reward.dtype)
        )

    def _body_contacts(self, data: mjx.Data) -> jax.Array:
        return jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._body_contact_geom_id
            ]
        )

    def _foot_normal_z(self, data: mjx.Data) -> jax.Array:
        """Returns each sole normal's alignment with world up."""
        return data.site_xmat[self._feet_site_id, 2, 2]

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        root_height = data.qpos[self._floating_base_height_addr]
        return (
            (root_height < -0.02)
            | (root_height > 0.55)
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        contact: jax.Array,
        success: jax.Array,
    ) -> dict[str, jax.Array]:
        t = info["getup_time"]
        if self._config.use_reference_motion:
            target_pose = clip_pose(
                trajectory_pose(t, self._default_actuator, jp),
                self._actuator_lowers,
                self._actuator_uppers,
                jp,
            )
        else:
            target_pose = self._default_actuator
        joint_error = jp.mean(
            jp.square(self.get_actuator_joints_qpos(data.qpos) - target_pose)
        )
        pose_tracking = jp.exp(-5.0 * joint_error)

        if self._config.use_reference_motion:
            target_pos, target_quat = root_trajectory(t, info["root_start_xy"], jp)
            target_up = jp.array(
                [
                    2.0
                    * (
                        target_quat[1] * target_quat[3]
                        + target_quat[0] * target_quat[2]
                    ),
                    2.0
                    * (
                        target_quat[2] * target_quat[3]
                        - target_quat[0] * target_quat[1]
                    ),
                    1.0 - 2.0 * (jp.square(target_quat[1]) + jp.square(target_quat[2])),
                ]
            )
        else:
            target_pos = jp.array(
                [
                    info["root_start_xy"][0],
                    info["root_start_xy"][1],
                    STANDING_ROOT_HEIGHT,
                ]
            )
            target_up = jp.array([0.0, 0.0, 1.0])
        current_up = self.get_gravity(data)
        orientation_tracking = jp.exp(-3.0 * jp.sum(jp.square(current_up - target_up)))
        root_height = data.qpos[self._floating_base_height_addr]
        height_tracking = jp.exp(-jp.square(root_height - target_pos[2]) / 0.0025)

        progress = (
            jp.clip(t / STAND_REACHED, 0.0, 1.0)
            if self._config.use_reference_motion
            else jp.array(1.0)
        )
        # Face-down has up-z ~= 0, so it must not receive half of the upright
        # reward.  Once the torso has rolled up, height has a dense gradient
        # all the way from the low 0.08 m crouch reached by the first policy to
        # the 0.16 m standing pose.
        upright = jp.clip((current_up[-1] - 0.15) / 0.85, 0.0, 1.0)
        height_progress = jp.clip(
            (root_height - PRONE_ROOT_HEIGHT)
            / (STANDING_ROOT_HEIGHT - PRONE_ROOT_HEIGHT),
            0.0,
            1.0,
        )
        clearance_progress = (
            jp.clip(
                (t - PLANT_REACHED) / (STAND_REACHED - PLANT_REACHED),
                0.0,
                1.0,
            )
            if self._config.use_reference_motion
            else upright
        )
        standing_height = upright * height_progress * clearance_progress
        feet_contact = jp.mean(contact.astype(jp.float32)) * clearance_progress
        # A collision boolean alone also accepts toe, edge, and even inverted
        # foot contacts.  Use a dense sole-normal score so a badly tilted foot
        # still receives a useful gradient toward world up.  Contact is scored
        # separately, allowing the policy to briefly lift a stuck foot, turn
        # it, and then plant it flat.
        foot_normal_z = jp.clip(self._foot_normal_z(data), -1.0, 1.0)
        per_foot_flatness = foot_flatness_scores(foot_normal_z)
        aggregate_flatness = jp.where(
            self._config.use_worst_foot_flatness,
            jp.min(per_foot_flatness),
            jp.mean(per_foot_flatness),
        )
        feet_flat = aggregate_flatness * upright * height_progress

        # A tilted foot is commonly trapped by an extended knee and displaced
        # hip pitch.  Penalize that configuration rather than positively
        # rewarding a temporary posture: the cost vanishes when either the
        # sole is flat or the leg has made enough room to reposition.
        knee_pos = self.get_actuator_joints_qpos(data.qpos)[self._knee_actuator_idx]
        knee_target = self._default_actuator[self._knee_actuator_idx]
        hip_pitch_pos = self.get_actuator_joints_qpos(data.qpos)[
            self._hip_pitch_actuator_idx
        ]
        hip_pitch_target = self._default_actuator[self._hip_pitch_actuator_idx]
        leg_reposition = jp.where(
            self._config.use_leg_reposition_cost,
            leg_reposition_cost(
                foot_normal_z,
                knee_pos,
                knee_target,
                hip_pitch_pos,
                hip_pitch_target,
            )
            * upright
            * height_progress,
            0.0,
        )
        body_contact = (
            jp.any(self._body_contacts(data)).astype(jp.float32) * clearance_progress
        )
        angular_velocity = jp.sum(jp.square(self.get_global_angvel(data))) * progress
        root_xy = data.qpos[
            self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2
        ]
        horizontal_drift = jp.sum(jp.square(root_xy - info["root_start_xy"]))

        return {
            "pose_tracking": pose_tracking,
            "orientation_tracking": orientation_tracking,
            "height_tracking": height_tracking,
            "upright": upright,
            "standing_height": standing_height,
            "standing": success.astype(jp.float32),
            "feet_contact": feet_contact,
            "feet_flat": feet_flat,
            "leg_reposition": leg_reposition,
            "body_contact": body_contact,
            "angular_velocity": angular_velocity,
            "horizontal_drift": horizontal_drift,
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "alive": jp.array(1.0),
        }
