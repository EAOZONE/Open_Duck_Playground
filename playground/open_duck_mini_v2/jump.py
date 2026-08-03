"""MuJoCo PPO task for a one-shot Open Duck vertical jump."""

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
from . import constants
from . import base as open_duck_mini_v2_base
from .jump_motion import (
    JUMP_DURATION,
    RECOVERY_START,
    clip_pose,
    phase_features,
    trajectory_pose,
)


USE_MOTOR_SPEED_LIMITS = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=150,
        action_repeat=1,
        action_scale=0.35,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,
        noise_config=config_dict.create(
            # Start the jump curriculum close to the nominal simulator.  A
            # later robust-training pass can raise this after a stable hop is
            # learned.
            level=0.1,
            action_min_delay=0,
            action_max_delay=1,
            imu_min_delay=0,
            imu_max_delay=1,
            scales=config_dict.create(
                hip_pos=0.02,
                knee_pos=0.03,
                ankle_pos=0.04,
                joint_vel=1.5,
                gravity=0.05,
                accelerometer=0.03,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                pose_tracking=3.0,
                height=20.0,
                clearance=15.0,
                vertical_velocity=2.0,
                airborne=5.0,
                jump_failure=-25.0,
                upright=1.5,
                angular_velocity=4.0,
                fall=-8.0,
                landing=20.0,
                stable_hold=8.0,
                horizontal_drift=-1.0,
                torques=-1.0e-3,
                action_rate=-0.05,
                impact=-0.5,
                alive=1.0,
            )
        ),
    )


class Jump(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Train a one-shot vertical hop and stable landing."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = jp.array(self._mj_model.keyframe("home").ctrl)
        self._actuators = self._mj_model.nu
        self._actuator_lowers = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._actuator_uppers = jp.array(self._mj_model.actuator_ctrlrange[:, 1])

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._site_id = self._mj_model.site("imu").id
        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )
        self._floating_base_height_addr = self._floating_base_qpos_addr + 2

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        qpos_noise_scale = np.zeros(self._actuators, dtype=np.float32)
        for idx, name in enumerate(self.actuator_names):
            if name in constants.JOINTS_ORDER_NO_HEAD:
                if "_hip" in name:
                    qpos_noise_scale[idx] = self._config.noise_config.scales.hip_pos
                elif "_knee" in name:
                    qpos_noise_scale[idx] = self._config.noise_config.scales.knee_pos
                elif "_ankle" in name:
                    qpos_noise_scale[idx] = self._config.noise_config.scales.ankle_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, xy_rng, yaw_rng, joint_rng, vel_rng = jax.random.split(rng, 5)
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        dxy = jax.random.uniform(xy_rng, (2,), minval=-0.01, maxval=0.01)
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[:2].add(dxy)
        yaw = jax.random.uniform(yaw_rng, (1,), minval=-0.08, maxval=0.08)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        base_qpos = base_qpos.at[3:7].set(
            math.quat_mul(base_qpos[3:7], quat)
        )
        qpos = self.set_floating_base_qpos(base_qpos, qpos)

        qpos = self.set_actuator_joints_qpos(
            self._default_actuator
            + jax.random.uniform(
                joint_rng, (self._actuators,), minval=-1.0, maxval=1.0
            )
            * self._qpos_noise_scale,
            qpos,
        )
        qvel = self.set_floating_base_qvel(
            jax.random.uniform(vel_rng, (6,), minval=-0.02, maxval=0.02), qvel
        )

        data = mjx_env.init(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=self.get_actuator_joints_qpos(qpos),
        )
        contact = self._contacts(data)
        root_height = data.qpos[self._floating_base_height_addr]
        phase = phase_features(jp.array(0.0), jp)
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "jump_time": jp.array(0.0),
            "root_height_start": root_height,
            "max_root_height": root_height,
            "airborne_steps": jp.array(0, dtype=jp.int32),
            "ever_airborne": jp.array(False),
            "landing_steps": jp.array(0, dtype=jp.int32),
            "last_contact": contact,
            "last_act": jp.zeros(self._actuators),
            "last_last_act": jp.zeros(self._actuators),
            "last_last_last_act": jp.zeros(self._actuators),
            "motor_targets": self._default_actuator,
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            "imitation_phase": jp.zeros(2),
            "jump_phase": phase,
        }

        metrics = {}
        for name, scale in self._config.reward_config.scales.items():
            if scale > 0:
                metrics[f"reward/{name}"] = jp.zeros(())
            elif scale < 0:
                metrics[f"cost/{name}"] = jp.zeros(())
        metrics["max_root_height"] = jp.zeros(())
        metrics["airborne_steps"] = jp.zeros(())
        metrics["landing_steps"] = jp.zeros(())

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
        state.info["rng"], action_delay_rng = jax.random.split(state.info["rng"])
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )[0]
        action_w_delay = action_history.reshape((-1, self._actuators))[action_idx]
        action_w_delay = action_w_delay.at[5:9].set(0.0)

        # Use the shared phase trajectory as a reference and train PPO on a
        # normalized residual.  This preserves the 14-output contract while
        # making the jump reachable during early exploration; a zero policy
        # action follows the same motion as the scripted controller.
        reference_pose = clip_pose(
            trajectory_pose(state.info["jump_time"], self._default_actuator, jp),
            self._actuator_lowers,
            self._actuator_uppers,
            jp,
        )
        motor_targets = reference_pose + action_w_delay * self._config.action_scale
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
        root_height = data.qpos[self._floating_base_height_addr]
        no_foot_contact = ~jp.any(contact)
        both_feet_contact = jp.all(contact)
        ever_airborne = state.info["ever_airborne"] | no_foot_contact
        airborne_steps = state.info["airborne_steps"] + no_foot_contact.astype(jp.int32)
        landed = (
            ever_airborne
            & both_feet_contact
            & ~jp.all(state.info["last_contact"])
        )
        landing_steps = state.info["landing_steps"] + landed.astype(jp.int32)
        state.info["ever_airborne"] = ever_airborne
        state.info["airborne_steps"] = airborne_steps
        state.info["landing_steps"] = landing_steps
        state.info["max_root_height"] = jp.maximum(
            state.info["max_root_height"], root_height
        )
        state.info["last_contact"] = contact

        state.info["jump_time"] = jp.minimum(
            JUMP_DURATION, state.info["jump_time"] + self.dt
        )
        state.info["jump_phase"] = phase_features(state.info["jump_time"], jp)
        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action_w_delay, state.info, contact, landed)
        scaled_rewards = {
            key: value * self._config.reward_config.scales[key]
            for key, value in rewards.items()
        }
        reward = jp.clip(sum(scaled_rewards.values()), -100.0, 100.0) * self.dt

        state.info["step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        for key, value in scaled_rewards.items():
            scale = self._config.reward_config.scales[key]
            if scale > 0:
                state.metrics[f"reward/{key}"] = value
            elif scale < 0:
                state.metrics[f"cost/{key}"] = -value
        state.metrics["max_root_height"] = state.info["max_root_height"]
        state.metrics["airborne_steps"] = state.info["airborne_steps"].astype(jp.float32)
        state.metrics["landing_steps"] = state.info["landing_steps"].astype(jp.float32)

        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    def _contacts(self, data: mjx.Data) -> jax.Array:
        return jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        upright = self.get_gravity(data)[-1]
        return (
            (upright < 0.2)
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
        )

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = gyro + (
            2 * jax.random.uniform(noise_rng, gyro.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.joint_vel
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = accelerometer + (
            2 * jax.random.uniform(noise_rng, accelerometer.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.accelerometer
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = gravity + (
            2 * jax.random.uniform(noise_rng, gravity.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.gravity

        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        noisy_gravity = imu_history.reshape((-1, 3))[0]

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = joint_angles + (
            2 * jax.random.uniform(noise_rng, joint_angles.shape) - 1
        ) * self._config.noise_config.level * self._qpos_noise_scale
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = joint_vel + (
            2 * jax.random.uniform(noise_rng, joint_vel.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.joint_vel

        command = jp.zeros(7)
        state = jp.hstack(
            [
                noisy_gyro,
                noisy_accelerometer,
                command,
                noisy_joint_angles - self._default_actuator,
                noisy_joint_vel * self._config.dof_vel_scale,
                info["last_act"],
                info["last_last_act"],
                info["last_last_last_act"],
                info["motor_targets"],
                contact,
                info["imitation_phase"],
                info["jump_phase"],
            ]
        )

        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        privileged_state = jp.hstack(
            [
                state,
                gyro,
                accelerometer,
                gravity,
                self.get_local_linvel(data),
                self.get_global_angvel(data),
                joint_angles - self._default_actuator,
                joint_vel,
                data.qpos[self._floating_base_height_addr],
                data.actuator_force,
                contact,
                feet_vel,
                info["airborne_steps"],
                info["ever_airborne"],
                info["max_root_height"],
            ]
        )
        return {"state": state, "privileged_state": privileged_state}

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        contact: jax.Array,
        landed: jax.Array,
    ) -> dict[str, jax.Array]:
        t = info["jump_time"]
        target_pose = clip_pose(
            trajectory_pose(t, self._default_actuator, jp),
            self._actuator_lowers,
            self._actuator_uppers,
            jp,
        )
        joint_error = jp.mean(
            jp.square(self.get_actuator_joints_qpos(data.qpos) - target_pose)
        )
        pose_tracking = jp.exp(-8.0 * joint_error)

        root_height = data.qpos[self._floating_base_height_addr]
        height_delta = root_height - info["root_height_start"]
        flight_phase = jp.clip((t - 0.55) / 0.50, 0.0, 1.0)
        desired_height = 0.05 * jp.sin(flight_phase * jp.pi)
        height = jp.exp(-jp.square(height_delta - desired_height) / 0.0009)
        clearance = jp.clip(height_delta / 0.04, 0.0, 1.0)
        desired_vertical_velocity = jp.where(
            (t >= 0.55) & (t < 1.05),
            0.05 * jp.pi / 0.50 * jp.cos(flight_phase * jp.pi),
            0.0,
        )
        vertical_velocity = jp.exp(
            -jp.square(data.qvel[self._floating_base_qvel_addr + 2] - desired_vertical_velocity)
            / 0.01
        )
        airborne = (info["airborne_steps"] >= 3).astype(jp.float32)
        jump_failure = (
            (t >= 0.55) & (~info["ever_airborne"])
        ).astype(jp.float32)
        upright = jp.clip(self.get_gravity(data)[-1], 0.0, 1.0)
        global_angvel = self.get_global_angvel(data)
        angular_velocity = jp.exp(
            -jp.sum(jp.square(global_angvel[:2])) / 0.20
        )
        fall = (upright < 0.2).astype(jp.float32)
        landing = landed.astype(jp.float32) * upright

        local_linvel = self.get_local_linvel(data)
        horizontal_drift = jp.sum(jp.square(local_linvel[:2]))
        stable_hold = (
            (t >= RECOVERY_START)
            & jp.all(contact)
            & (upright > 0.9)
            & (jp.linalg.norm(local_linvel[:2]) < 0.15)
        ).astype(jp.float32)
        impact = jp.sum(jp.square(self.get_global_linvel(data))) * landed.astype(jp.float32)

        return {
            "pose_tracking": pose_tracking,
            "height": height,
            "clearance": clearance,
            "vertical_velocity": vertical_velocity,
            "airborne": airborne,
            "jump_failure": jump_failure,
            "upright": upright,
            "angular_velocity": angular_velocity,
            "fall": fall,
            "landing": landing,
            "stable_hold": stable_hold,
            "horizontal_drift": horizontal_drift,
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "impact": impact,
            "alive": jp.array(1.0),
        }
