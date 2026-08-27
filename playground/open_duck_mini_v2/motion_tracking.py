"""Unified stand, locomotion, and upright motion-tracking task."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from playground.common.motion_bundle import MotionBundle, REFERENCE_OFFSETS_SECONDS
from playground.common.rewards import cost_action_rate, cost_torques, reward_alive

from . import joystick


REFERENCE_OFFSETS = jp.asarray(REFERENCE_OFFSETS_SECONDS, dtype=jp.float32)
TRANSITION_SECONDS = 0.4


def _bundle_path() -> str:
    return os.environ.get(
        "OPEN_DUCK_MOTION_BUNDLE",
        str(Path(__file__).resolve().parent / "data" / "motion_bundle_v1.npz"),
    )


def default_config() -> config_dict.ConfigDict:
    config = joystick.default_config()
    config.episode_length = 1000
    config.motion_switch_seconds = 5.0
    # This task uses the direct-frame bundle and must not instantiate the
    # legacy polynomial reference loader during its superclass setup.
    config.use_imitation_reward = False
    config.reference_state_init_probability = 0.7
    config.encoder_bias_scale = 0.0
    config.motion_mix = [0.30, 0.45, 0.25]
    config.reward_config.scales = config_dict.create(
        alive=10.0,
        reference_joint_position=5.0,
        reference_joint_velocity=0.5,
        reference_orientation=2.0,
        reference_height=2.0,
        reference_linear_velocity=2.5,
        reference_angular_velocity=2.0,
        reference_contacts=2.0,
        torques=-1.0e-3,
        action_rate=-0.5,
        action_second_difference=-0.1,
        foot_slip=-1.0,
        soft_joint_limits=-1.0,
        standing_drift=-2.0,
        termination=-200.0,
    )
    return config


class MotionTracking(joystick.Joystick):
    """Track direct-frame motion references using deployable observations."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict | None = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(task=task, config=config or default_config(), config_overrides=config_overrides)
        self.use_imitation_reward = False
        bundle = MotionBundle.load(_bundle_path())
        arrays = bundle.padded_arrays()
        self.motion_names = tuple(clip.name for clip in bundle.clips)
        self._motion_fps = float(bundle.manifest["fps"])
        self._motion = {name: jp.asarray(value) for name, value in arrays.items()}
        train_mask = arrays["splits"] == 0
        kind_weights = np_kind_weights(arrays["kinds"], train_mask, self._config.motion_mix)
        self._motion_weights = jp.asarray(kind_weights)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        del rng
        # Commands are derived from the selected reference, never sampled independently.
        return jp.zeros(7)

    def _choose_motion(self, rng: jax.Array) -> jax.Array:
        return jax.random.choice(rng, len(self.motion_names), p=self._motion_weights)

    def _frame_coordinates(self, motion_index: jax.Array, time_s: jax.Array):
        length = self._motion["lengths"][motion_index]
        loop = self._motion["loops"][motion_index]
        frame = jp.maximum(time_s, 0.0) * self._motion_fps
        frame = jp.where(loop, jp.mod(frame, jp.maximum(length - 1, 1)), jp.minimum(frame, length - 1))
        left = jp.floor(frame).astype(jp.int32)
        right = jp.minimum(left + 1, length - 1)
        return left, right, frame - left

    def _sample_field(self, field: str, motion_index: jax.Array, time_s: jax.Array) -> jax.Array:
        left, right, alpha = self._frame_coordinates(motion_index, time_s)
        values = self._motion[field]
        left_value = values[motion_index, left]
        right_value = values[motion_index, right]
        while alpha.ndim < left_value.ndim:
            alpha = alpha[..., None]
        return (1.0 - alpha) * left_value + alpha * right_value

    def _reference_vector(self, info: dict[str, Any], offset_s: jax.Array = jp.asarray(0.0)) -> jax.Array:
        time_s = info["motion_time"] + offset_s
        motion_index = info["motion_index"]
        reference = jp.concatenate(
            [
                self._sample_field("joint_position", motion_index, time_s),
                jp.atleast_1d(self._sample_field("root_height", motion_index, time_s)),
                self._sample_field("projected_gravity", motion_index, time_s),
                self._sample_field("local_linear_velocity", motion_index, time_s),
                self._sample_field("local_angular_velocity", motion_index, time_s),
                self._sample_field("foot_contacts", motion_index, time_s),
            ]
        )
        blend_alpha = jp.clip((info["blend_time"] + offset_s) / TRANSITION_SECONDS, 0.0, 1.0)
        return (1.0 - blend_alpha) * info["blend_from"] + blend_alpha * reference

    def _reference_window(self, info: dict[str, Any]) -> jax.Array:
        return jp.concatenate([self._reference_vector(info, offset) for offset in REFERENCE_OFFSETS])

    def _reference_joint_velocity(self, info: dict[str, Any]) -> jax.Array:
        # Differentiate the same blended position reference seen by the actor,
        # including transition segments.
        current = self._reference_vector(info)[:14]
        following = self._reference_vector(info, jp.asarray(self.dt))[:14]
        return (following - current) / self.dt

    def _ensure_motion_info(self, info: dict[str, Any]) -> None:
        """Seed reference fields during the base class's bootstrap reset."""

        if "motion_index" in info:
            return
        motion_index = jp.asarray(0, dtype=jp.int32)
        motion_time = jp.asarray(0.0)
        reference = self._raw_reference(motion_index, motion_time)
        info.update(
            motion_index=motion_index,
            motion_time=motion_time,
            motion_age=jp.asarray(0.0),
            blend_time=jp.asarray(TRANSITION_SECONDS),
            blend_from=reference,
        )

    def _command_and_phase(self, info: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
        reference = self._reference_vector(info)
        command = jp.concatenate([reference[18:21], reference[5:9]])
        length = self._motion["lengths"][info["motion_index"]]
        frame = jp.mod(info["motion_time"] * self._motion_fps, jp.maximum(length - 1, 1))
        phase = frame / jp.maximum(length - 1, 1)
        features = jp.asarray([jp.cos(2 * jp.pi * phase), jp.sin(2 * jp.pi * phase)])
        return command, features

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        rng, motion_rng, phase_rng, rsi_rng, noise_rng, bias_rng = jax.random.split(
            state.info["rng"], 6
        )
        motion_index = self._choose_motion(motion_rng)
        length = self._motion["lengths"][motion_index]
        kind = self._motion["kinds"][motion_index]
        random_phase = jax.random.uniform(phase_rng) * (length - 1) / self._motion_fps
        use_rsi = (kind == 1) | ((kind == 2) & jax.random.bernoulli(rsi_rng, self._config.reference_state_init_probability))
        motion_time = jp.where(use_rsi, random_phase, 0.0)
        first_reference = self._raw_reference(motion_index, motion_time)
        state.info.update(
            {
                "rng": rng,
                "motion_index": motion_index,
                "motion_time": motion_time,
                "motion_age": jp.asarray(0.0),
                "blend_time": jp.asarray(TRANSITION_SECONDS),
                "blend_from": first_reference,
                "encoder_bias": jax.random.uniform(
                    bias_rng,
                    (14,),
                    minval=-self._config.encoder_bias_scale,
                    maxval=self._config.encoder_bias_scale,
                ),
            }
        )

        # Persona-style reference-state initialization, limited to upright v1 clips.
        qpos = state.data.qpos
        qvel = state.data.qvel
        target_joint = first_reference[:14]
        target_joint += jax.random.uniform(noise_rng, (14,), minval=-0.02, maxval=0.02)
        qpos = self.set_actuator_joints_qpos(jp.where(use_rsi, target_joint, self.get_actuator_joints_qpos(qpos)), qpos)
        qpos = qpos.at[self._floating_base_qpos_addr + 2].set(
            jp.where(use_rsi, first_reference[14], qpos[self._floating_base_qpos_addr + 2])
        )
        qvel = self.set_actuator_joints_qvel(
            jp.where(use_rsi, self._sample_field("joint_velocity", motion_index, motion_time), self.get_actuator_joints_qvel(qvel)),
            qvel,
        )
        ctrl = self.get_actuator_joints_qpos(qpos)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        state.info["motor_targets"] = ctrl
        command, phase = self._command_and_phase(state.info)
        state.info["command"] = command
        state.info["imitation_phase"] = phase
        contact = self._contacts(data)
        obs = self._get_obs(data, state.info, contact)
        return state.replace(data=data, obs=obs)

    def _raw_reference(self, motion_index: jax.Array, time_s: jax.Array) -> jax.Array:
        return jp.concatenate(
            [
                self._sample_field("joint_position", motion_index, time_s),
                jp.atleast_1d(self._sample_field("root_height", motion_index, time_s)),
                self._sample_field("projected_gravity", motion_index, time_s),
                self._sample_field("local_linear_velocity", motion_index, time_s),
                self._sample_field("local_angular_velocity", motion_index, time_s),
                self._sample_field("foot_contacts", motion_index, time_s),
            ]
        )

    def _contacts(self, data: mjx.Data) -> jax.Array:
        from mujoco_playground._src.collision import geoms_colliding

        return jp.asarray([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        current_reference = self._reference_vector(state.info)
        state.info["motion_time"] += self.dt
        state.info["motion_age"] += self.dt
        state.info["blend_time"] += self.dt
        length = self._motion["lengths"][state.info["motion_index"]]
        loop = self._motion["loops"][state.info["motion_index"]]
        duration = (length - 1) / self._motion_fps
        switch_after = jp.where(loop, self._config.motion_switch_seconds, jp.minimum(duration + 0.4, self._config.motion_switch_seconds))
        should_switch = state.info["motion_age"] >= switch_after
        state.info["rng"], switch_rng = jax.random.split(state.info["rng"])
        next_index = self._choose_motion(switch_rng)
        state.info["motion_index"] = jp.where(should_switch, next_index, state.info["motion_index"])
        state.info["motion_time"] = jp.where(should_switch, 0.0, state.info["motion_time"])
        state.info["motion_age"] = jp.where(should_switch, 0.0, state.info["motion_age"])
        state.info["blend_time"] = jp.where(should_switch, 0.0, state.info["blend_time"])
        state.info["blend_from"] = jp.where(should_switch, current_reference, state.info["blend_from"])
        command, phase = self._command_and_phase(state.info)
        state.info["command"] = command
        state.info["imitation_phase"] = phase
        result = super().step(state, action)
        return result.replace(reward=self._signed_reward(result.metrics, result.done) * self.dt)

    def _signed_reward(self, metrics: dict[str, jax.Array], done: jax.Array) -> jax.Array:
        """Recombine already-scaled reward/cost metrics without zero clipping."""

        signed_total = jp.asarray(0.0)
        for name, scale in self._config.reward_config.scales.items():
            if name == "termination" or scale == 0:
                continue
            key = ("reward/" if scale > 0 else "cost/") + name
            # Joystick.step has already applied each configured scale before
            # placing the magnitude in reward/* or cost/*. Recombine those
            # logged values with their sign, without applying scales twice.
            signed_total += metrics[key] if scale > 0 else -metrics[key]
        signed_total += self._config.reward_config.scales.termination * done
        return signed_total

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array) -> mjx_env.Observation:
        self._ensure_motion_info(info)
        observation = super()._get_obs(data, info, contact)
        prefix = observation["state"].at[13:27].add(
            info.get("encoder_bias", jp.zeros(14))
        )
        actor = jp.concatenate([prefix, self._reference_window(info)])
        reference = self._reference_vector(info)
        joint_position = self.get_actuator_joints_qpos(data.qpos)
        joint_velocity = self.get_actuator_joints_qvel(data.qvel)
        reference_errors = jp.concatenate(
            [
                joint_position - reference[:14],
                joint_velocity - self._reference_joint_velocity(info),
                jp.atleast_1d(data.qpos[self._floating_base_qpos_addr + 2] - reference[14]),
                # The MuJoCo ``upvector`` sensor is the IMU frame's positive
                # Z axis in world coordinates.  Motion references use the same
                # body-up convention, so no gravity-style sign flip belongs
                # here despite the legacy get_gravity() method name.
                self.get_gravity(data) - reference[15:18],
                self.get_local_linvel(data) - reference[18:21],
                self.get_global_angvel(data) - reference[21:24],
                contact.astype(jp.float32) - reference[24:26],
            ]
        )
        privileged = jp.concatenate([actor, observation["privileged_state"], reference_errors])
        return {"state": actor, "privileged_state": privileged}

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics, done, first_contact
        reference = self._reference_vector(info)
        joint_position = self.get_actuator_joints_qpos(data.qpos)
        joint_velocity = self.get_actuator_joints_qvel(data.qvel)
        target_velocity = self._reference_joint_velocity(info)
        feet_velocity = data.sensordata[self._foot_linvel_sensor_adr][..., :2]
        action_second_difference = jp.mean(jp.square(action - 2 * info["last_act"] + info["last_last_act"]))
        lower_violation = jp.maximum(self._soft_lowers - joint_position, 0.0)
        upper_violation = jp.maximum(joint_position - self._soft_uppers, 0.0)
        is_stand = self._motion["kinds"][info["motion_index"]] == 0
        return {
            "alive": reward_alive(),
            "reference_joint_position": jp.exp(-10.0 * jp.mean(jp.square(joint_position - reference[:14]))),
            "reference_joint_velocity": jp.exp(-0.05 * jp.mean(jp.square(joint_velocity - target_velocity))),
            "reference_orientation": jp.exp(-8.0 * jp.mean(jp.square(self.get_gravity(data) - reference[15:18]))),
            "reference_height": jp.exp(-jp.square(data.qpos[self._floating_base_qpos_addr + 2] - reference[14]) / 0.0025),
            "reference_linear_velocity": jp.exp(-8.0 * jp.mean(jp.square(self.get_local_linvel(data) - reference[18:21]))),
            "reference_angular_velocity": jp.exp(-2.0 * jp.mean(jp.square(self.get_global_angvel(data) - reference[21:24]))),
            "reference_contacts": jp.mean((contact == (reference[24:26] > 0.5)).astype(jp.float32)),
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "action_second_difference": action_second_difference,
            "foot_slip": jp.sum(jp.square(feet_velocity) * contact[:, None]),
            "soft_joint_limits": jp.sum(jp.square(lower_violation) + jp.square(upper_violation)),
            "standing_drift": jp.where(is_stand, jp.sum(jp.square(self.get_local_linvel(data)[:2])), 0.0),
        }


def np_kind_weights(kinds, train_mask, mix):
    """Create exact category probabilities without importing NumPy in JIT paths."""

    import numpy as np

    weights = np.zeros(len(kinds), dtype=np.float32)
    for kind, probability in enumerate(mix):
        indices = np.flatnonzero((kinds == kind) & train_mask)
        if len(indices):
            weights[indices] = float(probability) / len(indices)
    total = weights.sum()
    if total <= 0:
        raise ValueError("motion bundle contains no train-split clips")
    return weights / total
