"""Regression tests for the face-down get-up task and reference motion."""

import unittest

import mujoco
import numpy as np

try:
    import jax

    from playground.open_duck_mini_v2.getup import (
        LEFT_FOOT_STUCK_QPOS,
        RIGHT_FOOT_STUCK_QPOS,
        STANDING_FOOT_NORMAL_MIN,
        GetUp,
        default_config,
        foot_flatness_scores,
        leg_reposition_cost,
    )
    from playground.open_duck_mini_v2.runner import configure_goal_only_getup

    HAS_JAX = True
except ModuleNotFoundError:
    HAS_JAX = False
from playground.open_duck_mini_v2.getup_motion import (
    FACE_DOWN_QUAT,
    GETUP_DURATION,
    PRONE_ROOT_HEIGHT,
    STANDING_ROOT_HEIGHT,
    GetUpMotionController,
    clip_pose,
    phase_features,
    root_trajectory,
    trajectory_pose,
)


class GetUpMotionTest(unittest.TestCase):
    def test_one_shot_controller(self) -> None:
        controller = GetUpMotionController()
        self.assertFalse(controller.active)
        self.assertTrue(controller.request_getup())
        self.assertFalse(controller.request_getup())
        controller.advance(GETUP_DURATION)
        self.assertFalse(controller.active)
        self.assertTrue(controller.request_getup())

    def test_reference_starts_prone_and_finishes_standing(self) -> None:
        start_pos, start_quat = root_trajectory(0.0)
        end_pos, end_quat = root_trajectory(GETUP_DURATION)
        np.testing.assert_allclose(start_pos, [0.0, 0.0, PRONE_ROOT_HEIGHT])
        np.testing.assert_allclose(start_quat, FACE_DOWN_QUAT)
        np.testing.assert_allclose(end_pos, [0.0, 0.0, STANDING_ROOT_HEIGHT])
        np.testing.assert_allclose(end_quat, [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(start_quat)), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(end_quat)), 1.0, places=6)

    def test_pose_and_phase_are_finite_and_within_model_limits(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )
        home = model.keyframe("home").ctrl.copy()
        lower = model.actuator_ctrlrange[:, 0]
        upper = model.actuator_ctrlrange[:, 1]
        for time_s in np.linspace(0.0, GETUP_DURATION, 81):
            pose = clip_pose(trajectory_pose(time_s, home), lower, upper)
            root_pos, root_quat = root_trajectory(time_s)
            features = phase_features(time_s)
            self.assertEqual(pose.shape, (14,))
            self.assertEqual(root_pos.shape, (3,))
            self.assertEqual(root_quat.shape, (4,))
            self.assertEqual(features.shape, (3,))
            self.assertTrue(np.isfinite(pose).all())
            self.assertTrue(np.isfinite(root_pos).all())
            self.assertTrue(np.isfinite(root_quat).all())
            np.testing.assert_array_less(lower - 1.0e-6, pose)
            np.testing.assert_array_less(pose, upper + 1.0e-6)
            self.assertAlmostEqual(float(np.linalg.norm(root_quat)), 1.0, places=5)
        np.testing.assert_allclose(trajectory_pose(GETUP_DURATION, home), home)
        self.assertEqual(phase_features(GETUP_DURATION)[0], 0.0)

    def test_reference_moves_legs_slowly_without_a_kick(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )
        home = model.keyframe("home").ctrl.copy()
        times = np.linspace(0.0, GETUP_DURATION, 701)
        poses = np.stack([trajectory_pose(time_s, home) for time_s in times])
        joint_speed = np.diff(poses, axis=0) / np.diff(times)[:, None]
        self.assertLess(float(np.max(np.abs(joint_speed))), 1.2)
        self.assertGreater(float(np.min(poses[:, 3])), 0.5)
        self.assertGreater(float(np.min(poses[:, 12])), 0.5)
        # Both yaw joints steer the knees to the same side and both knees flex
        # together instead of forming the old left/right scissor pose.
        self.assertLess(float(np.max(np.abs(poses[:, 0] - poses[:, 9]))), 0.01)
        self.assertLess(float(np.max(np.abs(poses[:, 3] - poses[:, 12]))), 0.02)

    def test_fall_collision_proxies_are_present(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )
        for name in (
            "trunk_collision",
            "head_collision",
            "left_thigh_collision",
            "left_shin_collision",
            "right_thigh_collision",
            "right_shin_collision",
        ):
            geom = model.geom(name)
            self.assertGreaterEqual(geom.id, 0)
            self.assertNotEqual(model.geom_contype[geom.id], 0)

    def test_reference_states_do_not_start_buried_in_the_floor(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )
        data = mujoco.MjData(model)
        actuator_qpos_addresses = np.array(
            [
                model.jnt_qposadr[model.actuator_trnid[index, 0]]
                for index in range(model.nu)
            ]
        )
        home = model.keyframe("home")
        for time_s in np.linspace(0.0, GETUP_DURATION, 141):
            root_pos, root_quat = root_trajectory(time_s)
            data.qpos[:] = home.qpos
            data.qpos[:3] = root_pos
            data.qpos[3:7] = root_quat
            data.qpos[actuator_qpos_addresses] = trajectory_pose(time_s, home.ctrl)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if data.ncon:
                self.assertGreaterEqual(
                    float(np.min(data.contact.dist[: data.ncon])), -0.002
                )

    def test_prone_quaternion_points_robot_forward_axis_down(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )
        data = mujoco.MjData(model)
        data.qpos[:] = model.keyframe("home").qpos
        data.qpos[3:7] = FACE_DOWN_QUAT
        mujoco.mj_forward(model, data)
        sensor = model.sensor("forwardvector")
        address = int(model.sensor_adr[sensor.id])
        dimension = int(model.sensor_dim[sensor.id])
        forward = data.sensordata[address : address + dimension]
        self.assertLess(float(forward[2]), -0.99)

    def test_standing_pose_has_flat_foot_frames(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )
        data = mujoco.MjData(model)
        data.qpos[:] = model.keyframe("home").qpos
        mujoco.mj_forward(model, data)
        for name in ("left_foot", "right_foot"):
            site_id = model.site(name).id
            # The site's local +Z axis is the sole normal.  Its world-Z
            # component is one for a perfectly flat, right-side-up foot.
            sole_normal_z = data.site_xmat[site_id].reshape(3, 3)[2, 2]
            self.assertGreater(float(sole_normal_z), 0.99)

    @unittest.skipUnless(HAS_JAX, "JAX is provided by the training container")
    def test_worst_foot_flatness_cannot_be_hidden_by_flat_foot(self) -> None:
        both_flat = np.asarray(foot_flatness_scores(np.array([1.0, 1.0])))
        one_tilted = np.asarray(foot_flatness_scores(np.array([1.0, 0.5])))
        self.assertAlmostEqual(float(np.min(both_flat)), 1.0)
        self.assertLess(float(np.min(one_tilted)), 0.6)
        self.assertGreater(float(np.mean(one_tilted)), float(np.min(one_tilted)))

    @unittest.skipUnless(HAS_JAX, "JAX is provided by the training container")
    def test_leg_reposition_cost_vanishes_for_either_safe_exit(self) -> None:
        target_knee = np.array([1.37, 1.38])
        target_hip = np.array([-0.63, 0.64])
        stuck_cost = float(
            leg_reposition_cost(
                np.array([1.0, 0.53]),
                np.array([1.37, 0.93]),
                target_knee,
                np.array([-0.63, 1.10]),
                target_hip,
            )
        )
        flat_cost = float(
            leg_reposition_cost(
                np.array([1.0, 1.0]),
                np.array([1.37, 0.93]),
                target_knee,
                np.array([-0.63, 1.10]),
                target_hip,
            )
        )
        repositioned_cost = float(
            leg_reposition_cost(
                np.array([1.0, 0.53]),
                target_knee,
                target_knee,
                target_hip,
                target_hip,
            )
        )
        self.assertGreater(stuck_cost, 0.1)
        self.assertAlmostEqual(flat_cost, 0.0)
        self.assertAlmostEqual(repositioned_cost, 0.0)

    @unittest.skipUnless(HAS_JAX, "JAX is provided by the training container")
    def test_stuck_curriculum_anchors_are_valid_and_mirrored(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )
        data = mujoco.MjData(model)
        expected_tilted = ("right_foot", "left_foot")
        for qpos, tilted_name in zip(
            (RIGHT_FOOT_STUCK_QPOS, LEFT_FOOT_STUCK_QPOS),
            expected_tilted,
        ):
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            data.ctrl[:] = qpos[7:]
            mujoco.mj_forward(model, data)
            self.assertTrue(np.isfinite(data.qpos).all())
            if data.ncon:
                self.assertGreaterEqual(
                    float(np.min(data.contact.dist[: data.ncon])), -0.001
                )
            tilted_normal = data.site_xmat[model.site(tilted_name).id].reshape(3, 3)[
                2, 2
            ]
            other_name = "left_foot" if tilted_name == "right_foot" else "right_foot"
            flat_normal = data.site_xmat[model.site(other_name).id].reshape(3, 3)[2, 2]
            self.assertLess(float(tilted_normal), STANDING_FOOT_NORMAL_MIN)
            self.assertGreater(float(flat_normal), 0.99)

    @unittest.skipUnless(HAS_JAX, "JAX is provided by the training container")
    def test_goal_only_reset_modes_preserve_actor_contract(self) -> None:
        for reset_mix, expected_mode in (
            ([1.0, 0.0, 0.0], 0),
            ([0.0, 1.0, 0.0], 1),
            ([0.0, 0.0, 1.0], 2),
        ):
            config = default_config()
            config.use_reference_motion = False
            config.goal_only_reset_mix = reset_mix
            env = GetUp(config=config)
            state = env.reset(jax.random.PRNGKey(expected_mode))
            self.assertEqual(state.obs["state"].shape, (104,))
            np.testing.assert_array_equal(
                np.asarray(state.info["jump_phase"]), np.zeros(3)
            )
            self.assertEqual(int(state.info["reset_mode"]), expected_mode)
            self.assertTrue(np.isfinite(np.asarray(state.data.qpos)).all())

    @unittest.skipUnless(HAS_JAX, "JAX is provided by the training container")
    def test_goal_only_ablation_profiles_are_reproducible(self) -> None:
        balanced = configure_goal_only_getup(
            default_config(), "balanced", training=True
        )
        corrective = configure_goal_only_getup(
            default_config(), "corrective", training=True
        )
        combined = configure_goal_only_getup(
            default_config(), "combined", training=True
        )
        self.assertFalse(balanced.use_leg_reposition_cost)
        self.assertTrue(corrective.use_leg_reposition_cost)
        self.assertEqual(corrective.goal_only_reset_mix, [1.0, 0.0, 0.0])
        self.assertEqual(combined.goal_only_reset_mix, [0.5, 0.25, 0.25])

    @unittest.skipUnless(HAS_JAX, "JAX is provided by the training container")
    def test_dense_standing_reward_does_not_hit_clip(self) -> None:
        config = default_config()
        dense_positive_ceiling = sum(
            scale
            for name, scale in config.reward_config.scales.items()
            if scale > 0.0 and name != "standing"
        )
        self.assertGreater(config.reward_clip, dense_positive_ceiling)
        self.assertLess(
            config.reward_clip,
            dense_positive_ceiling + config.reward_config.scales.standing,
        )


if __name__ == "__main__":
    unittest.main()
