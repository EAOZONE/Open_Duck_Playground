"""Fast regression tests for the Open Duck jump interfaces."""

import unittest

import mujoco
import numpy as np

from playground.open_duck_mini_v2.jump_motion import (
    JUMP_DURATION,
    JumpMotionController,
    clip_pose,
    phase_features,
    trajectory_pose,
)


class JumpMotionTest(unittest.TestCase):
    def test_one_shot_controller(self) -> None:
        controller = JumpMotionController()
        self.assertFalse(controller.active)
        self.assertTrue(controller.request_jump())
        self.assertFalse(controller.request_jump())
        controller.advance(JUMP_DURATION)
        self.assertFalse(controller.active)
        self.assertTrue(controller.request_jump())

    def test_phase_features_and_pose_are_finite(self) -> None:
        home = np.linspace(-0.2, 0.2, 14, dtype=np.float32)
        for time_s in np.linspace(0.0, JUMP_DURATION, 41):
            pose = trajectory_pose(time_s, home)
            features = phase_features(time_s)
            self.assertEqual(pose.shape, (14,))
            self.assertEqual(features.shape, (3,))
            self.assertTrue(np.isfinite(pose).all())
            self.assertTrue(np.isfinite(features).all())
        self.assertEqual(phase_features(JUMP_DURATION)[0], 0.0)

    def test_model_limits_and_head_hold(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )
        home = model.keyframe("home").ctrl.copy()
        lower = model.actuator_ctrlrange[:, 0]
        upper = model.actuator_ctrlrange[:, 1]
        for time_s in np.linspace(0.0, JUMP_DURATION, 41):
            pose = clip_pose(trajectory_pose(time_s, home), lower, upper)
            np.testing.assert_array_less(lower - 1e-6, pose)
            np.testing.assert_array_less(pose, upper + 1e-6)
            np.testing.assert_allclose(pose[5:9], home[5:9])


if __name__ == "__main__":
    unittest.main()
