"""Regression tests for the walking-to-hop controller handoff."""

import unittest
from types import SimpleNamespace

import numpy as np

from playground.open_duck_mini_v2.jump_motion import JumpMotionController
from playground.open_duck_mini_v2.mujoco_infer import MjInfer


class CombinedMotionTest(unittest.TestCase):
    @staticmethod
    def _infer() -> MjInfer:
        infer = MjInfer.__new__(MjInfer)
        infer.hop = JumpMotionController()
        infer.hop_requested = False
        infer.motion_mode = "walking"
        infer.settle_elapsed = 0.0
        infer.commands = [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0]
        infer._latched_forward_command = 0.2
        infer._hop_button_was_pressed = False
        infer.default_actuator = np.linspace(-0.2, 0.2, 14, dtype=np.float32)
        infer.prev_motor_targets = np.ones(14, dtype=np.float32)
        infer.last_action = np.ones(14, dtype=np.float32)
        infer.last_last_action = np.ones(14, dtype=np.float32)
        infer.last_last_last_action = np.ones(14, dtype=np.float32)
        infer.model = SimpleNamespace(
            actuator_ctrlrange=np.column_stack(
                [
                    np.full(14, -2.0, dtype=np.float32),
                    np.full(14, 2.0, dtype=np.float32),
                ]
            )
        )
        return infer

    def test_keyboard_hop_works_while_xbox_input_is_enabled(self) -> None:
        infer = self._infer()
        infer.use_xbox_controller = True
        infer.key_callback(74)
        self.assertTrue(infer.hop_requested)

    def test_xbox_hop_uses_a_rising_button_edge(self) -> None:
        infer = self._infer()

        infer._update_hop_button([0, 1])
        self.assertTrue(infer.hop_requested)

        infer.hop_requested = False
        infer._update_hop_button([0, 1])
        self.assertFalse(infer.hop_requested)

        infer._update_hop_button([0, 0])
        infer._update_hop_button([0, 1])
        self.assertTrue(infer.hop_requested)

    def test_walk_settle_hop_recover_walk_sequence(self) -> None:
        infer = self._infer()
        walking_target = np.full(14, 0.75, dtype=np.float32)

        self.assertTrue(infer.request_hop())
        infer._prepare_motion_step(0.1)
        self.assertEqual(infer.motion_mode, "settling")
        np.testing.assert_allclose(infer.commands[:3], 0.0)
        self.assertEqual(infer._latched_forward_command, 0.0)
        np.testing.assert_array_equal(
            infer._target_for_motion_mode(walking_target, 0.1), walking_target
        )

        infer._prepare_motion_step(infer.HOP_SETTLE_DURATION)
        self.assertEqual(infer.motion_mode, "hopping")
        self.assertTrue(infer.hop.active)

        # The hop owns the target during this mode, even if a walking target
        # is accidentally supplied by a caller.
        infer._target_for_motion_mode(walking_target, 0.2)
        hop_target = infer._target_for_motion_mode(walking_target, 0.2)
        self.assertFalse(np.allclose(hop_target, walking_target))

        while infer.motion_mode == "hopping":
            infer._target_for_motion_mode(None, 0.2)
        self.assertEqual(infer.motion_mode, "recovering")
        np.testing.assert_array_equal(
            infer._target_for_motion_mode(None, 0.0), infer.default_actuator
        )

        infer._prepare_motion_step(infer.HOP_RECOVERY_DURATION)
        self.assertEqual(infer.motion_mode, "walking")
        np.testing.assert_array_equal(infer.last_action, 0.0)
        np.testing.assert_array_equal(infer.last_last_action, 0.0)
        np.testing.assert_array_equal(infer.last_last_last_action, 0.0)
        np.testing.assert_array_equal(
            infer.prev_motor_targets, infer.default_actuator
        )

    def test_repeated_requests_are_rejected_until_recovery_finishes(self) -> None:
        infer = self._infer()
        self.assertTrue(infer.request_hop())
        self.assertFalse(infer.request_hop())
        infer._prepare_motion_step(0.1)
        self.assertFalse(infer.request_hop())


if __name__ == "__main__":
    unittest.main()
