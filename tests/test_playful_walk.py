"""Tests for the playful-walk cadence and inference cue."""

import unittest

import numpy as np

from playground.open_duck_mini_v2 import playful_walk
from playground.open_duck_mini_v2.mujoco_infer import MjInfer


class PlayfulWalkCadenceTest(unittest.TestCase):
    def test_one_alternating_accent_every_three_cycles(self) -> None:
        sides = [
            int(playful_walk.skip_side_for_cycle(cycle, np)) for cycle in range(12)
        ]
        self.assertEqual(sides, [-1, -1, 0, -1, -1, 1, -1, -1, 0, -1, -1, 1])

    def test_only_accented_foot_gets_higher_target(self) -> None:
        np.testing.assert_allclose(
            playful_walk.desired_foot_heights(2, np), [0.065, 0.045]
        )
        np.testing.assert_allclose(
            playful_walk.desired_foot_heights(5, np), [0.045, 0.065]
        )

    def test_inference_cue_uses_existing_observation_slot(self) -> None:
        infer = MjInfer.__new__(MjInfer)
        infer.playful_policy = True
        infer.playful_mode = True
        infer.gait_cycle = 2
        infer.commands = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        infer._update_playful_policy_cue(moving=True)
        self.assertEqual(infer.commands[playful_walk.SKIP_COMMAND_INDEX], 0.5)
        infer._update_playful_policy_cue(moving=False)
        self.assertEqual(infer.commands[playful_walk.SKIP_COMMAND_INDEX], 0.0)

    def test_normal_mode_suppresses_the_accent_cue(self) -> None:
        infer = MjInfer.__new__(MjInfer)
        infer.playful_policy = True
        infer.playful_mode = False
        infer.gait_cycle = 2
        infer.commands = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        infer._update_playful_policy_cue(moving=True)
        self.assertEqual(infer.commands[playful_walk.SKIP_COMMAND_INDEX], 0.0)


if __name__ == "__main__":
    unittest.main()
