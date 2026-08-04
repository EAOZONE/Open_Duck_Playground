"""Compatibility checks for the dedicated standing policy environment."""

import unittest

import jax
import numpy as np

from playground.open_duck_mini_v2.standing import Standing


class StandingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = Standing()
        cls.state = cls.env.reset(jax.random.PRNGKey(0))

    def test_actor_observation_matches_walking_policy_interface(self) -> None:
        self.assertEqual(self.state.obs["state"].shape, (101,))
        self.assertEqual(self.env.action_size, 14)

    def test_reset_uses_stationary_command_and_home_motor_targets(self) -> None:
        np.testing.assert_allclose(self.state.info["command"], 0.0)
        np.testing.assert_allclose(self.state.info["imitation_phase"], 0.0)
        np.testing.assert_allclose(
            self.state.info["motor_targets"], self.env._default_actuator
        )

    def test_stationary_rewards_penalize_drift_and_foot_motion(self) -> None:
        self.assertIn("cost/planar_velocity", self.state.metrics)
        self.assertIn("cost/feet_motion", self.state.metrics)


if __name__ == "__main__":
    unittest.main()
