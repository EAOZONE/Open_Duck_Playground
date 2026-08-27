"""Fast interface checks for the unified motion-tracking environment."""

from pathlib import Path
import unittest

import jax
import jax.numpy as jp
import numpy as np

from playground.common.motion_bundle import MotionBundle
from playground.open_duck_mini_v2.motion_tracking import MotionTracking, default_config


class MotionTrackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = default_config()
        config.noise_config.level = 0.0
        config.push_config.enable = False
        cls.env = MotionTracking(config=config)
        cls.state = cls.env.reset(jax.random.PRNGKey(7))

    def test_actor_and_critic_contracts_are_separate(self) -> None:
        self.assertEqual(self.state.obs["state"].shape, (205,))
        self.assertGreater(self.state.obs["privileged_state"].shape[0], 205)
        self.assertEqual(self.env.action_size, 14)

    def test_legacy_prefix_and_reference_window_have_stable_widths(self) -> None:
        actor = np.asarray(self.state.obs["state"])
        self.assertEqual(actor[:101].shape, (101,))
        self.assertEqual(actor[101:].shape, (104,))
        np.testing.assert_allclose(
            actor[101:], np.asarray(self.env._reference_window(self.state.info)), atol=1e-6
        )

    def test_physics_step_keeps_signed_reward_and_shape(self) -> None:
        result = self.env.step(self.state, jp.zeros(14))
        self.assertEqual(result.obs["state"].shape, (205,))
        self.assertTrue(np.isfinite(float(result.reward)))
        self.assertIn("cost/action_second_difference", result.metrics)

    def test_orientation_reward_and_critic_use_positive_body_up_vector(self) -> None:
        reference = np.asarray(self.env._reference_vector(self.state.info))
        actual_up = np.asarray(self.env.get_gravity(self.state.data))
        contact = self.env._contacts(self.state.data)
        rewards = self.env._get_reward(
            self.state.data,
            jp.zeros(14),
            self.state.info,
            {},
            jp.asarray(0.0),
            jp.zeros(2, dtype=bool),
            contact,
        )
        expected_reward = np.exp(-8.0 * np.mean(np.square(actual_up - reference[15:18])))
        self.assertAlmostEqual(
            float(rewards["reference_orientation"]), float(expected_reward), places=6
        )

        observation = self.env._get_obs(self.state.data, self.state.info, contact)
        reference_errors = np.asarray(observation["privileged_state"])[-40:]
        np.testing.assert_allclose(
            reference_errors[29:32], actual_up - reference[15:18], atol=1.0e-6
        )

    def test_signed_reward_subtracts_logged_costs_without_rescaling(self) -> None:
        metrics = {}
        for name, scale in self.env._config.reward_config.scales.items():
            if name == "termination" or scale == 0:
                continue
            metrics[("reward/" if scale > 0 else "cost/") + name] = jp.asarray(0.0)
        metrics["reward/alive"] = jp.asarray(3.0)
        metrics["cost/action_rate"] = jp.asarray(2.0)
        self.assertAlmostEqual(float(self.env._signed_reward(metrics, jp.asarray(0.0))), 1.0)

    def test_numpy_runtime_and_jax_reference_sampling_match(self) -> None:
        name = "head_nod"
        index = self.env.motion_names.index(name)
        info = dict(self.state.info)
        info["motion_index"] = jp.asarray(index)
        info["motion_time"] = jp.asarray(0.51)
        info["blend_time"] = jp.asarray(0.4)
        info["blend_from"] = self.env._raw_reference(info["motion_index"], info["motion_time"])
        bundle = MotionBundle.load(
            Path(__file__).parents[1]
            / "playground/open_duck_mini_v2/data/motion_bundle_v1.npz"
        )
        np.testing.assert_allclose(
            np.asarray(self.env._reference_window(info)),
            bundle.get(name).reference_window(0.51),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
