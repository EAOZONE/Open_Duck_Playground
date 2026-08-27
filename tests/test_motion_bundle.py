"""Regression tests for the versioned direct-frame motion contract."""

from pathlib import Path
import unittest

import numpy as np

from playground.common.motion_bundle import (
    BUNDLE_SCHEMA,
    MotionBundle,
    UNIFIED_OBSERVATION_SIZE,
    make_policy_manifest,
    validate_policy_manifest,
)


BUNDLE = Path(__file__).parents[1] / "playground/open_duck_mini_v2/data/motion_bundle_v1.npz"


class MotionBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = MotionBundle.load(BUNDLE)

    def test_bundle_has_training_and_holdout_upright_motions(self) -> None:
        self.assertEqual(self.bundle.manifest["schema"], BUNDLE_SCHEMA)
        self.assertGreaterEqual(len(self.bundle.clips), 12)
        self.assertIn("stand", self.bundle.by_name)
        self.assertTrue(any(clip.kind == "locomotion" for clip in self.bundle.clips))
        self.assertTrue(any(clip.split == "holdout" for clip in self.bundle.clips))
        self.assertTrue(any("__mirror" in clip.name for clip in self.bundle.clips))
        self.assertTrue(any("__tempo" in clip.name for clip in self.bundle.clips))
        self.assertTrue(any("__amplitude" in clip.name for clip in self.bundle.clips))
        self.assertLessEqual(
            max(float(np.max(np.abs(clip.joint_velocity))) for clip in self.bundle.clips),
            4.0 + 1e-4,
        )
        for clip in self.bundle.clips:
            np.testing.assert_allclose(
                np.linalg.norm(clip.projected_gravity, axis=1), 1.0, atol=1.0e-3
            )

    def test_reference_window_is_four_times_26(self) -> None:
        clip = self.bundle.get("head_nod")
        self.assertEqual(clip.reference_vector(0.5).shape, (26,))
        self.assertEqual(clip.reference_window(0.5).shape, (104,))
        self.assertEqual(clip.phase_features(0.5).shape, (2,))
        np.testing.assert_allclose(clip.sample(0.0)["foot_contacts"], [1.0, 1.0])

    def test_policy_manifest_locks_the_205_input_contract(self) -> None:
        manifest = make_policy_manifest()
        self.assertEqual(manifest["observation_size"], UNIFIED_OBSERVATION_SIZE)
        self.assertEqual(manifest["observation_layout"][-1]["start"], 101)
        validate_policy_manifest(manifest)
        manifest["joint_names"] = list(reversed(manifest["joint_names"]))
        with self.assertRaises(ValueError):
            validate_policy_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
