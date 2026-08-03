"""Regression tests for the phase-locked excited walk overlay."""

import unittest

import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer


class ExpressiveWalkTest(unittest.TestCase):
    @staticmethod
    def _infer(phase: tuple[float, float], command: float = 0.20) -> MjInfer:
        infer = MjInfer.__new__(MjInfer)
        infer.expressive_walk = True
        infer.standing = False
        infer.policy_only = False
        infer.head_bobble_amplitude = MjInfer.DEFAULT_HEAD_BOBBLE_AMPLITUDE
        infer.antenna_wiggle_amplitude = MjInfer.DEFAULT_ANTENNA_WIGGLE_AMPLITUDE
        infer.step_bounce_amplitude = MjInfer.DEFAULT_STEP_BOUNCE_AMPLITUDE
        infer.imitation_phase = np.asarray(phase, dtype=np.float32)
        infer.commands = [command, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        infer.motor_targets = np.zeros(14, dtype=np.float32)
        return infer

    def test_full_walk_moves_head_and_visually_wiggles_antennae(self) -> None:
        phases = [
            (np.cos(theta), np.sin(theta))
            for theta in np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
        ]
        targets = []
        for phase in phases:
            infer = self._infer(phase)
            infer._apply_expressive_style()
            targets.append(infer.motor_targets)
        targets = np.asarray(targets)

        self.assertGreater(np.ptp(targets[:, 6]), 0.10)  # head pitch
        self.assertGreater(np.ptp(targets[:, 7]), 0.04)  # head yaw
        self.assertGreater(np.ptp(targets[:, 8]), 0.10)  # head roll/antennae
        self.assertGreater(np.max(np.abs(targets[:, 3])), 0.01)  # knee spring
        np.testing.assert_allclose(targets[:, 3], targets[:, 12])

    def test_overlay_stops_with_zero_movement_command(self) -> None:
        infer = self._infer((0.0, 1.0), command=0.0)
        before = infer.motor_targets.copy()
        infer._apply_expressive_style()
        np.testing.assert_array_equal(infer.motor_targets, before)

    def test_overlay_is_bounded_by_configured_amplitudes(self) -> None:
        for theta in np.linspace(0.0, 2.0 * np.pi, 33):
            infer = self._infer((np.cos(theta), np.sin(theta)))
            infer._apply_expressive_style()
            self.assertLessEqual(
                abs(float(infer.motor_targets[6])),
                infer.head_bobble_amplitude + 1e-6,
            )
            self.assertLessEqual(
                abs(float(infer.motor_targets[8])),
                infer.antenna_wiggle_amplitude + 1e-6,
            )
            self.assertLessEqual(
                abs(float(infer.motor_targets[3])),
                infer.step_bounce_amplitude + 1e-6,
            )


if __name__ == "__main__":
    unittest.main()
