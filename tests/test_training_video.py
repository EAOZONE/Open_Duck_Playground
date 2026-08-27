"""Fast scheduling checks for periodic training videos."""

from pathlib import Path
import tempfile
import unittest

from playground.common.training_video import TrainingVideoRecorder


class DummyEnv:
    motion_names = ("stand", "head_nod")


class TrainingVideoTest(unittest.TestCase):
    def test_first_callback_records_then_respects_step_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrainingVideoRecorder(
                DummyEnv(), directory, interval_steps=5_000_000,
                rollout_steps=10, motions=("stand",),
            )
            self.assertTrue(recorder.should_record(5_000_000))
            recorder.last_recorded_step = 5_000_000
            self.assertFalse(recorder.should_record(9_999_999))
            self.assertTrue(recorder.should_record(10_000_000))
            self.assertEqual(recorder.output_dir, Path(directory) / "videos")

    def test_unknown_showcase_motion_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                TrainingVideoRecorder(
                    DummyEnv(), directory, interval_steps=1,
                    rollout_steps=1, motions=("not_a_motion",),
                )


if __name__ == "__main__":
    unittest.main()
