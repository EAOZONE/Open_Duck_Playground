"""Checkpoint-path safety checks for the common training runner."""

from pathlib import Path
import tempfile
import unittest

from playground.common.runner import BaseRunner


class RunnerPathTest(unittest.TestCase):
    def test_relative_checkpoint_is_resolved_for_orbax(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            relative = Path(directory).relative_to(Path.cwd())
            self.assertEqual(
                BaseRunner.resolve_checkpoint_path(relative),
                Path(directory).resolve().as_posix(),
            )

    def test_missing_checkpoint_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            BaseRunner.resolve_checkpoint_path("checkpoint_that_does_not_exist")

    def test_checkpoint_step_is_parsed_for_resumed_logs(self) -> None:
        self.assertEqual(
            BaseRunner.checkpoint_step("2026_08_27_130731_30474240"), 30_474_240
        )
        self.assertEqual(BaseRunner.checkpoint_step("unified_walk_warmstart"), 0)


if __name__ == "__main__":
    unittest.main()
