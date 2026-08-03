"""Runs training and evaluation loop for Open Duck Mini V2."""

import argparse
import os

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import joystick, jump, standing


class OpenDuckMiniV2Runner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
            "jump": (jump, jump.Jump),
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task)
        self.eval_env = self.env_file[1](task=args.task)
        # Learn the delicate landing correction in nominal dynamics first.
        # Walking keeps its existing randomized training path unchanged.
        self.randomizer = None if args.env == "jump" else randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    # parser.add_argument("--num_timesteps", type=int, default=300000000)
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override PPO parallel environment count (use a small value for CPU smoke tests).",
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=None,
        help="Override PPO evaluation count.",
    )
    parser.add_argument("--env", type=str, default="joystick", help="env")
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    parser.add_argument(
        "--reference_motion",
        type=str,
        default=None,
        help="Polynomial reference pickle. Omit to use the checked-in default.",
    )
    parser.add_argument(
        "--no-imitation-reward",
        action="store_true",
        help="Train a policy using task rewards only; do not load reference motion data.",
    )
    parser.add_argument(
        "--high-clearance",
        action="store_true",
        help="Train a higher-foot-lift walking gait with commands up to 0.20 m/s.",
    )
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    if args.reference_motion:
        os.environ["OPEN_DUCK_REFERENCE_MOTION"] = args.reference_motion
    os.environ["OPEN_DUCK_USE_IMITATION_REWARD"] = (
        "0" if args.no_imitation_reward else "1"
    )
    os.environ["OPEN_DUCK_HIGH_CLEARANCE"] = "1" if args.high_clearance else "0"

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
