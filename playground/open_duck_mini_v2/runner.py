"""Runs training and evaluation loop for Open Duck Mini V2."""

import argparse
import os

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import getup, joystick, jump, standing


GETUP_ABLATIONS = ("balanced", "corrective", "combined")


def configure_goal_only_getup(config, variant: str, *, training: bool):
    """Applies a reproducible goal-only reward/reset ablation."""
    if variant not in GETUP_ABLATIONS:
        raise ValueError(f"Unknown get-up ablation {variant}")
    config.use_reference_motion = False
    config.use_worst_foot_flatness = True
    config.use_leg_reposition_cost = variant in ("corrective", "combined")
    config.goal_only_reset_mix = (
        [0.5, 0.25, 0.25] if training and variant == "combined" else [1.0, 0.0, 0.0]
    )
    return config


class OpenDuckMiniV2Runner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "getup": (getup, getup.GetUp),
            "joystick": (joystick, joystick.Joystick),
            "jump": (jump, jump.Jump),
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        if args.env == "getup":
            use_reference_motion = not args.goal_only_getup
            getup_ablation = getattr(args, "getup_ablation", "combined")
            training_config = getup.default_config()
            if use_reference_motion:
                training_config.use_reference_motion = True
                training_config.reference_state_init_probability = 0.7
                training_config.goal_only_reset_mix = [1.0, 0.0, 0.0]
            else:
                configure_goal_only_getup(
                    training_config, getup_ablation, training=True
                )
                training_config.reference_state_init_probability = 0.0
            self.env = getup.GetUp(task=args.task, config=training_config)
            # Evaluation is deliberately the harder, requested condition:
            # every episode starts face-down at phase zero.
            eval_config = getup.default_config()
            if use_reference_motion:
                eval_config.use_reference_motion = True
                eval_config.goal_only_reset_mix = [1.0, 0.0, 0.0]
            else:
                configure_goal_only_getup(eval_config, getup_ablation, training=False)
            self.eval_env = getup.GetUp(task=args.task, config=eval_config)
            mode = "reference-guided" if use_reference_motion else "goal-only"
            suffix = "" if use_reference_motion else f" ({getup_ablation})"
            print(f"Get-up training mode: {mode}{suffix}")
        else:
            self.env = self.env_file[1](task=args.task)
            self.eval_env = self.env_file[1](task=args.task)
        # Learn the delicate landing correction in nominal dynamics first.
        # Walking keeps its existing randomized training path unchanged.
        self.randomizer = (
            None if args.env in ("jump", "getup") else randomize.domain_randomize
        )
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override PPO learning rate.",
    )
    parser.add_argument(
        "--entropy-cost",
        type=float,
        default=None,
        help="Override PPO entropy cost.",
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
    parser.add_argument(
        "--playful-walk",
        action="store_true",
        help=(
            "Fine-tune a 45 mm swing-height walk with one alternating 65 mm "
            "accent step every three gait cycles."
        ),
    )
    parser.add_argument(
        "--goal-only-getup",
        action="store_true",
        help=(
            "Train get-up directly from face-down to standing without a "
            "reference animation, phase signal, or moving pose target."
        ),
    )
    parser.add_argument(
        "--getup-ablation",
        choices=GETUP_ABLATIONS,
        default="combined",
        help=(
            "Goal-only get-up experiment: worst-foot reward only, add the "
            "corrective leg cost, or add both the cost and reset curriculum."
        ),
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
    os.environ["OPEN_DUCK_PLAYFUL_WALK"] = "1" if args.playful_walk else "0"

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
