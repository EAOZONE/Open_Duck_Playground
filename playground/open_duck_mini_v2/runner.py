"""Runs training and evaluation loop for Open Duck Mini V2."""

import argparse
import os

# The training runner is headless. EGL lets periodic MuJoCo videos use the
# NVIDIA renderer without requiring a desktop window.
os.environ.setdefault("MUJOCO_GL", "egl")

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import getup, joystick, jump, motion_tracking, standing


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
            "motion_tracking": (motion_tracking, motion_tracking.MotionTracking),
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        if args.env == "getup":
            back_getup = getattr(args, "back_getup", False)
            use_reference_motion = not args.goal_only_getup and not back_getup
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
                if back_getup:
                    training_config.fallen_orientation = "back"
                    back_stage = getattr(args, "back_getup_stage", "foundation")
                    training_config.back_getup_reset_mix = (
                        [0.0, 0.55, 0.30, 0.15]
                        if back_stage == "foundation"
                        else [0.55, 0.25, 0.15, 0.05]
                    )
            self.env = getup.GetUp(task=args.task, config=training_config)
            # Evaluation is deliberately the harder, requested condition:
            # every episode starts face-down at phase zero.
            eval_config = getup.default_config()
            if use_reference_motion:
                eval_config.use_reference_motion = True
                eval_config.goal_only_reset_mix = [1.0, 0.0, 0.0]
            else:
                configure_goal_only_getup(eval_config, getup_ablation, training=False)
                if back_getup:
                    eval_config.fallen_orientation = "back"
                    eval_config.back_getup_reset_mix = [1.0, 0.0, 0.0, 0.0]
            self.eval_env = getup.GetUp(task=args.task, config=eval_config)
            mode = "reference-guided" if use_reference_motion else "goal-only"
            if back_getup:
                mode += f" back-start/{args.back_getup_stage}"
            suffix = "" if use_reference_motion else f" ({getup_ablation})"
            print(f"Get-up training mode: {mode}{suffix}")
        elif args.env == "motion_tracking":
            training_config = motion_tracking.default_config()
            if args.unified_stage == "locomotion":
                training_config.motion_mix = [0.4, 0.6, 0.0]
                training_config.push_config.enable = False
                training_config.noise_config.level = 0.0
                training_config.noise_config.action_max_delay = 1
                training_config.noise_config.imu_max_delay = 1
            elif args.unified_stage == "mixed":
                training_config.push_config.enable = False
                training_config.noise_config.level = 0.0
                training_config.noise_config.action_max_delay = 1
                training_config.noise_config.imu_max_delay = 1
            else:
                training_config.encoder_bias_scale = 0.02
            self.env = motion_tracking.MotionTracking(task=args.task, config=training_config)
            self.eval_env = motion_tracking.MotionTracking(task=args.task, config=training_config)
            if args.video:
                video_config = motion_tracking.default_config()
                video_config.noise_config.level = 0.0
                video_config.noise_config.action_max_delay = 1
                video_config.noise_config.imu_max_delay = 1
                video_config.push_config.enable = False
                video_config.encoder_bias_scale = 0.0
                self.video_env = motion_tracking.MotionTracking(
                    task=args.task, config=video_config
                )
        else:
            self.env = self.env_file[1](task=args.task)
            self.eval_env = self.env_file[1](task=args.task)
        # Learn the delicate landing correction in nominal dynamics first.
        # Walking keeps its existing randomized training path unchanged.
        self.randomizer = (
            None
            if args.env in ("jump", "getup")
            or (args.env == "motion_tracking" and args.unified_stage != "sim2real")
            else randomize.domain_randomize
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
        "--target-total-timesteps",
        type=int,
        default=None,
        help=(
            "On resume, train only enough additional steps for the checkpoint "
            "suffix plus new steps to reach this global total."
        ),
    )
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
        "--video",
        action="store_true",
        help="Record a deterministic motion showcase at training evaluation callbacks.",
    )
    parser.add_argument(
        "--video-interval-steps",
        type=int,
        default=5_000_000,
        help="Minimum environment steps between showcase videos.",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=150,
        help="Frames per showcase motion (150 frames is 3 seconds at 50 Hz).",
    )
    parser.add_argument(
        "--video-motions",
        nargs="+",
        default=("stand", "walk_0.148_0.037_-0.074", "head_nod", "bow"),
        help="Fixed named clips included in every comparable showcase.",
    )
    parser.add_argument("--video-width", type=int, default=480)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--video-camera", default=None)
    parser.add_argument(
        "--video-strict",
        action="store_true",
        help="Stop training if a requested video cannot be rendered or encoded.",
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
        "--motion-bundle",
        type=str,
        default=None,
        help="Compiled open_duck.motion_bundle.v1 NPZ for unified motion tracking.",
    )
    parser.add_argument(
        "--unified-stage",
        choices=("locomotion", "mixed", "sim2real"),
        default="mixed",
        help="Unified policy curriculum stage.",
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
        "--back-getup",
        action="store_true",
        help=(
            "Fine-tune a goal-only get-up policy from randomized supine/back "
            "resets while retaining late-stage standing curriculum states."
        ),
    )
    parser.add_argument(
        "--back-getup-stage",
        choices=("foundation", "full"),
        default="foundation",
        help=(
            "Back-get-up curriculum phase: learn side/crouch/stand first, "
            "then introduce complete back-down recoveries."
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
    if args.motion_bundle:
        os.environ["OPEN_DUCK_MOTION_BUNDLE"] = args.motion_bundle
    os.environ["OPEN_DUCK_USE_IMITATION_REWARD"] = (
        "0" if args.no_imitation_reward else "1"
    )
    os.environ["OPEN_DUCK_HIGH_CLEARANCE"] = "1" if args.high_clearance else "0"
    os.environ["OPEN_DUCK_PLAYFUL_WALK"] = "1" if args.playful_walk else "0"

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
