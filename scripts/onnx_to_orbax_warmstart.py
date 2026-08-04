#!/usr/bin/env python3
"""Reconstruct a Brax PPO warm-start checkpoint from an exported actor ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from flax.training import orbax_utils
import jax
import jax.numpy as jp
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime
from orbax import checkpoint as ocp

# Orbax 0.11.24 calls this newer JAX monitoring hook. TensorFlow currently
# installs the same no-op compatibility hook when the normal runner imports
# the ONNX exporter, but this standalone converter intentionally avoids the
# TensorFlow startup cost.
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *args, **kwargs: None


def _initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def extract_actor(
    onnx_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Extract observation statistics and Flax-compatible dense weights."""
    model = onnx.load(onnx_path.as_posix())
    arrays = _initializers(model)

    mean_matches = [
        value for name, value in arrays.items() if "/sub/ReadVariableOp" in name
    ]
    reciprocal_std_matches = [
        value for name, value in arrays.items() if "truediv_recip" in name
    ]
    if len(mean_matches) != 1 or len(reciprocal_std_matches) != 1:
        raise ValueError("Could not uniquely identify ONNX observation mean/std")

    layers: dict[str, dict[str, np.ndarray]] = {}
    for name, value in arrays.items():
        match = re.search(r"hidden_(\d+)", name)
        if match is None:
            continue
        layer_name = f"hidden_{int(match.group(1))}"
        if "Cast/ReadVariableOp" in name:
            layers.setdefault(layer_name, {})["kernel"] = value
        elif "BiasAdd/ReadVariableOp" in name:
            layers.setdefault(layer_name, {})["bias"] = value

    if not layers or any(
        set(values) != {"kernel", "bias"} for values in layers.values()
    ):
        raise ValueError("ONNX actor does not contain complete dense-layer weights")

    mean = np.asarray(mean_matches[0], dtype=np.float32)
    reciprocal_std = np.asarray(reciprocal_std_matches[0], dtype=np.float32)
    if np.any(reciprocal_std <= 0.0):
        raise ValueError("ONNX observation reciprocal std must be positive")
    return mean, 1.0 / reciprocal_std, layers


def build_checkpoint_params(
    mean: np.ndarray,
    std: np.ndarray,
    layers: dict[str, dict[str, np.ndarray]],
    *,
    action_size: int,
    privileged_observation_size: int,
    statistics_count: float,
):
    """Build the normalizer, recovered actor, and a fresh critic."""
    state_size = int(mean.shape[0])
    hidden_names = sorted(layers, key=lambda name: int(name.rsplit("_", 1)[1]))
    hidden_sizes = tuple(
        int(layers[name]["bias"].shape[0]) for name in hidden_names[:-1]
    )
    expected_output = action_size * 2
    if int(layers[hidden_names[-1]]["bias"].shape[0]) != expected_output:
        raise ValueError(
            f"Final ONNX layer has {layers[hidden_names[-1]]['bias'].shape[0]} outputs; "
            f"expected {expected_output} for {action_size} actions"
        )

    observation_sizes = {
        "state": (state_size,),
        "privileged_state": (privileged_observation_size,),
    }
    ppo_network = ppo_networks.make_ppo_networks(
        observation_sizes,
        action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    policy_params = ppo_network.policy_network.init(jax.random.PRNGKey(0))
    value_params = ppo_network.value_network.init(jax.random.PRNGKey(1))

    for name in hidden_names:
        for parameter in ("kernel", "bias"):
            recovered = jp.asarray(layers[name][parameter], dtype=jp.float32)
            expected_shape = policy_params["params"][name][parameter].shape
            if recovered.shape != expected_shape:
                raise ValueError(
                    f"{name}/{parameter} shape {recovered.shape} does not match "
                    f"current PPO shape {expected_shape}"
                )
            policy_params["params"][name][parameter] = recovered

    count = jp.asarray(statistics_count, dtype=jp.float32)
    normalizer = running_statistics.RunningStatisticsState(
        count=count,
        mean={
            "state": jp.asarray(mean),
            "privileged_state": jp.zeros(privileged_observation_size),
        },
        summed_variance={
            "state": jp.square(jp.asarray(std)) * count,
            "privileged_state": jp.ones(privileged_observation_size) * count,
        },
        std={
            "state": jp.asarray(std),
            "privileged_state": jp.ones(privileged_observation_size),
        },
    )
    params = ppo_losses.PPONetworkParams(policy=policy_params, value=value_params)
    return normalizer, params, ppo_network


def verify_actor(
    onnx_path: Path,
    normalizer,
    params,
    ppo_network,
) -> float:
    """Return max absolute deterministic-action error against ONNX Runtime."""
    session = onnxruntime.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )
    observations = np.stack(
        [
            np.zeros_like(np.asarray(normalizer.mean["state"])),
            np.ones_like(np.asarray(normalizer.mean["state"])),
            np.asarray(normalizer.mean["state"]),
        ]
    ).astype(np.float32)
    errors = []
    for observation in observations:
        onnx_action = session.run(None, {"obs": observation[None]})[0][0]
        network_observation = {
            "state": jp.asarray(observation)[None],
            "privileged_state": jp.zeros(
                (1, normalizer.mean["privileged_state"].shape[0])
            ),
        }
        logits = ppo_network.policy_network.apply(
            normalizer, params.policy, network_observation
        )
        brax_action = np.asarray(
            ppo_network.parametric_action_distribution.mode(logits)
        )[0]
        errors.append(float(np.max(np.abs(onnx_action - brax_action))))
    return max(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-size", type=int, default=14)
    parser.add_argument("--privileged-observation-size", type=int, default=212)
    parser.add_argument(
        "--statistics-count",
        type=float,
        default=150_000_000.0,
        help="Synthetic normalizer count used to preserve the recovered mean/std.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    mean, std, layers = extract_actor(onnx_path)
    normalizer, params, ppo_network = build_checkpoint_params(
        mean,
        std,
        layers,
        action_size=args.action_size,
        privileged_observation_size=args.privileged_observation_size,
        statistics_count=args.statistics_count,
    )
    max_error = verify_actor(onnx_path, normalizer, params, ppo_network)
    if max_error > 1e-5:
        raise RuntimeError(f"Recovered actor differs from ONNX by {max_error:.3e}")

    output.parent.mkdir(parents=True, exist_ok=True)
    target = normalizer, params
    save_args = orbax_utils.save_args_from_target(target)
    ocp.PyTreeCheckpointer().save(output.as_posix(), target, save_args=save_args)
    print(f"Saved warm-start checkpoint: {output}")
    print(f"Recovered actor max absolute error: {max_error:.3e}")
    print(
        "Critic is freshly initialized; PPO will learn it while preserving "
        "the recovered actor and observation normalization."
    )


if __name__ == "__main__":
    main()
