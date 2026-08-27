#!/usr/bin/env python3
"""Verify a unified ONNX actor against its saved JAX/Orbax actor."""

from __future__ import annotations

import argparse
from pathlib import Path

from orbax import checkpoint as ocp

from onnx_to_orbax_warmstart import (
    build_checkpoint_params,
    extract_actor,
    verify_actor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--privileged-observation-size", type=int, default=417)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()

    mean, std, layers = extract_actor(args.onnx)
    normalizer, params, network = build_checkpoint_params(
        mean,
        std,
        layers,
        action_size=14,
        privileged_observation_size=args.privileged_observation_size,
        statistics_count=1.0,
    )
    normalizer, params = ocp.PyTreeCheckpointer().restore(
        args.checkpoint.resolve().as_posix(), item=(normalizer, params)
    )
    error = verify_actor(args.onnx.resolve(), normalizer, params, network)
    print(f"JAX/ONNX maximum absolute action error: {error:.3e}")
    if error > args.tolerance:
        raise SystemExit(
            f"parity failure: {error:.3e} exceeds tolerance {args.tolerance:.3e}"
        )


if __name__ == "__main__":
    main()
