#!/usr/bin/env python3
"""Run the unified policy's six-case randomized evaluation matrix."""

from __future__ import annotations

import argparse
import json

from evaluate_motion_tracking import SCENARIO_TASK, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output")
    args = parser.parse_args()
    results = {
        scenario: evaluate(args.onnx, args.episodes, args.steps, scenario)
        for scenario in SCENARIO_TASK
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
