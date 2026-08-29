"""Fixed-seed benchmark for the bounded 50 Hz local planning path."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from firi.planning.plannerv2 import FIRIPlanner
from main import generate_random_endpoints
from obstacle_generator import place_obstacles


BOUNDS = np.array([[0.0, 0.0, 0.0], [6.0, 20.0, 4.0]])
BOUNDARY = [[0.0, 6.0], [0.0, 20.0], [0.0, 4.0]]


def timing_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def run_density(density, seeds, safety_margin, budget):
    runs = []
    for seed in range(seeds):
        np.random.seed(seed)
        start, goal = generate_random_endpoints(BOUNDS, 1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            stage_start = time.perf_counter()
            obstacles = place_obstacles(
                BOUNDARY, start, goal, 3, 2, 3, density, 2
            )
            obstacle_ms = (time.perf_counter() - stage_start) * 1000.0
            stage_start = time.perf_counter()
            planner = FIRIPlanner(
                obstacles, tuple(BOUNDS[1] - BOUNDS[0]), BOUNDS
            )
            planner.set_safety_margin(safety_margin)
            initialization_ms = (time.perf_counter() - stage_start) * 1000.0

        path = planner.plan_realtime(
            start,
            goal,
            reference_path=np.vstack([start, goal]),
            time_budget=budget,
            max_repairs=1,
        )
        stats = dict(planner.last_realtime_stats)
        verified_safe = path is not None and not planner.check_path_safety(path)
        runs.append(
            {
                "seed": seed,
                "obstacle_generation_ms": obstacle_ms,
                "planner_initialization_ms": initialization_ms,
                "planning_ms": stats["elapsed_seconds"] * 1000.0,
                "success": stats["success"],
                "verified_safe": verified_safe,
                "deadline_met": stats["deadline_met"],
                "repairs": stats["repairs"],
                "remaining_collisions": stats["remaining_collisions"],
            }
        )
    return {
        "density": density,
        "seeds": seeds,
        "safety_margin_m": safety_margin,
        "budget_ms": budget * 1000.0,
        "success_count": sum(run["success"] for run in runs),
        "verified_safe_count": sum(run["verified_safe"] for run in runs),
        "deadline_met_count": sum(run["deadline_met"] for run in runs),
        "planning_timing": timing_summary(
            [run["planning_ms"] for run in runs]
        ),
        "failed_seeds": [run["seed"] for run in runs if not run["success"]],
        "deadline_miss_seeds": [
            run["seed"] for run in runs if not run["deadline_met"]
        ],
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--densities", nargs="+", default=["medium", "high"])
    parser.add_argument("--safety-margin", type=float, default=0.3)
    parser.add_argument("--budget-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "method": "bounded spherical local guide with exact 3-D verification",
        "scope": "per-cycle planning; Python import and visualization excluded",
        "results": {
            density: run_density(
                density,
                args.seeds,
                args.safety_margin,
                args.budget_ms / 1000.0,
            )
            for density in args.densities
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for density, result in payload["results"].items():
        print(
            density,
            {
                "success": f"{result['success_count']}/{result['seeds']}",
                "verified_safe": (
                    f"{result['verified_safe_count']}/{result['seeds']}"
                ),
                "deadline": f"{result['deadline_met_count']}/{result['seeds']}",
                **result["planning_timing"],
                "failed_seeds": result["failed_seeds"],
            },
        )


if __name__ == "__main__":
    main()
