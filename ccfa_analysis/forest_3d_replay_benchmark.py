"""Replay the realtime layer on saved full-safe 3-D forest references."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from firi.planning.plannerv2 import FIRIPlanner  # noqa: E402
from firi.planning.spherical_projection import (  # noqa: E402
    SphericalProjectionConfig,
    SphericalProjectionGuide,
)
from forest_simulator.generate_forest_scenarios import (  # noqa: E402
    BOUNDS,
    SAFETY_MARGIN,
    TIME_BUDGET,
    add_tree_obstacles,
    avoidance_direction,
)
from obstacle_generator import ObstacleSet  # noqa: E402


def timing_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def summarize(runs):
    return {
        "scenarios": len(runs),
        "success_count": sum(run["success"] for run in runs),
        "safe_count": sum(run["safe"] for run in runs),
        "deadline_met_count": sum(run["deadline_met"] for run in runs),
        "unsafe_output_count": sum(run["unsafe_output"] for run in runs),
        "timing": timing_summary([run["elapsed_ms"] for run in runs]),
        "avoidance_direction_counts": {
            direction: sum(run["avoidance_direction"] == direction for run in runs)
            for direction in ("left", "right", "up", "down")
        },
        "failed_seeds": [run["seed"] for run in runs if not run["success"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=17)
    parser.add_argument("--height", type=int, default=13)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    scenarios = [
        scenario
        for scenario in source["scenarios"]
        if scenario["metrics"]["full_safe"]
    ]
    runs = []
    for scenario in scenarios:
        obstacles = ObstacleSet()
        for tree in scenario["trees"]:
            add_tree_obstacles(obstacles, tree)
        with contextlib.redirect_stdout(io.StringIO()):
            planner = FIRIPlanner(
                obstacles, tuple(BOUNDS[1] - BOUNDS[0]), BOUNDS
            )
            planner.set_safety_margin(SAFETY_MARGIN)
        planner.realtime_guide = SphericalProjectionGuide(
            obstacles,
            SphericalProjectionConfig(
                width=args.width,
                height=args.height,
                far=8.0,
                safety_radius=SAFETY_MARGIN,
                max_candidates=16,
            ),
        )
        reference = np.asarray(scenario["reference_path"], dtype=float)
        segment_index = scenario["injected_segment"]
        planner.check_path_safety(reference)
        planner.realtime_guide.render_obstacles(
            reference[segment_index],
            reference[segment_index + 1] - reference[segment_index],
        )
        path = planner.plan_realtime(
            reference[0],
            reference[-1],
            reference_path=reference,
            time_budget=TIME_BUDGET,
            max_repairs=1,
        )
        stats = planner.last_realtime_stats
        safe = bool(path is not None and not planner.check_path_safety(path))
        run = {
            "id": scenario["id"],
            "density": scenario["density"],
            "seed": scenario["seed"],
            "elapsed_ms": stats["elapsed_seconds"] * 1000.0,
            "deadline_met": stats["deadline_met"],
            "success": stats["success"] and safe,
            "safe": safe,
            "unsafe_output": path is not None and not safe,
            "avoidance_direction": avoidance_direction(
                reference, path, segment_index
            ),
        }
        runs.append(run)
        print(
            f"{run['id']}: success={run['success']} "
            f"time={run['elapsed_ms']:.2f} ms"
        )

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": str(args.input),
            "condition": "realtime replay conditioned on full_safe reference paths",
            "projection_resolution": [args.width, args.height],
            "time_budget_ms": TIME_BUDGET * 1000.0,
            "safety_margin_m": SAFETY_MARGIN,
        },
        "overall": summarize(runs),
        "summaries": {
            density: summarize(
                [run for run in runs if run["density"] == density]
            )
            for density in ("sparse", "medium", "dense")
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"Wrote {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
