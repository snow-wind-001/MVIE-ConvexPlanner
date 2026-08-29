"""Reproducible, headless runtime and safety benchmark for MVIE-ConvexPlanner.

Run from an isolated working directory with the repository on PYTHONPATH.  The
script deliberately does not call any plotting code and does not modify the
planner implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import time
import warnings
from pathlib import Path

import numpy as np

from firi.planning.plannerv2 import FIRIPlanner
from main import fix_path_collisions, generate_random_endpoints
from obstacle_generator import place_obstacles
from path_planner import calculate_path_length


SPACE_BOUNDS = np.array([[0.0, 0.0, 0.0], [6.0, 20.0, 4.0]])


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 95),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def primitive_clearance(point: np.ndarray, obstacle) -> float:
    center = np.asarray(obstacle.center, dtype=float)
    if obstacle.shape == "sphere":
        return float(np.linalg.norm(point - center) - obstacle.radius)
    if obstacle.shape == "cylinder":
        radial = np.linalg.norm(point[:2] - center[:2]) - obstacle.radius
        axial = abs(point[2] - center[2]) - obstacle.height / 2
        outside = np.linalg.norm(np.maximum([radial, axial], 0.0))
        inside = min(max(radial, axial), 0.0)
        return float(outside + inside)
    if obstacle.shape == "cuboid":
        delta = np.abs(point - center) - np.asarray(obstacle.size) / 2
        outside = np.linalg.norm(np.maximum(delta, 0.0))
        inside = min(float(np.max(delta)), 0.0)
        return float(outside + inside)
    return float("inf")


def path_min_clearance(path: np.ndarray, obstacles, step: float = 0.02) -> float:
    minimum = float("inf")
    for p1, p2 in zip(path[:-1], path[1:]):
        length = float(np.linalg.norm(p2 - p1))
        count = max(2, int(np.ceil(length / step)) + 1)
        for t in np.linspace(0.0, 1.0, count):
            point = p1 * (1.0 - t) + p2 * t
            minimum = min(
                minimum,
                *(primitive_clearance(point, obs) for obs in obstacles),
            )
    return float(minimum)


def run_seed(
    seed: int,
    density: str,
    safety_margin: float = 0.3,
    use_spherical_guidance: bool = True,
) -> dict:
    np.random.seed(seed)
    timings: dict[str, float] = {}
    captured = io.StringIO()
    warning_messages: list[str] = []

    start_point, goal_point = generate_random_endpoints(SPACE_BOUNDS, margin=1.0)
    space_boundary = [[SPACE_BOUNDS[0][i], SPACE_BOUNDS[1][i]] for i in range(3)]

    with contextlib.redirect_stdout(captured), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        obstacles = place_obstacles(
            space_boundary,
            start_point,
            goal_point,
            n_spheres=3,
            n_cylinders=2,
            n_cuboids=3,
            density=density,
            num_on_path=2,
        )
        timings["obstacle_generation"] = time.perf_counter() - started

        started = time.perf_counter()
        planner = FIRIPlanner(
            obstacles=obstacles,
            space_size=tuple(SPACE_BOUNDS[1] - SPACE_BOUNDS[0]),
            space_bounds=SPACE_BOUNDS,
        )
        timings["planner_initialization"] = time.perf_counter() - started

        started = time.perf_counter()
        nominal_path = np.asarray(
            planner.plan_path(
                start_point,
                goal_point,
                initial_waypoints=None,
                smoothing=True,
                max_replanning_attempts=7,
                safety_margin=safety_margin,
                use_spherical_guidance=use_spherical_guidance,
            ),
            dtype=float,
        )
        timings["nominal_planning"] = time.perf_counter() - started

        nominal_collision_segments = planner.check_path_safety(nominal_path)
        stored_collision_segments = list(planner.path_collisions)

        started = time.perf_counter()
        if nominal_collision_segments:
            final_path = fix_path_collisions(
                nominal_path, planner, SPACE_BOUNDS, max_rounds=8
            )
        else:
            final_path = nominal_path.copy()
        timings["collision_repair"] = time.perf_counter() - started
        timings["core_total"] = sum(timings.values())

        final_collision_segments = planner.check_path_safety(final_path)
        warning_messages = [str(item.message) for item in caught]

    log = captured.getvalue()
    nominal_length = float(calculate_path_length(nominal_path))
    final_length = float(calculate_path_length(final_path))

    return {
        "seed": seed,
        "density": density,
        "timings_seconds": timings,
        "obstacle_count": len(obstacles.obstacle_list),
        "nominal_points": len(nominal_path),
        "final_points": len(final_path),
        "nominal_collision_count": len(nominal_collision_segments),
        "final_collision_count": len(final_collision_segments),
        "planner_stored_collision_count": len(stored_collision_segments),
        "stored_collision_state_stale": stored_collision_segments
        != nominal_collision_segments,
        "slsqp_converged": "轨迹优化收敛 (" in log,
        "slsqp_failed": "轨迹优化未完全收敛:" in log,
        "fallback_used": "启用启发式重规划" in log,
        "bspline_accepted": "B-spline平滑成功且安全" in log,
        "corridor_failure_count": log.count("走廊") and log.count("计算失败"),
        "nominal_path_length": nominal_length,
        "final_path_length": final_length,
        "repair_length_ratio": final_length / nominal_length,
        "nominal_min_clearance": path_min_clearance(nominal_path, obstacles),
        "final_min_clearance": path_min_clearance(final_path, obstacles),
        "inside_bounds": bool(
            np.all(final_path >= SPACE_BOUNDS[0] - 1e-12)
            and np.all(final_path <= SPACE_BOUNDS[1] + 1e-12)
        ),
        "warnings": warning_messages,
        "exception": None,
        "log_tail": log.splitlines()[-16:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--density", default="medium")
    parser.add_argument("--safety-margin", type=float, default=0.3)
    parser.add_argument("--no-guidance", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    Path("temp").mkdir(exist_ok=True)
    runs = []
    for seed in range(args.seeds):
        started = time.perf_counter()
        try:
            result = run_seed(
                seed,
                args.density,
                safety_margin=args.safety_margin,
                use_spherical_guidance=not args.no_guidance,
            )
        except Exception as exc:  # keep the batch running and expose failures
            result = {
                "seed": seed,
                "density": args.density,
                "exception": f"{type(exc).__name__}: {exc}",
            }
        runs.append(result)
        elapsed = time.perf_counter() - started
        print(
            f"seed={seed} elapsed={elapsed:.3f}s "
            f"exception={result.get('exception')} "
            f"nominal_col={result.get('nominal_collision_count')} "
            f"final_col={result.get('final_collision_count')}",
            flush=True,
        )

    successful = [run for run in runs if run.get("exception") is None]
    timing_keys = [
        "obstacle_generation",
        "planner_initialization",
        "nominal_planning",
        "collision_repair",
        "core_total",
    ]
    summary = {
        "run_count": len(runs),
        "completed_count": len(successful),
        "exception_count": len(runs) - len(successful),
        "timings_seconds": {
            key: summarize([run["timings_seconds"][key] for run in successful])
            for key in timing_keys
        },
        "nominal_collision_run_count": sum(
            run["nominal_collision_count"] > 0 for run in successful
        ),
        "final_collision_run_count": sum(
            run["final_collision_count"] > 0 for run in successful
        ),
        "slsqp_converged_count": sum(run["slsqp_converged"] for run in successful),
        "slsqp_failed_count": sum(run["slsqp_failed"] for run in successful),
        "fallback_used_count": sum(run["fallback_used"] for run in successful),
        "stale_collision_state_count": sum(
            run["stored_collision_state_stale"] for run in successful
        ),
        "obstacle_count_values": sorted(
            {run["obstacle_count"] for run in successful}
        ),
        "final_min_clearance": summarize(
            [run["final_min_clearance"] for run in successful]
        ),
        "repair_length_ratio": summarize(
            [run["repair_length_ratio"] for run in successful]
        ),
    }
    payload = {
        "configuration": {
            "seeds": list(range(args.seeds)),
            "density": args.density,
            "space_bounds": SPACE_BOUNDS.tolist(),
            "obstacles_requested": {"sphere": 3, "cylinder": 2, "cuboid": 3},
            "num_on_path": 2,
            "planner_safety_margin_argument": args.safety_margin,
            "collision_threshold_used_by_planner": args.safety_margin,
            "use_spherical_guidance": not args.no_guidance,
        },
        "summary": summary,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
