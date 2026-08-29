"""Generate reproducible forest-flight evidence for the Three.js simulator.

The browser is deliberately only a renderer.  Every path and safety metric in
the emitted JSON is produced by the repository's real ``FIRIPlanner``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from firi.planning.plannerv2 import FIRIPlanner  # noqa: E402
from obstacle_generator import ObstacleSet  # noqa: E402


BOUNDS = np.array([[-6.0, 0.0, 0.4], [6.0, 36.0, 11.5]])
SAFETY_MARGIN = 0.30
TIME_BUDGET = 0.020
DENSITIES = {
    "sparse": {"label": "稀疏林", "trees": 12, "minimum_spacing": 1.35},
    "medium": {"label": "中密林", "trees": 20, "minimum_spacing": 1.15},
    "dense": {"label": "密集林", "trees": 28, "minimum_spacing": 1.00},
}


def path_length(path: np.ndarray | None) -> float | None:
    if path is None or len(path) < 2:
        return None
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def resample_polyline(path: np.ndarray, maximum_spacing: float = 4.0) -> np.ndarray:
    """Densify a global path for the bounded local reference-path API."""
    result = [np.asarray(path[0], dtype=float)]
    for start, end in zip(path[:-1], path[1:]):
        length = float(np.linalg.norm(end - start))
        pieces = max(1, int(np.ceil(length / maximum_spacing)))
        result.extend(start + (end - start) * (index / pieces) for index in range(1, pieces + 1))
    return np.asarray(result)


def add_tree_obstacles(obstacles: ObstacleSet, tree: dict) -> None:
    """Add exactly the geometry encoded in a renderer-facing tree record."""
    obstacles.add_obstacle(
        "cylinder",
        np.asarray(tree["center"], dtype=float),
        radius=tree["radius"],
        height=tree["height"],
    )
    for branch in tree["branches"]:
        obstacles.add_obstacle(
            "capsule",
            radius=branch["radius"],
            start=np.asarray(branch["start"], dtype=float),
            end=np.asarray(branch["end"], dtype=float),
        )
    for crown in tree["canopy_spheres"]:
        obstacles.add_obstacle(
            "sphere",
            np.asarray(crown["center"], dtype=float),
            radius=crown["radius"],
        )


def make_tree(
    rng: np.random.Generator,
    xy: np.ndarray,
    radius: float,
    tree_height: float,
    *,
    direct_blocker: bool = False,
    dynamic: bool = False,
    forced_branch: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> dict:
    """Create a trunk, slanted branches and a multi-sphere crown."""
    xy = np.asarray(xy, dtype=float)
    center = np.array([xy[0], xy[1], tree_height / 2.0])
    branches = []
    if forced_branch is not None:
        branch_start, branch_end, branch_radius = forced_branch
        branches.append(
            {
                "start": np.asarray(branch_start, dtype=float).tolist(),
                "end": np.asarray(branch_end, dtype=float).tolist(),
                "radius": float(branch_radius),
                "route_blocker": True,
            }
        )

    base_angle = rng.uniform(0.0, 2.0 * np.pi)
    while len(branches) < 3:
        index = len(branches)
        angle = base_angle + index * 2.0 * np.pi / 3.0 + rng.uniform(-0.24, 0.24)
        length = rng.uniform(0.85, 1.45)
        branch_height = rng.uniform(3.5, min(tree_height - 1.5, 7.2))
        branch_start = np.array([xy[0], xy[1], branch_height])
        branch_end = branch_start + np.array(
            [
                np.cos(angle) * length,
                np.sin(angle) * length,
                rng.uniform(-0.35, 0.65),
            ]
        )
        branch_end[0] = np.clip(branch_end[0], BOUNDS[0, 0] + 0.1, BOUNDS[1, 0] - 0.1)
        branch_end[1] = np.clip(branch_end[1], BOUNDS[0, 1] + 0.1, BOUNDS[1, 1] - 0.1)
        branch_end[2] = np.clip(branch_end[2], 2.7, tree_height - 0.7)
        branches.append(
            {
                "start": branch_start.tolist(),
                "end": branch_end.tolist(),
                "radius": float(rng.uniform(0.075, 0.13)),
                "route_blocker": False,
            }
        )

    canopy_spheres = []
    for index, branch in enumerate(branches):
        endpoint = np.asarray(branch["end"], dtype=float)
        angle = base_angle + index * 2.0 * np.pi / 3.0
        canopy_center = endpoint + np.array(
            [0.22 * np.cos(angle), 0.22 * np.sin(angle), rng.uniform(1.0, 1.8)]
        )
        canopy_center[0] = np.clip(canopy_center[0], BOUNDS[0, 0] + 0.3, BOUNDS[1, 0] - 0.3)
        canopy_center[1] = np.clip(canopy_center[1], BOUNDS[0, 1] + 0.3, BOUNDS[1, 1] - 0.3)
        canopy_radius = float(rng.uniform(0.62, 0.92))
        canopy_center[2] = np.clip(
            canopy_center[2],
            4.8 + canopy_radius,
            BOUNDS[1, 2] - canopy_radius - 0.1,
        )
        canopy_spheres.append(
            {"center": canopy_center.tolist(), "radius": canopy_radius}
        )

    return {
        "center": center.tolist(),
        "radius": float(radius),
        "height": float(tree_height),
        "branches": branches,
        "canopy_spheres": canopy_spheres,
        "direct_blocker": direct_blocker,
        "dynamic": dynamic,
    }


def build_forest(seed: int, density: str):
    config = DENSITIES[density]
    rng = np.random.default_rng(seed)
    obstacles = ObstacleSet()
    centers: list[np.ndarray] = []
    trees = []
    start = np.array([rng.uniform(-2.0, 2.0), 1.0, rng.uniform(2.4, 4.6)])
    goal = np.array([rng.uniform(-2.0, 2.0), 35.0, rng.uniform(2.4, 5.4)])

    def add_tree(xy, radius, tree_height, *, direct_blocker=False):
        tree = make_tree(
            rng,
            np.asarray(xy, dtype=float),
            float(radius),
            float(tree_height),
            direct_blocker=direct_blocker,
        )
        add_tree_obstacles(obstacles, tree)
        centers.append(np.asarray(xy, dtype=float))
        trees.append(tree)

    # One trunk is intentionally placed on the straight line.  The global
    # planner therefore has to demonstrate avoidance rather than merely
    # validate an unobstructed reference.
    direct_t = 0.42
    direct_point = start * (1.0 - direct_t) + goal * direct_t
    add_tree(direct_point[:2], 0.34, 10.2, direct_blocker=True)

    for _ in range(config["trees"] - 1):
        for _attempt in range(800):
            xy = np.array([rng.uniform(-5.45, 5.45), rng.uniform(3.0, 33.0)])
            if np.linalg.norm(xy - start[:2]) < 2.1:
                continue
            if np.linalg.norm(xy - goal[:2]) < 2.1:
                continue
            if any(
                np.linalg.norm(xy - existing) < config["minimum_spacing"]
                for existing in centers
            ):
                continue
            add_tree(xy, rng.uniform(0.20, 0.40), rng.uniform(8.4, 10.6))
            break
        else:
            raise RuntimeError(f"could not place all trees for {density=} {seed=}")
    return start, goal, obstacles, trees


def choose_dynamic_obstacle(
    reference_path: np.ndarray,
    rng: np.random.Generator,
    seed: int,
):
    lengths = np.linalg.norm(np.diff(reference_path, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    segment_mid_arcs = cumulative[:-1] + lengths / 2.0
    segment_index = int(np.argmin(np.abs(segment_mid_arcs - cumulative[-1] / 2.0)))
    midpoint = (reference_path[segment_index] + reference_path[segment_index + 1]) / 2.0
    forward = reference_path[segment_index + 1] - reference_path[segment_index]
    forward /= np.linalg.norm(forward)
    up_reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, up_reference)) > 0.95:
        up_reference = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_reference)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    desired_direction = ("up", "down", "left", "right")[seed % 4]
    if desired_direction == "left":
        side = 1.0
    elif desired_direction == "right":
        side = -1.0
    else:
        side = -1.0 if segment_index % 2 else 1.0
    if desired_direction in ("left", "right"):
        tree_xy = midpoint[:2].copy()
    else:
        tree_xy = midpoint[:2] + side * 1.25 * right[:2]
    tree_xy[0] = np.clip(tree_xy[0], BOUNDS[0, 0] + 0.6, BOUNDS[1, 0] - 0.6)
    if desired_direction in ("up", "down"):
        branch_start = np.array([tree_xy[0], tree_xy[1], midpoint[2]])
        # Extend through the reference path so the newly perceived obstacle is
        # a genuine lateral branch crossing.
        branch_end = midpoint - side * 0.48 * right
        branch_end = np.clip(branch_end, BOUNDS[0] + 0.15, BOUNDS[1] - 0.15)
        route_branch = (branch_start, branch_end, 0.12)
    else:
        # A newly perceived full-height tree forces a lateral maneuver; its
        # branches and canopy remain active collision geometry as well.
        route_branch = None
    tree = make_tree(
        rng,
        tree_xy,
        radius=0.30 if desired_direction in ("left", "right") else 0.27,
        tree_height=10.7 if desired_direction in ("left", "right") else 9.8,
        dynamic=True,
        forced_branch=route_branch,
    )
    if desired_direction in ("left", "right"):
        # Keep the randomly generated crown above the local flight layer so
        # the deliberate side-closing foliage below controls left-vs-right.
        for branch in tree["branches"]:
            start = np.asarray(branch["start"], dtype=float)
            end = np.asarray(branch["end"], dtype=float)
            lift = max(0.0, 6.3 - min(start[2], end[2]))
            start[2] += lift
            end[2] += lift
            branch["start"] = start.tolist()
            branch["end"] = end.tolist()
        for crown in tree["canopy_spheres"]:
            crown["center"][2] = max(crown["center"][2], 8.0)
    blocked_side = {
        "up": -up,
        "down": up,
        "left": right,
        "right": -right,
    }[desired_direction]
    if desired_direction in ("left", "right"):
        bias_lobes = ((0.95, 0.75),)
    else:
        bias_lobes = ()
    for bias_offset, bias_radius in bias_lobes:
        bias_center = midpoint + bias_offset * blocked_side
        bias_center = np.clip(bias_center, BOUNDS[0] + 0.2, BOUNDS[1] - 0.2)
        tree["canopy_spheres"].append(
            {
                "center": bias_center.tolist(),
                "radius": bias_radius,
                "route_bias": desired_direction,
            }
        )
    return segment_index, tree


def minimum_clearance(path: np.ndarray | None, trees: list[dict]) -> float | None:
    if path is None or len(path) < 2:
        return None
    sampled = []
    for start, end in zip(path[:-1], path[1:]):
        count = max(2, int(np.ceil(np.linalg.norm(end - start) * 100.0)) + 1)
        alpha = np.linspace(0.0, 1.0, count)[:, None]
        sampled.append(start * (1.0 - alpha) + end * alpha)
    points = np.vstack(sampled)
    best = np.inf
    for tree in trees:
        center = np.asarray(tree["center"], dtype=float)
        radial = np.linalg.norm(points[:, :2] - center[:2], axis=1) - tree["radius"]
        axial = np.abs(points[:, 2] - center[2]) - tree["height"] / 2.0
        outside = np.linalg.norm(
            np.maximum(np.column_stack((radial, axial)), 0.0), axis=1
        )
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        best = min(best, float(np.min(outside + inside)))
        for branch in tree["branches"]:
            start = np.asarray(branch["start"], dtype=float)
            end = np.asarray(branch["end"], dtype=float)
            axis = end - start
            denominator = float(np.dot(axis, axis))
            if denominator <= 1e-20:
                closest = np.broadcast_to(start, points.shape)
            else:
                alpha = ((points - start) @ axis) / denominator
                closest = start + np.clip(alpha, 0.0, 1.0)[:, None] * axis
            signed = np.linalg.norm(points - closest, axis=1) - branch["radius"]
            best = min(best, float(np.min(signed)))
        for crown in tree["canopy_spheres"]:
            signed = (
                np.linalg.norm(points - np.asarray(crown["center"]), axis=1)
                - crown["radius"]
            )
            best = min(best, float(np.min(signed)))
    return best


def avoidance_direction(
    reference_path: np.ndarray,
    realtime_path: np.ndarray | None,
    segment_index: int,
) -> str | None:
    if realtime_path is None or len(realtime_path) <= len(reference_path):
        return None
    start = reference_path[segment_index]
    end = reference_path[segment_index + 1]
    forward = end - start
    forward /= np.linalg.norm(forward)
    up_reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, up_reference)) > 0.95:
        up_reference = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_reference)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    bypass = realtime_path[segment_index + 1]
    along = np.dot(bypass - start, forward)
    closest = start + np.clip(along, 0.0, np.linalg.norm(end - start)) * forward
    offset = bypass - closest
    horizontal = float(np.dot(offset, right))
    vertical = float(np.dot(offset, up))
    if abs(horizontal) >= abs(vertical):
        return "right" if horizontal >= 0.0 else "left"
    return "up" if vertical >= 0.0 else "down"


def percentile_summary(values):
    array = np.asarray(values, dtype=float)
    return {
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(np.max(array)),
    }


def run_scenario(density: str, seed: int):
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 100_003)
    start, goal, obstacles, trees = build_forest(seed, density)
    space_size = tuple(BOUNDS[1] - BOUNDS[0])

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full_planner = FIRIPlanner(obstacles, space_size, BOUNDS)
        full_planner.set_safety_margin(SAFETY_MARGIN)
        full_started = time.perf_counter()
        full_path = full_planner.plan_path(
            start,
            goal,
            smoothing=False,
            safety_margin=SAFETY_MARGIN,
            use_spherical_guidance=True,
        )
        full_ms = (time.perf_counter() - full_started) * 1000.0

    full_path = np.asarray(full_path, dtype=float)
    full_safe = not full_planner.check_path_safety(full_path)
    reference_path = resample_polyline(full_path)
    injected_segment, dynamic_tree = choose_dynamic_obstacle(
        reference_path, rng, seed
    )
    add_tree_obstacles(obstacles, dynamic_tree)
    all_trees = [*trees, dynamic_tree]

    with contextlib.redirect_stdout(io.StringIO()):
        realtime_planner = FIRIPlanner(obstacles, space_size, BOUNDS)
        realtime_planner.set_safety_margin(SAFETY_MARGIN)
    reference_collisions = realtime_planner.check_path_safety(reference_path)
    # The benchmark scope is a steady-state 50 Hz cycle.  Warm only the exact
    # kernels used below; do not run an untimed repair or alter planner state.
    realtime_planner.check_path_safety(reference_path)
    realtime_planner.realtime_guide.render_obstacles(
        reference_path[injected_segment],
        reference_path[injected_segment + 1] - reference_path[injected_segment],
    )
    realtime_path = realtime_planner.plan_realtime(
        start,
        goal,
        reference_path=reference_path,
        time_budget=TIME_BUDGET,
        max_repairs=1,
    )
    stats = dict(realtime_planner.last_realtime_stats)
    reactive_safe = bool(
        realtime_path is not None
        and not realtime_planner.check_path_safety(realtime_path)
    )
    reference_length = path_length(reference_path)
    reactive_length = path_length(realtime_path)
    direction = avoidance_direction(reference_path, realtime_path, injected_segment)

    return {
        "id": f"{density}-{seed:02d}",
        "density": density,
        "density_label": DENSITIES[density]["label"],
        "seed": seed,
        "start": start.tolist(),
        "goal": goal.tolist(),
        "trees": all_trees,
        "reference_path": reference_path.tolist(),
        "realtime_path": None if realtime_path is None else np.asarray(realtime_path).tolist(),
        "reference_collision_segments": reference_collisions,
        "injected_segment": injected_segment,
        "metrics": {
            "full_planning_ms": full_ms,
            "full_safe": full_safe,
            "reference_points": len(reference_path),
            "reference_length_m": reference_length,
            "reference_min_clearance_m": minimum_clearance(reference_path, trees),
            "reactive_planning_ms": stats["elapsed_seconds"] * 1000.0,
            "reactive_validation_ms": stats["validation_seconds"] * 1000.0,
            "reactive_guide_ms": stats["guide_seconds"] * 1000.0,
            "deadline_met": stats["deadline_met"],
            "reactive_success": stats["success"],
            "reactive_safe": reactive_safe,
            "repairs": stats["repairs"],
            "remaining_collisions": stats["remaining_collisions"],
            "reactive_length_m": reactive_length,
            "reactive_min_clearance_m": minimum_clearance(realtime_path, all_trees),
            "avoidance_direction": direction,
            "length_overhead_percent": None
            if reactive_length is None or reference_length is None
            else (reactive_length / reference_length - 1.0) * 100.0,
        },
    }


def summarize(scenarios, density):
    selected = [scenario for scenario in scenarios if scenario["density"] == density]
    timings = [scenario["metrics"]["reactive_planning_ms"] for scenario in selected]
    direction_counts = {
        direction: sum(
            scenario["metrics"]["avoidance_direction"] == direction
            for scenario in selected
        )
        for direction in ("left", "right", "up", "down")
    }
    return {
        "density": density,
        "label": DENSITIES[density]["label"],
        "scenarios": len(selected),
        "trees_per_scenario": DENSITIES[density]["trees"] + 1,
        "full_safe_count": sum(scenario["metrics"]["full_safe"] for scenario in selected),
        "reactive_success_count": sum(
            scenario["metrics"]["reactive_success"] for scenario in selected
        ),
        "reactive_safe_count": sum(
            scenario["metrics"]["reactive_safe"] for scenario in selected
        ),
        "deadline_met_count": sum(
            scenario["metrics"]["deadline_met"] for scenario in selected
        ),
        "timing": percentile_summary(timings),
        "avoidance_direction_counts": direction_counts,
        "failed_seeds": [
            scenario["seed"]
            for scenario in selected
            if not scenario["metrics"]["reactive_success"]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--densities", nargs="+", choices=DENSITIES, default=list(DENSITIES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "src/data/forest-results.json",
    )
    args = parser.parse_args()

    scenarios = []
    for density in args.densities:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            scenario = run_scenario(density, seed)
            scenarios.append(scenario)
            metrics = scenario["metrics"]
            print(
                f"{scenario['id']}: full={metrics['full_safe']} "
                f"reactive={metrics['reactive_success']} "
                f"time={metrics['reactive_planning_ms']:.2f} ms"
            )

    summaries = [summarize(scenarios, density) for density in args.densities]
    all_timings = [scenario["metrics"]["reactive_planning_ms"] for scenario in scenarios]
    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "planner": "MVIE-ConvexPlanner / bounded spherical realtime layer",
            "scope": "steady-state planning cycle; imports, planner construction and rendering excluded",
            "bounds": BOUNDS.tolist(),
            "safety_margin_m": SAFETY_MARGIN,
            "time_budget_ms": TIME_BUDGET * 1000.0,
            "seed_start": args.seed_start,
            "seed_count_per_density": args.seeds,
            "reference_max_spacing_m": 4.0,
            "collision_geometry": (
                "analytic vertical trunk cylinders, finite 3-D branch capsules, "
                "and multi-sphere canopies; renderer geometry matches collision geometry"
            ),
            "test_protocol": (
                "full FIRI reference path through a 3-D forest, then one newly perceived "
                "crossing branch attached to an off-route tree, "
                "followed by one bounded realtime repair"
            ),
        },
        "overall": {
            "scenarios": len(scenarios),
            "reactive_success_count": sum(
                scenario["metrics"]["reactive_success"] for scenario in scenarios
            ),
            "reactive_safe_count": sum(
                scenario["metrics"]["reactive_safe"] for scenario in scenarios
            ),
            "deadline_met_count": sum(
                scenario["metrics"]["deadline_met"] for scenario in scenarios
            ),
            "timing": percentile_summary(all_timings),
            "avoidance_direction_counts": {
                direction: sum(
                    scenario["metrics"]["avoidance_direction"] == direction
                    for scenario in scenarios
                )
                for direction in ("left", "right", "up", "down")
            },
        },
        "summaries": summaries,
        "scenarios": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {len(scenarios)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
