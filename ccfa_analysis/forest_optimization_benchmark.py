"""Ablate realtime collision-check optimizations on fixed forest scenarios."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from firi.planning.plannerv2 import FIRIPlanner  # noqa: E402
from obstacle_generator import ObstacleSet  # noqa: E402


def timing_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def build_planner(scenario, bounds, safety_margin):
    obstacles = ObstacleSet()
    for tree in scenario["trees"]:
        obstacles.add_obstacle(
            "cylinder",
            np.asarray(tree["center"], dtype=float),
            radius=tree["radius"],
            height=tree["height"],
        )
    with contextlib.redirect_stdout(io.StringIO()):
        planner = FIRIPlanner(
            obstacles, tuple(bounds[1] - bounds[0]), bounds
        )
        planner.set_safety_margin(safety_margin)
    return planner


def run_variant(payload, variant, repeats, warmups):
    bounds = np.asarray(payload["metadata"]["bounds"], dtype=float)
    safety_margin = float(payload["metadata"]["safety_margin_m"])
    budget = float(payload["metadata"]["time_budget_ms"]) / 1000.0
    runs = []
    for scenario in payload["scenarios"]:
        planner = build_planner(scenario, bounds, safety_margin)
        if variant == "sampled_segments":
            # Keeps the new vectorized point/path mask, but disables the exact
            # sphere/cylinder segment fast path to isolate its contribution.
            planner._analytic_segment_collision = lambda _p1, _p2: (False, False)
        reference = np.asarray(scenario["reference_path"], dtype=float)

        for _ in range(warmups):
            planner.plan_realtime(
                reference[0], reference[-1], reference_path=reference,
                time_budget=budget, max_repairs=1,
            )
        for repeat in range(repeats):
            path = planner.plan_realtime(
                reference[0], reference[-1], reference_path=reference,
                time_budget=budget, max_repairs=1,
            )
            stats = dict(planner.last_realtime_stats)
            verified_safe = bool(
                path is not None and not planner.check_path_safety(path)
            )
            runs.append(
                {
                    "scenario_id": scenario["id"],
                    "density": scenario["density"],
                    "repeat": repeat,
                    "success": stats["success"],
                    "verified_safe": verified_safe,
                    "deadline_met": stats["deadline_met"],
                    "planning_ms": stats["elapsed_seconds"] * 1000.0,
                    "validation_ms": stats["validation_seconds"] * 1000.0,
                    "guide_ms": stats["guide_seconds"] * 1000.0,
                }
            )
    return runs


def summarize(runs, densities):
    summaries = {}
    for density in [*densities, "overall"]:
        selected = runs if density == "overall" else [
            run for run in runs if run["density"] == density
        ]
        scenario_ids = sorted({run["scenario_id"] for run in selected})
        summaries[density] = {
            "calls": len(selected),
            "scenarios": len(scenario_ids),
            "success_count": sum(run["success"] for run in selected),
            "verified_safe_count": sum(run["verified_safe"] for run in selected),
            "deadline_met_count": sum(run["deadline_met"] for run in selected),
            "scenario_success_count": sum(
                all(run["success"] for run in selected if run["scenario_id"] == scenario_id)
                for scenario_id in scenario_ids
            ),
            "failed_scenarios": [
                scenario_id
                for scenario_id in scenario_ids
                if not all(
                    run["success"]
                    for run in selected
                    if run["scenario_id"] == scenario_id
                )
            ],
            "planning_timing": timing_summary(
                [run["planning_ms"] for run in selected]
            ),
            "validation_timing": timing_summary(
                [run["validation_ms"] for run in selected]
            ),
            "guide_timing": timing_summary(
                [run["guide_ms"] for run in selected]
            ),
        }
    return summaries


def recorded_baseline(payload):
    return {
        "overall": payload["overall"],
        "densities": {
            summary["density"]: summary for summary in payload["summaries"]
        },
        "note": "single run per fixed scenario, recorded before analytic segment optimization",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "forest_simulator/src/data/forest-results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "ccfa_analysis/forest_optimization_benchmark.json",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    try:
        input_reference = args.input.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        input_reference = args.input.name
    densities = [summary["density"] for summary in payload["summaries"]]

    variants = {}
    raw_runs = {}
    for variant in ("sampled_segments", "analytic_segments"):
        runs = run_variant(payload, variant, args.repeats, args.warmups)
        raw_runs[variant] = runs
        variants[variant] = summarize(runs, densities)
        overall = variants[variant]["overall"]
        print(
            variant,
            f"safe={overall['verified_safe_count']}/{overall['calls']}",
            f"deadline={overall['deadline_met_count']}/{overall['calls']}",
            f"p95={overall['planning_timing']['p95_ms']:.2f} ms",
            f"max={overall['planning_timing']['max_ms']:.2f} ms",
        )

    result = {
        "protocol": {
            "input": input_reference,
            "repeats_per_scenario": args.repeats,
            "warmups_per_scenario": args.warmups,
            "scope": "steady-state plan_realtime; construction and verification excluded from timing",
            "variants": {
                "sampled_segments": "vectorized point/path mask with sampled segment fallback",
                "analytic_segments": "vectorized point/path mask plus exact sphere/in-height cylinder segments",
            },
        },
        "recorded_baseline": recorded_baseline(payload),
        "variants": variants,
        "runs": raw_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
