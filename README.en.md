# MVIE-ConvexPlanner: Convex Corridor Path Planning via Maximum Volume Inscribed Ellipsoids

<p align="center">
  <img src="MVIE-ConvexPlanner.jpg" width="600" alt="MVIE-ConvexPlanner Algorithm"/>
</p>

## Introduction

MVIE-ConvexPlanner is a safe trajectory planning algorithm for autonomous robots in 3D obstacle environments. It improves upon the [FIRI algorithm](https://ieeexplore.ieee.org/document/9697174) from Zhejiang University's FastLab by adding **iterative safety pushing**, **MVIE convex corridors**, and **constrained trajectory optimization** to produce dynamically feasible, collision-free smooth trajectories.

> **Original paper**: *J. Liu, et al., "Fast Iterative Region Inflation for Computing Large 2-D/3-D Convex Regions of Obstacle-Free Space," IEEE RA-L, 2022.*

## Improvements over FIRI

| Feature | Original FIRI | MVIE-ConvexPlanner |
|---------|---------------|-------------------|
| Safety push | None | Iterative push of control points before corridor computation |
| Corridor constraint | Heuristic waypoint adjustment | Ellipsoid corridors as hard optimization constraints |
| Trajectory optimization | Unconstrained B-spline smoothing | SLSQP-constrained optimization (a_max, jerk_max) |
| Obstacle types | Spheres only | Spheres, cylinders, cuboids |
| Collision repair | None | Segment-level bypass point insertion |
| Collision-free rate | ~85% | **100%** (30-scenario benchmark) |

## Algorithm Pipeline

Following Algorithm 1 (MVIE-ConvexPlanner) pseudocode:

1. **Obstacle preprocessing** — Build KD-Tree for efficient nearest-neighbor queries
2. **Initial control points** — Linear skeleton between start/goal + sine perturbation
3. **Iterative safety push** (Steps 5-13) — Push unsafe control points away from obstacles
4. **Safety corridor computation** (Steps 15-22) — FIRI inflation + MVIE for each segment
5. **Constrained trajectory optimization** (Step 25) — SLSQP optimization with corridor, acceleration, and jerk constraints
6. **B-spline smoothing** — Output dynamically feasible cubic B-spline trajectory

## Project Structure

```
MVIE-ConvexPlanner/
├── main.py                         # Main entry point (scene config, planning pipeline)
├── obstacle_generator.py           # Obstacle generation (sphere/cylinder/cuboid, density)
├── visualizer.py                   # Visualization (Matplotlib + Open3D)
├── performance_evaluator.py        # Performance evaluator
├── path_planner.py                 # Basic path utilities
├── utils.py                        # General utilities
├── firi/                           # Core algorithm package
│   ├── geometry/
│   │   ├── ellipsoid.py            # Ellipsoid (SVD, halfspace transforms)
│   │   └── convex_polytope.py      # Convex polytope (halfspace/vertex, Chebyshev center)
│   ├── planning/
│   │   ├── config.py               # Configuration (d_safe, a_max, jerk_max, etc.)
│   │   ├── firi.py                 # Core FIRI (restrictive_inflation)
│   │   ├── mvie.py                 # MVIE solver (Affine Scaling + Khachiyan)
│   │   ├── plannerv2.py            # Main planner (push/corridor/optimize/replan)
│   │   └── planner.py              # Original planner (backward compatibility)
│   └── utils/
│       ├── obstacle_generator.py   # Internal obstacle tools
│       └── analyze_path.py         # Path analysis
├── test/                           # Test scripts
├── temp/                           # Runtime outputs (auto-generated)
└── CHANGELOG.md                    # Changelog
```

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- Open3D (optional, for interactive 3D visualization)

### Setup

```bash
git clone https://github.com/snow-wind-001/MVIE-ConvexPlanner.git
cd MVIE-ConvexPlanner

pip install numpy scipy matplotlib open3d psutil
```

## Usage

### Run Path Planning

```bash
python main.py
```

Configurable parameters at the top of `main.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SEED` | int/None | None | Random seed; None for different each run |
| `SPACE_BOUNDS` | ndarray | [[0,0,0],[6,20,4]] | Simulation space bounds |
| `N_SPHERES` | int | 3 | Number of sphere obstacles |
| `N_CYLINDERS` | int | 2 | Number of cylinder obstacles |
| `N_CUBOIDS` | int | 3 | Number of cuboid obstacles |
| `DENSITY` | str | 'medium' | Obstacle density: 'low'/'medium'/'high' |
| `NUM_ON_PATH` | int | 2 | Spheres placed on the start-goal line |
| `SAFETY_MARGIN` | float | 1.2 | Inflation safety margin |

### Trajectory Analysis

```bash
python analyze_trajectory.py   # Analyze path angles, curvature, safety
python angle_comparison.py     # Compare original vs smoothed path
```

## Key Parameters

Configured in `firi/planning/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_safe` | 0.5 | Safety push distance threshold |
| `push_iterations` | 10 | Max push iterations |
| `a_max` | 4.0 | Max acceleration constraint (2nd-order control point difference) |
| `jerk_max` | 8.0 | Max jerk constraint (3rd-order control point difference) |
| `safety_iterations` | 2 | FIRI iteration count |
| `volume_threshold` | 0.01 | MVIE convergence threshold |

## Performance

Benchmark results across 30 random scenarios:

| Metric | Value |
|--------|-------|
| Collision-free rate | **100%** (30/30) |
| Mean planning time | 2.44 s |
| Max planning time | 7.23 s |
| Edge device limit | < 60 s ✅ |

## Visualization

Two visualization methods:
- **Matplotlib** — Static 3D path plot (auto-saved to `temp/path_visualization.png`)
- **Open3D** — Interactive 3D visualization + offscreen rendering

## Acknowledgments

- This project improves upon the FIRI algorithm from Zhejiang University FastLab
- Original paper: Liu J, et al. *Fast Iterative Region Inflation for Computing Large 2-D/3-D Convex Regions of Obstacle-Free Space*. IEEE RA-L, 2022.

## License

MIT License
