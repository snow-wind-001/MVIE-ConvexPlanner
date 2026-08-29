# Literature Search: MVIE Convex-Corridor Trajectory Planning

Date: 2026-08-27
Search purpose: closest-work and novelty-risk analysis for the current MVIE-ConvexPlanner repository
Target venue/family: robotics and autonomous-systems journals, using an IEEE T-RO/RA-L-level novelty and evidence lens
Source-quality policy: applied; final entries use stable proceedings, DOI, or arXiv records

## Summary

- The region-generation core is closest to IRIS, RILS, and FIRI. FIRI already alternates restrictive inflation and MVIE while enforcing seed-set manageability.
- Convex-corridor trajectory optimization with continuous-time safety and dynamic constraints is already represented by GCOPTER and related safe-flight-corridor planners.
- B-spline safety and dynamic-feasibility mechanisms based on convex-hull properties and knot-time adjustment predate this repository.
- The most important recent overlap is joint convex-cover and trajectory optimization: Wu et al. optimize waypoint-containing maximum-volume regions to improve downstream trajectories.
- The current repository therefore needs a narrower mechanism-level novelty claim. A pipeline-level claim based only on combining safety push, FIRI/MVIE, SLSQP, and B-splines is not defensible against this literature set.

## Paper Table

| # | Title | Year | Venue/source | Stable link | Type | Insight | Completeness | Numeric evidence | Overall | Relevance note |
| ---: | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | Computing Large Convex Regions of Obstacle-Free Space Through Semidefinite Programming | 2015 | Algorithmic Foundations of Robotics XI | [Springer proceedings](https://link.springer.com/book/10.1007/978-3-319-16595-0) | pure method | 5 | 4 | 3 | A | IRIS anchor: alternating separating hyperplanes and a large inscribed ellipsoid already define the classical region-inflation pattern. |
| 2 | Generating Large Convex Polytopes Directly on Point Clouds | 2020 | arXiv public preprint | [arXiv:2010.08744](https://arxiv.org/abs/2010.08744) | pure method | 4 | 4 | 4 | A | Direct point-cloud free-polytope construction with convexity and obstacle exclusion; undercuts a generic “supports point clouds/3-D obstacles” claim. |
| 3 | Fast Iterative Region Inflation for Computing Large 2-D/3-D Convex Regions of Obstacle-Free Space | 2025 | IEEE Transactions on Robotics; arXiv preprint | [DOI](https://doi.org/10.1109/TRO.2025.3562482), [arXiv:2403.02977](https://arxiv.org/abs/2403.02977) | pure method | 5 | 5 | 5 | Risk | The nearest region-generation work. It already couples RsI and MVIE, guarantees seed-set manageability, and targets quality/efficiency jointly. |
| 4 | Robust and Efficient Quadrotor Trajectory Generation for Fast Autonomous Flight | 2019 | IEEE Robotics and Automation Letters; arXiv preprint | [arXiv:1907.01531](https://arxiv.org/abs/1907.01531) | method + benchmark | 4 | 5 | 5 | Risk | Uses B-spline convex-hull properties for safety/dynamic feasibility and non-uniform knot-time adjustment; directly challenges the repository's raw finite-difference constraint claim. |
| 5 | Real-time Trajectory Generation for Quadrotors using B-spline based Non-uniform Kinodynamic Search | 2019 | arXiv public preprint | [arXiv:1904.12348](https://arxiv.org/abs/1904.12348) | pure method | 4 | 4 | 4 | Risk | Searches B-spline control points and derives convex-hull-based dynamic-feasibility conditions. |
| 6 | EGO-Planner: An ESDF-free Gradient-based Local Planner for Quadrotors | 2020 | arXiv public preprint | [arXiv:2008.08835](https://arxiv.org/abs/2008.08835) | method + benchmark | 4 | 5 | 5 | B | Strong baseline for efficient local replanning and collision-aware trajectory deformation without a precomputed ESDF. |
| 7 | Geometrically Constrained Trajectory Optimization for Multicopters | 2021 | arXiv public preprint, updated 2022 | [arXiv:2103.00190](https://arxiv.org/abs/2103.00190) | pure method | 5 | 5 | 5 | Risk | GCOPTER already optimizes trajectories inside polyhedral or ball-shaped safe-flight corridors with continuous-time velocity/acceleration limits. |
| 8 | Bubble Planner: Planning High-speed Smooth Quadrotor Trajectories using Receding Corridors | 2022 | arXiv public preprint | [arXiv:2202.12177](https://arxiv.org/abs/2202.12177) | method + benchmark | 4 | 4 | 4 | B | Combines receding overlapping corridors with minimum-control-effort trajectory optimization for high-speed flight. |
| 9 | Motion Planning around Obstacles with Convex Optimization | 2022 | arXiv public preprint | [arXiv:2205.04422](https://arxiv.org/abs/2205.04422) | pure method | 5 | 5 | 4 | A | Graphs of Convex Sets demonstrates a broader convex-optimization route for obstacle-aware motion planning under dynamics. |
| 10 | Towards Optimizing a Convex Cover of Collision-Free Space for Trajectory Generation | 2024 | arXiv public preprint, updated 2025 | [arXiv:2406.09631](https://arxiv.org/abs/2406.09631) | pure method | 5 | 4 | 4 | Risk | Jointly updates waypoint-containing, maximum-volume convex regions to improve downstream trajectory optimization; the strongest overlap with a coupled corridor/planner claim. |
| 11 | Differentiable Collision-Free Parametric Corridors | 2024 | arXiv public preprint | [arXiv:2407.12283](https://arxiv.org/abs/2407.12283) | pure method | 4 | 4 | 4 | B | Represents smooth differentiable collision-free corridors, showing that corridor parameterization itself remains an active contribution axis. |

Scores describe source quality for this search, not acceptance probability. Numeric-evidence scores reflect the inspectable evaluation coverage reported in the paper metadata/body; they do not reproduce or compare individual result values.

## Closest-Work Clusters

### 1. Region inflation and MVIE

- Representative papers: IRIS, direct point-cloud convex polytopes, FIRI.
- Already covered: iterative obstacle-separating regions, inscribed-ellipsoid enlargement, point-cloud operation, 2-D/3-D convex free-region generation, seed-set containment.
- Remaining gap: a repository-specific claim would need either a new certified inflation mechanism, a better optimization objective tied to trajectory feasibility, or a shape-aware formulation with measurable tightness/runtime benefits.
- Effect on this project: “FIRI + MVIE” cannot itself be presented as new, and the current implementation must first reproduce the source algorithm faithfully.

### 2. Corridor-constrained trajectory optimization

- Representative papers: GCOPTER, Bubble Planner, Graphs of Convex Sets.
- Already covered: trajectories constrained to convex regions, geometric/dynamic constraints, spatial-temporal optimization, corridor sequences and overlaps.
- Remaining gap: certified feasibility of adjacent-corridor transitions, task-specific robustness, or a demonstrably better coupled objective.
- Effect on this project: SLSQP over three samples per segment is weaker than existing continuous-time formulations and is not a novelty delta.

### 3. B-spline kinodynamic feasibility

- Representative papers: Fast-Planner, non-uniform kinodynamic B-spline search, EGO-Planner.
- Already covered: control-point optimization, convex-hull safety reasoning, velocity/acceleration control points, time allocation, online replanning.
- Remaining gap: a new proof, constraint transformation, or substantially different system setting.
- Effect on this project: unscaled second/third control-point differences cannot support a physical acceleration/jerk claim without knot timing and continuous-time verification.

### 4. Coupled corridor and trajectory design

- Representative papers: Wu et al. 2024/2025 and differentiable parametric corridors.
- Already covered: optimizing corridor geometry or seed selection for downstream trajectory quality and differentiable corridor representations.
- Remaining gap: a simpler certified alternating algorithm, a formal feasibility condition for mixed primitives, or a real-time solver with decisive ablations.
- Effect on this project: the best rescue route is to make corridor–trajectory compatibility the central mechanism, not to present a sequential module stack as the contribution.

## Opportunity Map

| Cluster | Status | Open gap | Possible direction | Evidence needed | Risk |
| --- | --- | --- | --- | --- | --- |
| FIRI reproduction | covered central claim | Current code does not implement source-faithful RsI/MVIE | Build a verified reference implementation before extending it | Unit tests against analytic polytopes and source benchmarks | High |
| Adjacent-corridor feasibility | mechanism gap | Independent segment ellipsoids need not overlap or contain shared control points | Optimize shared waypoints and corridor intersections jointly | Feasibility proof or monotonic merit decrease; overlap ablation | Medium–high |
| Continuous-time safety/dynamics | covered but implementation gap | Three segment samples and time-free differences are insufficient | Use Bézier/B-spline convex-hull constraints with explicit knot durations | Dense verification plus certified bound; velocity/acceleration/jerk plots | Medium |
| Mixed primitive geometry | crowded but open | Current FIRI path reduces cylinders/cuboids to bounding spheres | Use analytic support functions and quantify conservativeness | Tightness/runtime study versus point-cloud and bounding-sphere variants | Medium–high; requires more search |
| Failure-aware repair | engineering gap | Heuristic bypass succeeds locally but lacks a scientific mechanism | Topology-aware corridor repair with termination/progress condition | Failure taxonomy, success/runtime comparison, adversarial scenes | Medium–high |

## Benchmark And Dataset Candidates

| Candidate | Source | Task | Metrics | Fit | Risk |
| --- | --- | --- | --- | --- | --- |
| FIRI public implementation/benchmarks | [FIRI paper](https://arxiv.org/abs/2403.02977) | Convex-region generation | runtime, volume/quality, seed manageability | Essential for validating the claimed FIRI component | Exact protocol must match the paper |
| Fast-Planner simulation and real-flight scenarios | [Fast-Planner paper](https://arxiv.org/abs/1907.01531) | B-spline local trajectory generation | success, runtime, safety, dynamic feasibility | Strong trajectory baseline | ROS/system integration effort |
| GCOPTER random SFC benchmarks | [GCOPTER paper](https://arxiv.org/abs/2103.00190) | Corridor-constrained trajectory optimization | objective, runtime, feasibility | Direct optimizer baseline | Requires continuous-time formulation |
| Parameterized convex-cover environments | [Convex-cover paper](https://arxiv.org/abs/2406.09631) | Coupled cover/trajectory quality | cover geometry and downstream trajectory cost | Directly tests the proposed rescue direction | Implementation complexity and protocol matching |

## Citation And Positioning Cautions

- Cite the final FIRI publication as IEEE Transactions on Robotics 2025; the repository's RA-L 2022 attribution is inconsistent with the verified DOI/arXiv record.
- Do not state that FIRI only supports spherical obstacles. Its public formulation targets obstacle point sets and general 2-D/3-D free-region computation.
- Do not claim dynamic feasibility from raw control-point differences without a time parameterization and a continuous-time bound.
- Do not treat “30 random scenes” as a benchmark until seeds, per-scene outputs, baselines, failure criteria, and aggregate uncertainty are archived.
