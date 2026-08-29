# CCFA Innovation Review: MVIE-ConvexPlanner

Date: 2026-08-27
Mode: standard
Review lens: robotics/autonomous-systems SCI journal, calibrated against an IEEE T-RO/RA-L-level contribution bar
Literature status: searched and body-verified for the closest region-generation and trajectory-optimization clusters
Repository state tracking: no `ccfa.yaml` was present, so no project-state file was modified

## Verdict First

**Current conference/journal readiness: low. Development potential: medium. Weighted idea score: 2.50/5.**

The current project is a useful engineering prototype, but its present paper claim is not sufficiently novel for a strong robotics SCI journal. FIRI already contains the iterative RsI–MVIE mechanism; GCOPTER already handles safe-corridor trajectory optimization with continuous-time geometric and dynamic constraints; Fast-Planner and related work already use B-spline structure for safety and dynamic feasibility; recent convex-cover work already couples waypoint-containing maximum-volume regions to downstream trajectory quality.

After subtracting that prior art, the repository's remaining distinctive content is an engineering safety-recovery stack: pre-FIRI waypoint pushing, approximate per-segment regions, sampled SLSQP constraints, B-spline smoothing, and heuristic collision repair. That stack is not yet a defensible algorithmic contribution because the region solver is not a source-faithful FIRI/MVIE implementation, the SLSQP problem is infeasible in the inspected saved case, the acceleration/jerk quantities are not time-scaled physical derivatives, and the reported 30-scene aggregate has no archived per-scene evidence or benchmark driver in the repository.

**Recommendation: high-risk reformulate.** Preserve the project, but choose one mechanism-level contribution and repair correctness before drafting a paper. The strongest rescue route is joint corridor–trajectory feasibility: optimize shared waypoints and overlapping corridor intersections together, then enforce continuous-time B-spline/Bézier safety and dynamic constraints with explicit timing.

## Project Goal And Current Stage

| Field | Assessment |
| --- | --- |
| Project goal | Generate collision-free, smooth paths through mixed 3-D obstacle scenes using FIRI/MVIE-inspired safe regions and constrained optimization. |
| Current stage | Working prototype with one saved illustrative run; pre-validation for publication. |
| Known artifacts | Python planner, saved obstacle/path pickles, one performance JSON, visualizations, README/changelog claims. |
| Missing artifacts | Source-faithful algorithm validation, benchmark script, fixed seed list, per-scene records, baseline implementations, ablations, continuous-time feasibility checks, statistical aggregation. |
| Gate decision | Do not write a novelty-forward paper yet. Pass correctness and reproducibility gates first. |
| Next owner | `ccf-experiment-designer` after the method is repaired; `ccf-paper-writer` only after benchmark evidence exists. |

## Normalized Idea

| Element | Current formulation |
| --- | --- |
| Problem | Safe and smooth 3-D path generation in cluttered scenes containing spheres, cylinders, and cuboids. |
| Claimed gap | FIRI alone is presented as lacking waypoint recovery, hard corridor constraints, dynamic constraints, and collision repair. |
| Root challenge | A geometric free-space abstraction must remain compatible with a time-parameterized trajectory while preserving collision and dynamic feasibility. |
| Current insight | Push unsafe waypoints outward, build one FIRI/MVIE-inspired region per segment, optimize control points, then repair any remaining collision. |
| Mechanism | KD-tree nearest-neighbor queries; center-directed safety push; approximate halfspaces; candidate inscribed ellipsoid; SLSQP over segment samples and raw finite differences; B-spline smoothing; bypass insertion. |
| Expected evidence | Higher collision-free rate with acceptable runtime, path length, clearance, and smoothness. |
| Current evidence | One saved case with a collision-free repaired path; changelog/README assertions of 20- and 30-scene runs without archived raw run table. |
| Main limitation | The saved case reaches safety through post-optimization repair, while the corridor-constrained SLSQP stage does not converge to a feasible solution. |

## Code-Evidence Audit

### What is implemented

- The main executable imports `FIRIPlanner` from `firi/planning/plannerv2.py` (`main.py:12`) and executes safety push, corridor computation, trajectory optimization, B-spline smoothing, fallback replanning, and a second collision-repair layer (`main.py:329-394`; `firi/planning/plannerv2.py:392-461`).
- Safety push iteratively moves interior waypoints when a clearance surrogate is below `d_safe` (`firi/planning/plannerv2.py:156-223`).
- Each path segment uses its two endpoints and midpoint as seeds (`firi/planning/plannerv2.py:228-251`).
- SLSQP minimizes squared second differences with ellipsoid membership at `t={0,0.5,1}` plus raw second- and third-difference bounds (`firi/planning/plannerv2.py:256-357`).
- The final executable recomputes collisions and inserts one or two bypass points if necessary (`main.py:178-270`, `main.py:381-394`).

### Correctness and claim gaps

| Issue | Repository evidence | Scientific consequence | Required repair |
| --- | --- | --- | --- |
| The FIRI implementation does not perform the documented ellipsoid-standard-space RsI operation | `restrictive_inflation` constructs center/radius-based halfspaces directly and never calls the available ellipsoid halfspace transforms (`firi/planning/firi.py:61-163`) | The implementation should not be described as reproducing the source FIRI algorithm | Reimplement and test RsI from the published formulation or rename the component as an approximation |
| Mixed primitives are reduced to bounding radii during region generation | Cuboids use half the size-vector norm; cylinders use `max(radius,height/2)` (`firi/planning/firi.py:74-84`) | “Native sphere/cylinder/cuboid corridor generation” is overstated and can be highly conservative | Use exact support functions or document the bounding-sphere approximation and measure its tightness |
| The MVIE solver is not translation invariant in an analytic box test | A shifted 2×2×2 box returned `Q=0.5 I`, volume 1.481, while the unit cube returned `Q=I`, volume 4.189 | The solver does not reliably compute a maximum-volume inscribed ellipsoid | Replace it with a verified formulation; add translation, scaling, containment, and optimality regression tests |
| Segment ellipsoids do not contain the segment seeds | In the saved case, every corridor had at least one endpoint/midpoint with ellipsoid quadratic value above 1 | The SLSQP initial point is infeasible; adjacent segment constraints may have no shared feasible waypoint | Constrain trajectories to overlapping polytopes or jointly optimize corridor overlap and shared waypoints |
| SLSQP failure can still produce a candidate | The code evaluates `result.x` even when `result.success` is false (`firi/planning/plannerv2.py:342-351`) | Collision count alone does not certify satisfaction of corridor or dynamic constraints | Reject any solution with optimizer failure or explicit constraint violation |
| Collision constraints sample only three points per segment | `t={0,0.5,1}` (`firi/planning/plannerv2.py:301-310`) | No continuous segment or spline safety guarantee | Use polytope/Bézier/B-spline convex-hull containment or an adaptive certified bound |
| `a_max` and `jerk_max` are raw control-point differences | No knot duration appears in the second/third-difference formulas (`firi/planning/plannerv2.py:312-326`) | These values cannot be interpreted as physical acceleration or jerk | Introduce time parameterization and derivative control points with correct powers of knot duration |
| B-spline smoothing returns only sampled positions | `bspline_smooth` generates `len(path)` samples without returning knot times or derivative splines (`firi/planning/plannerv2.py:608-616`) | Dynamic feasibility cannot be measured or certified after smoothing | Preserve spline control points and knots; evaluate continuous velocity, acceleration, and jerk |
| Safety-margin parameters are inconsistent | `plan_path(... safety_margin=...)` does not use the argument; collision checking defaults to hard-coded 0.3; configuration also defines 0.02 and 0.5 values | Results are difficult to reproduce and claims about the configured safety margin are ambiguous | Define one documented clearance contract and pass it through generation, optimization, repair, and evaluation |
| Postprocessing can stale the collision ledger | Clipping, deduplication, and simplification occur after the last planner collision assignment without recomputation (`firi/planning/plannerv2.py:437-446`) | Internal safety status may not match the returned path | Recompute all safety and constraint metrics after the final geometry change |
| The performance chart treats smoothness as seconds | `record_value` stores all values in `durations`; the chart includes numeric fields that do not end in selected suffixes (`performance_evaluator.py:89-92`, `148-185`) | The existing timing chart is scientifically misleading | Separate metrics from durations and validate units before plotting |

## Reproduced Diagnostic Results

These are local checks run on 2026-08-27 against the current repository; they are diagnostic evidence, not a benchmark.

| Check | Result | Interpretation |
| --- | --- | --- |
| Python syntax compilation | Passed for the main planner, FIRI, MVIE, and geometry modules | The inspected code imports/compiles in the current environment |
| Shifted-box MVIE invariance | Failed: volume 1.481 versus 4.189 for the same-size centered cube | MVIE output changes under translation and is not maximal in the shifted case |
| Saved-case corridor seed feasibility | Failed in all 5 corridors | At least one endpoint or midpoint lies outside every corresponding candidate MVIE |
| Saved-case SLSQP | Failed with “Positive directional derivative for linesearch” | The corridor-constrained optimizer did not reach a feasible optimum |
| Saved-case raw planner output | Minimum analytic planner-clearance surrogate −0.363 m; 2 colliding segments under the current checker | The optimizer output is unsafe |
| Saved-case repaired output | Minimum surrogate 0.334 m at a 0.30 m threshold; no sampled collision | Safety in this case comes from the post-hoc repair layer |
| Saved-case path-length effect | 17.642 m to 19.341 m, approximately +9.6% | Repair trades path length for clearance in this one case |
| Archived aggregate evidence | One `performance_data.json`; no 30-row run table or batch driver | README/changelog aggregate claims are not independently reproducible from committed artifacts |

## Closest Prior Art And Novelty Delta

| Closest work | What it already covers | Overlap with this project | Remaining novelty delta | Risk |
| --- | --- | --- | --- | --- |
| IRIS, Deits and Tedrake, 2015 | Alternating collision-separating polytopes and large inscribed ellipsoids | Core safe-region pattern | None unless a new inflation/feasibility mechanism is added | High |
| Direct point-cloud convex polytopes, Zhong et al., 2020 | Fast guaranteed convex free regions directly from point clouds | 3-D free-region generation | Mixed-primitive exactness could differ, but current code uses bounding spheres | Medium–high |
| FIRI, Wang et al., IEEE T-RO 2025 | RsI–MVIE iteration, seed manageability, quality and efficiency | Nearest named algorithm and claimed base | Current safety push and repair are outside FIRI, but are heuristic and unproven | Fatal to a generic FIRI-improvement claim |
| Fast-Planner, Zhou et al., RA-L 2019 | B-spline safety/dynamics via convex hull and knot-time adjustment | Smooth dynamically feasible trajectory goal | Current finite differences are weaker, not new | High |
| GCOPTER, Wang et al., arXiv 2021/2022 | Continuous-time safe-corridor and dynamic constraints | Corridor-constrained backend | Current sampled SLSQP offers no positive novelty delta | High |
| Convex-cover optimization, Wu et al., arXiv 2024/2025 | Joint waypoint/corridor design for downstream trajectory quality | Closest rescue direction | A simpler certified overlap-feasibility mechanism may remain open | High but fixable |

**Novelty delta after subtraction:** the code has a practical recovery sequence that pushes waypoints and inserts bypass points after an infeasible/unsafe nominal solve. This is presently an integration contribution, not a new planning principle. It becomes scientifically stronger only if the repair step is formalized around a measurable failure mode, compared against strong baselines, and shown to provide a repeatable benefit beyond generic local replanning.

## Independent Expert Panel

### Field expert

- Score tendency: 2.5/5; confidence 4/5.
- Strongest argument: safe corridor generation and trajectory recovery remain important in cluttered 3-D planning.
- Rejection-grade concern: the claimed gap is largely covered by FIRI, GCOPTER, Fast-Planner, and coupled convex-cover optimization.
- Score-change condition: identify one failure mode not handled by those methods and demonstrate a mechanism that resolves it under a matched protocol.

### Method expert

- Score tendency: 1.5/5; confidence 5/5.
- Strongest argument: the code exposes a clear modular path from geometry to repair.
- Rejection-grade concern: the named MVIE is not maximal in a translation test; corridor constraints are infeasible in the saved case; dynamic limits lack time scaling.
- Score-change condition: pass analytic geometry tests, enforce optimizer feasibility, and derive continuous-time dynamics from an explicit spline timing model.

### Experiment expert

- Score tendency: 1/5; confidence 5/5.
- Strongest argument: one saved case is traceable and the repair effect can be visualized.
- Rejection-grade concern: no archived batch results, baselines, ablations, seed list, uncertainty, or continuous-time constraint measurements support the 30-scene claims.
- Score-change condition: release a deterministic benchmark matrix with per-scene outcomes and comparisons to source-faithful baselines.

### AC / venue expert

- Score tendency: 2/5; confidence 3/5 because the exact journal is unspecified.
- Strongest argument: an application-focused journal may value a reliable mixed-obstacle planning system.
- Rejection-grade concern: at a T-RO/RA-L bar, the current contribution reads as a fragile integration with unsupported algorithm names and claims.
- Score-change condition: make one mechanism the paper's center, prove or certify its key property, and show decisive multi-setting evidence.

### Skeptical prior-art expert

- Score tendency: 2/5; confidence 4/5.
- Strongest “already known” objection: every major stage has a close established analogue, and recent work already optimizes the corridor for the trajectory rather than merely sequencing modules.
- Score-change condition: show a novelty table with exact assumptions, guarantees, objective, solver, and evidence that no closest method shares.

### Panel synthesis

- Agreement: the problem is relevant, but the current method and evidence are not publication-ready.
- Strongest accept axis: a reproducible, failure-aware safety recovery mechanism could become a practical contribution.
- Strongest reject axis: method correctness fails before novelty can be credited.
- Most valuable next evidence: source-faithful FIRI/GCOPTER baselines plus a certified shared-corridor feasibility test.
- Panel-calibrated recommendation: high-risk reformulate, with medium development potential.

## Rubric Scores

| Dimension | Weight | Score (1–5) | Confidence (1–5) | Deduction / evidence basis | Repair condition |
| --- | ---: | ---: | ---: | --- | --- |
| Problem importance | 12 | 4 | 4 | Safe 3-D motion planning is important; the repository does not yet isolate a new bottleneck | Define the exact unresolved failure mode and target operating regime |
| Novelty against likely prior work | 14 | 2 | 4 | FIRI, GCOPTER, Fast-Planner, and coupled convex-cover optimization cover the central components | Establish a mechanism-level delta after a source-faithful comparison |
| Conceptual innovation | 12 | 2 | 4 | Current insight is a sequential combination plus heuristic repair | Formalize a new corridor–trajectory compatibility or repair principle |
| Method soundness | 14 | 2 | 5 | MVIE invariance failure, infeasible sampled corridor constraints, time-free dynamics | Pass analytic correctness tests and continuous-time feasibility checks |
| Elegance and simplicity | 8 | 3 | 4 | Modular pipeline is understandable but duplicates fallback/repair logic in planner and `main.py` | Unify safety recovery under one explicit objective/state machine |
| Feasibility under resources | 8 | 4 | 4 | Python prototype runs quickly in the saved case | Add deterministic configuration and remove solver/geometry fragility |
| Experimental convincibility | 10 | 1 | 5 | One archived run, no baseline or batch evidence | Publish per-scene benchmark data, ablations, uncertainty, and failure cases |
| Venue and audience fit | 8 | 2 | 3 | Strong-journal lens assumed; current artifact resembles an engineering demo | Narrow venue or raise contribution/evidence to a main-track method paper |
| Timeliness and topic heat | 6 | 4 | 4 | Corridor and convex-set planning remain active | Tie the method to a current unresolved constraint rather than the broad topic |
| Risk-adjusted acceptance potential | 8 | 2 | 3 | Correctness and novelty blockers dominate | Clear all fatal gates before manuscript investment |

**Weighted final score:** 2.50/5. This is a diagnostic score, not an acceptance probability.
**Current readiness:** low.
**Development potential:** medium.
**Overall confidence:** high for code/method diagnosis; medium for venue fit because no target journal was supplied.

## Fixability And Upgrade Plan

| Issue | Severity | Fix class | Can fix before writing? | Required change/evidence | Expected diagnostic effect |
| --- | --- | --- | --- | --- | --- |
| Source-faithful FIRI/MVIE | High | method redesign | Yes | Translation/scaling-invariant MVIE, polytope containment tests, source benchmark parity | Raises method soundness if all tests pass |
| Adjacent corridor incompatibility | High | method redesign | Yes | Shared-waypoint/intersection feasibility or joint corridor–trajectory optimization | Could create the main novelty delta |
| Time-free dynamic constraints | High | method redesign | Yes | Explicit knot durations and continuous velocity/acceleration/jerk bounds | Converts a misleading claim into a defensible one |
| Heuristic repair as hidden success source | High | problem refinement | Yes | Either remove it from the claimed method or formalize it as the central recovery mechanism | Improves claim–mechanism alignment |
| Missing benchmarks and ablations | High | evidence design | Partly | Fixed maps/seeds, baselines, per-scene logs, CIs, failure taxonomy | Raises experimental score from 1 only after data exists |
| FIRI citation and obstacle-support wording | Medium | novelty grounding | Yes | Correct 2025 T-RO citation and point-set formulation | Removes an avoidable credibility failure |

### Best rescue route: feasibility-aware corridor co-design

1. Reproduce a verified FIRI region generator and a verified trajectory backend.
2. Define the failure: independently generated segment regions may not share a feasible transition/control-point set.
3. Optimize shared waypoints and adjacent corridor overlaps with one merit function that couples region quality, overlap margin, and trajectory cost.
4. Use time-parameterized spline control points and continuous-time safety/dynamic bounds.
5. Compare against sequential FIRI→GCOPTER, Fast-Planner/EGO-Planner, and the current repair stack.
6. Report not only success but why failures occur: empty overlap, optimizer infeasibility, insufficient clearance, or dynamic-limit violation.

If this route produces a verified mechanism and decisive evidence, novelty and method-soundness scores could plausibly move from 2 toward 3–4. No score movement should be claimed before those conditions are met.

## Minimum Publication Evidence

- Region generation: volume/quality, runtime, seed containment, obstacle exclusion, translation/scaling invariance, and corridor-overlap margin.
- Trajectory: continuous minimum clearance, path/time cost, integrated jerk or snap, maximum velocity/acceleration/jerk, and solver feasibility residuals.
- Reliability: at least a deterministic multi-seed scenario matrix with narrow passages, density sweeps, mixed primitives, and adversarial failures.
- Comparisons: source-faithful FIRI/RILS region generation, GCOPTER or another continuous corridor backend, and one B-spline local planner.
- Ablations: no safety push, no joint overlap objective, no repair, approximate versus exact primitive geometry, sampled versus continuous constraints.
- Reporting: median and dispersion for runtime/path cost, confidence interval for success rate, identical hardware, fixed seeds, and archived per-run records.

## Evidence That Would Change The Verdict

The verdict changes materially if the project demonstrates all of the following:

1. Verified FIRI/MVIE correctness on analytic and published benchmarks.
2. A new corridor–trajectory or repair mechanism with an explicit invariant, guarantee, or consistently measurable effect.
3. Continuous-time collision and dynamic feasibility rather than sampled surrogates.
4. Strong-baseline, multi-setting evidence with reproducible raw records.

Until then, describe the repository as an experimental prototype, not a novel MVIE/FIRI trajectory-planning algorithm.
