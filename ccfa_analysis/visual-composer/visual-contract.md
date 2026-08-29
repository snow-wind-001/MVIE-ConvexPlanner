# Visual Contract

## Artifact 1: Method Overview Draft

- Target venue / format: robotics SCI journal; full-width two-column figure; approximately 178 mm wide; landscape 16:7.
- Destination: paper mechanism figure.
- Core claim: the current implementation obtains safety through a nominal region/optimization path followed by explicit collision verification and recovery.
- Reviewer question: Which operations produce the candidate path, where are constraints applied, and which branch actually recovers an unsafe candidate?
- Evidence layer: method/mechanism.
- Source method content: `firi/planning/plannerv2.py`, `firi/planning/firi.py`, `firi/planning/mvie.py`, and `main.py`.
- Statistics / uncertainty: none; the architecture figure contains no numerical results.
- Figure prototype: one left-to-right computation graph with one feedback/recovery branch.
- Caption role: explain the verified execution order without claiming that every stage is novel or certified.
- Manuscript placement: method overview, immediately after the problem formulation and before implementation details.
- Output status: GPT Image 2 raster draft unavailable in this session because no verified built-in `image_gen` capability is exposed. No substitute model or pure-SVG architecture was used.
- Reference images sent externally: none.
- Traceability: every displayed module and edge maps to the source files above.

### Representation-to-operation map

| Representation | Operator | Output |
| --- | --- | --- |
| Start, goal, mixed obstacle primitives | Initial waypoint generator | Initial control points |
| Initial control points and nearest obstacle | Iterative safety push | Pushed control points |
| Segment endpoints and midpoint | Restrictive halfspace construction | Convex polytope |
| Convex polytope | Candidate MVIE solver | Candidate ellipsoid |
| Control points and per-segment ellipsoids | Sampled SLSQP | Optimized control points |
| Optimized control points | B-spline smoothing | Candidate path |
| Candidate path and obstacle checker | Collision check | Safe/unsafe decision |
| Unsafe candidate | Corridor-center move, perturbation, and bypass insertion | Repaired candidate, fed back to collision check |
| Safe candidate | Final output | Final path |

### Typed connections

- Solid dark arrows: primary data flow.
- Thin blue arrows: code-specific safety-recovery flow.
- Dashed magenta edge: unsafe feedback from collision check to heuristic repair.
- Green terminal edge: path passes the current collision checker.
- No training, gradient, retrieval, or learned-model edges are present.

### Exact label inventory

`Start and goal`; `Obstacle primitives`; `Initial control points`; `Safety push`; `Segment seeds`; `Restrictive halfspaces`; `Convex polytope`; `Candidate MVIE`; `Sampled SLSQP`; `B-spline smoothing`; `Collision check`; `Heuristic repair`; `Unsafe`; `Safe`; `Final path`.

`MVIE` and `SLSQP` must remain acronym-only. No expansion should be invented inside the image.

### Low-fidelity blueprint

```text
Start and goal ─┐
                ├─> Initial control points -> Safety push -> Segment seeds
Obstacles ──────┘                                      |
                                                        v
                     Restrictive halfspaces -> Convex polytope -> Candidate MVIE
                                                                          |
                                                                          v
                     Final path <- Safe <- Collision check <- B-spline <- Sampled SLSQP
                                               |
                                             Unsafe
                                               v
                                         Heuristic repair
                                               └──────── feedback to Collision check
```

The final composition should give more area to the corridor/optimization and recovery branch than to generic inputs. It must remain a computation graph, not a sequence of presentation cards.

## Artifact 2: Saved-Case Planning Figure

- Target venue / format: robotics SCI journal; full-width two-column figure; SVG, vector PDF, and 600-dpi PNG.
- Core claim: in one archived repository case, the nominal planner output violates the current clearance surrogate, while the post-hoc repaired path clears the configured 0.30 m collision threshold.
- Reviewer question: Where does the path change, and does the repaired path exceed the same clearance rule used by the planner?
- Evidence layer: qualitative case study and limitation.
- Source data: `temp/final_path.pkl`, `temp/smoothed_path.pkl`, `temp/obstacles.pkl`.
- Figure type: three panels: top view, side view, and clearance-surrogate profile.
- Statistics / uncertainty: no aggregate statistics; one deterministic archived case only.
- Traceability: `ccfa_analysis/figures/plot_planning_case.py`.
- Palette: Okabe-Ito blue for the repaired path, vermillion for the nominal output, neutral gray obstacles, and purple threshold.
- Accessibility: critical distinctions use line style, markers, and direct legend entries in addition to color.
- Caption role: bound the claim to one case and identify the clearance surrogate and threshold.
- Manuscript placement: qualitative-results or failure-analysis subsection; not the main quantitative result.
- Output formats: editable SVG, vector PDF, and 600-dpi PNG preview.
- No-fabrication status: passed; all geometry and values are loaded from repository artifacts.
