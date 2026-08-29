# Paper Figure Deliverables

## Figure 1: Method Overview

Status: prompt and low-fidelity computation blueprint complete; raster generation not executed because this session does not expose a verified GPT Image 2 built-in capability. The complete minimized prompt is in `../visual-composer/architecture-prompt.txt`. No repository source, result file, identity, or reference image was sent to an external image model.

The existing repository image `MVIE-ConvexPlanner.jpg` should not be submitted unchanged. It contains malformed text (`ru(t)`), implies a stronger corridor/MVIE constraint story than the inspected execution supports, and lacks a visible recovery branch even though repair produces safety in the saved case.

## Figure 2: Saved-Case Planning And Clearance

Files:

- `fig2_planning_case.svg`: canonical editable vector source with live text.
- `fig2_planning_case.pdf`: vector PDF for manuscript inclusion.
- `fig2_planning_case.png`: 600-dpi preview.
- `plot_planning_case.py`: deterministic rendering script.

Recommended caption:

> **Saved-case effect of collision repair.** Top and side projections compare the nominal planner output with the repaired path in the archived repository scene. The final panel reports the analytic clearance surrogate used by the current collision checker along normalized path length. The nominal path crosses occupied space, whereas the repaired path remains above the 0.30 m threshold in this case. This single example illustrates the recovery mechanism and is not an aggregate success-rate result.

Suggested LaTeX placement:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig2_planning_case.pdf}
  \caption{Saved-case effect of collision repair. Top and side projections compare the nominal planner output with the repaired path in the archived repository scene. The final panel reports the analytic clearance surrogate used by the current collision checker along normalized path length. The nominal path crosses occupied space, whereas the repaired path remains above the 0.30\,m threshold in this case. This single example illustrates the recovery mechanism and is not an aggregate success-rate result.}
  \label{fig:planning-case}
\end{figure*}
```

## Render QA Ledger

| Issue | Artifact | Severity | Fix | Status |
| --- | --- | --- | --- | --- |
| Label clipping | Figure 2 | High | Rendered at final full-width aspect and inspected | Passed |
| Color-only distinction | Figure 2 | Medium | Added dashed/solid line styles and distinct markers | Passed |
| Claim exceeds evidence | Figure 2 caption | High | Bounded caption to one archived case | Passed |
| Vector editability | Figure 2 SVG/PDF | Medium | Kept SVG text live and exported PDF from Matplotlib vector source | Passed |
| Architecture topology | Figure 1 | High | Frozen from source files and documented in visual contract | Passed at specification stage |
| GPT Image 2 identity | Figure 1 | High | Did not substitute another image backend | Blocked in current session |

## No-Fabrication Status

- Figure 2 uses only saved repository geometry and the planner's implemented clearance surrogate.
- No aggregate benchmark values were plotted.
- Figure 1 contains no performance numbers and has not been presented as a generated artifact.
