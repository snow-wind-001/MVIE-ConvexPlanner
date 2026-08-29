#!/usr/bin/env python3
"""Render a traceable, paper-sized qualitative planning figure.

The figure uses only the repository's saved ``temp/`` artifacts.  It does not
claim aggregate performance; it visualizes one recorded planning case and the
same clearance surrogate used by ``FIRIPlanner.check_point_collision``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "temp"
OUT_DIR = Path(__file__).resolve().parent

PROPOSED = "#0072B2"
RAW = "#D55E00"
NEUTRAL = "#666666"
THRESHOLD = "#CC79A7"


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def obstacle_clearance(point: np.ndarray, obstacle) -> float:
    """Match the analytic clearance surrogate used by the current planner."""
    center = np.asarray(obstacle.center, dtype=float)
    delta = np.asarray(point, dtype=float) - center
    if obstacle.shape == "sphere":
        return float(np.linalg.norm(delta) - obstacle.radius)
    if obstacle.shape == "cylinder":
        radial = np.linalg.norm(delta[:2]) - obstacle.radius
        axial = abs(delta[2]) - obstacle.height / 2.0
        return float(max(radial, axial))
    if obstacle.shape == "cuboid":
        return float(np.max(np.abs(delta) - np.asarray(obstacle.size) / 2.0))
    raise ValueError(f"Unsupported obstacle shape: {obstacle.shape}")


def resample_path(path: np.ndarray, count: int = 400) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    targets = np.linspace(0.0, total, count)
    samples = np.empty((count, 3), dtype=float)
    segment = 0
    for i, distance in enumerate(targets):
        while segment < len(lengths) - 1 and distance > cumulative[segment + 1]:
            segment += 1
        local = 0.0 if lengths[segment] == 0 else (
            distance - cumulative[segment]
        ) / lengths[segment]
        samples[i] = (1.0 - local) * path[segment] + local * path[segment + 1]
    return targets / max(total, 1e-12), samples


def clearance_profile(path: np.ndarray, obstacles) -> tuple[np.ndarray, np.ndarray]:
    progress, samples = resample_path(path)
    clearance = np.array(
        [min(obstacle_clearance(point, obs) for obs in obstacles) for point in samples]
    )
    return progress, clearance


def add_obstacle_projection(ax, obstacle, projection: str, label: str | None = None):
    center = np.asarray(obstacle.center, dtype=float)
    style = {
        "facecolor": "#D9D9D9",
        "edgecolor": NEUTRAL,
        "linewidth": 0.75,
        "alpha": 0.82,
        "zorder": 1,
        "label": label,
    }

    if projection == "top":  # horizontal y, vertical x
        if obstacle.shape in {"sphere", "cylinder"}:
            patch = Circle((center[1], center[0]), obstacle.radius, **style)
        else:
            size = np.asarray(obstacle.size, dtype=float)
            patch = Rectangle(
                (center[1] - size[1] / 2, center[0] - size[0] / 2),
                size[1],
                size[0],
                **style,
            )
    elif projection == "side":  # horizontal y, vertical z
        if obstacle.shape == "sphere":
            patch = Circle((center[1], center[2]), obstacle.radius, **style)
        elif obstacle.shape == "cylinder":
            patch = Rectangle(
                (center[1] - obstacle.radius, center[2] - obstacle.height / 2),
                2 * obstacle.radius,
                obstacle.height,
                **style,
            )
        else:
            size = np.asarray(obstacle.size, dtype=float)
            patch = Rectangle(
                (center[1] - size[1] / 2, center[2] - size[2] / 2),
                size[1],
                size[2],
                **style,
            )
    else:
        raise ValueError(projection)
    ax.add_patch(patch)


def style_projection(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)


def main() -> None:
    raw_path = np.asarray(load_pickle(DATA_DIR / "final_path.pkl"), dtype=float)
    repaired_path = np.asarray(load_pickle(DATA_DIR / "smoothed_path.pkl"), dtype=float)
    obstacle_set = load_pickle(DATA_DIR / "obstacles.pkl")
    obstacles = obstacle_set.obstacle_list

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), constrained_layout=False)
    ax_top, ax_side, ax_clear = axes

    for i, obstacle in enumerate(obstacles):
        add_obstacle_projection(ax_top, obstacle, "top", "Obstacle" if i == 0 else None)
        add_obstacle_projection(ax_side, obstacle, "side")

    for ax, vertical_index in [(ax_top, 0), (ax_side, 2)]:
        ax.plot(
            raw_path[:, 1],
            raw_path[:, vertical_index],
            color=RAW,
            linewidth=1.25,
            linestyle=(0, (3.2, 2.0)),
            marker="o",
            markersize=2.2,
            label="Planner output",
            zorder=3,
        )
        ax.plot(
            repaired_path[:, 1],
            repaired_path[:, vertical_index],
            color=PROPOSED,
            linewidth=1.65,
            label="Repaired path",
            zorder=4,
        )
        ax.scatter(
            repaired_path[0, 1], repaired_path[0, vertical_index],
            s=20, marker="o", facecolor="white", edgecolor="#222222",
            linewidth=0.8, label="Start" if ax is ax_top else None, zorder=5,
        )
        ax.scatter(
            repaired_path[-1, 1], repaired_path[-1, vertical_index],
            s=28, marker="*", facecolor="#222222", edgecolor="#222222",
            linewidth=0.6, label="Goal" if ax is ax_top else None, zorder=5,
        )
        ax.set_xlim(0, 20)

    ax_top.set_ylim(0, 6)
    style_projection(ax_top, "Longitudinal position, $y$ [m]", "Lateral position, $x$ [m]")
    ax_top.set_title("(a) Top view", loc="left", fontsize=7.8, pad=3)

    ax_side.set_ylim(0, 4)
    style_projection(ax_side, "Longitudinal position, $y$ [m]", "Height, $z$ [m]")
    ax_side.set_title("(b) Side view", loc="left", fontsize=7.8, pad=3)

    raw_s, raw_clearance = clearance_profile(raw_path, obstacles)
    repaired_s, repaired_clearance = clearance_profile(repaired_path, obstacles)
    ax_clear.plot(raw_s, raw_clearance, color=RAW, linewidth=1.25,
                  linestyle=(0, (3.2, 2.0)), label="Planner output")
    ax_clear.plot(repaired_s, repaired_clearance, color=PROPOSED,
                  linewidth=1.65, label="Repaired path")
    ax_clear.axhline(0.30, color=THRESHOLD, linewidth=1.0,
                     linestyle=(0, (2.0, 1.8)), label="Safety threshold")
    ax_clear.fill_between(
        repaired_s, repaired_clearance, 0.30,
        where=repaired_clearance < 0.30,
        color=THRESHOLD, alpha=0.18, linewidth=0,
    )
    ax_clear.set_xlim(0, 1)
    lower = min(float(raw_clearance.min()), float(repaired_clearance.min()), 0.0)
    upper = max(float(raw_clearance.max()), float(repaired_clearance.max()), 0.30)
    pad = max(0.08, 0.08 * (upper - lower))
    ax_clear.set_ylim(lower - pad, upper + pad)
    style_projection(ax_clear, "Normalized path length", "Clearance surrogate [m]")
    ax_clear.set_title("(c) Collision-check margin", loc="left", fontsize=7.8, pad=3)

    handles, labels = ax_top.get_legend_handles_labels()
    clear_handles, clear_labels = ax_clear.get_legend_handles_labels()
    lookup = dict(zip(labels + clear_labels, handles + clear_handles))
    order = ["Obstacle", "Planner output", "Repaired path", "Start", "Goal", "Safety threshold"]
    fig.legend(
        [lookup[name] for name in order],
        order,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.19, top=0.82, wspace=0.36)

    stem = OUT_DIR / "fig2_planning_case"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"raw_min_clearance={raw_clearance.min():.6f}")
    print(f"repaired_min_clearance={repaired_clearance.min():.6f}")
    print(f"raw_path_length={np.linalg.norm(np.diff(raw_path, axis=0), axis=1).sum():.6f}")
    print(f"repaired_path_length={np.linalg.norm(np.diff(repaired_path, axis=0), axis=1).sum():.6f}")


if __name__ == "__main__":
    main()
