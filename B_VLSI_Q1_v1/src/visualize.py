from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .data import BlockData
from .fixed_outline import FixedOutlineSolution, FixedSAHistory
from .structures import Layout, SAHistory


def plot_layout(
    blocks: BlockData,
    layout: Layout,
    save_path: str | Path,
    annotate: bool = True,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    for i, name in enumerate(blocks.names):
        rect = Rectangle(
            (layout.x[i], layout.y[i]),
            layout.width[i],
            layout.height[i],
            fill=False,
            linewidth=0.8,
        )
        ax.add_patch(rect)
        if annotate:
            ax.text(
                layout.x[i] + layout.width[i] / 2,
                layout.y[i] + layout.height[i] / 2,
                name,
                ha="center",
                va="center",
                fontsize=6,
            )

    ax.set_xlim(0, layout.W)
    ax.set_ylim(0, layout.H)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Floorplan: W={layout.W}, H={layout.H}, Area={layout.area}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_convergence(history: SAHistory, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.iteration, history.current_area, label="Current area", linewidth=0.8)
    ax.plot(history.iteration, history.best_area, label="Best area", linewidth=1.4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Area")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_fixed_layout(
    blocks: BlockData,
    solution: FixedOutlineSolution,
    save_path: str | Path,
    annotate: bool = True,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    layout = solution.layout
    dx = solution.score.offset_x
    dy = solution.score.offset_y

    fig, ax = plt.subplots(figsize=(9, 9))
    outline = Rectangle(
        (0, 0),
        solution.outline_side,
        solution.outline_side,
        fill=False,
        edgecolor="crimson",
        linewidth=1.8,
        linestyle="--",
    )
    ax.add_patch(outline)
    for i, name in enumerate(blocks.names):
        x = float(layout.x[i]) + dx
        y = float(layout.y[i]) + dy
        rect = Rectangle(
            (x, y),
            layout.width[i],
            layout.height[i],
            facecolor="steelblue",
            edgecolor="white",
            linewidth=0.45,
            alpha=0.72,
        )
        ax.add_patch(rect)
        if annotate:
            ax.text(
                x + layout.width[i] / 2,
                y + layout.height[i] / 2,
                name,
                ha="center",
                va="center",
                fontsize=5.5,
            )

    ax.set_xlim(0, solution.outline_side)
    ax.set_ylim(0, solution.outline_side)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Fixed outline {solution.outline_side:.3f} x {solution.outline_side:.3f}, "
        f"HPWL={solution.score.hpwl:.1f}"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_fixed_convergence(history: FixedSAHistory, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(history.iteration, history.current_hpwl, linewidth=0.7, label="Current")
    axes[0].plot(history.iteration, history.best_hpwl, linewidth=1.2, label="Best")
    axes[0].set_ylabel("HPWL")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history.iteration, history.current_overflow, linewidth=0.7, label="Current")
    axes[1].plot(history.iteration, history.best_overflow, linewidth=1.2, label="Best")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Normalized overflow")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
