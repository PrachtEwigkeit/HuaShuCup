from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .data import BlockData
from .fixed_outline import FixedOutlineSolution, FixedSAHistory
from .q3_search import MinimumOutlineResult


def _clean_number(value: float) -> int | float:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return int(rounded)
    return round(float(value), 9)


def save_fixed_layout_csv(
    blocks: BlockData,
    solution: FixedOutlineSolution,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = solution.layout
    dx = solution.score.offset_x
    dy = solution.score.offset_y
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["block", "x", "y", "width", "height", "rotated"])
        for i, name in enumerate(blocks.names):
            writer.writerow(
                [
                    name,
                    _clean_number(float(layout.x[i]) + dx),
                    _clean_number(float(layout.y[i]) + dy),
                    int(layout.width[i]),
                    int(layout.height[i]),
                    int(bool(layout.rotated[i])),
                ]
            )


def save_fixed_history_csv(history: FixedSAHistory, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "current_hpwl",
                "best_hpwl",
                "current_overflow",
                "best_overflow",
                "current_feasible",
                "best_feasible",
            ]
        )
        writer.writerows(
            zip(
                history.iteration,
                history.current_hpwl,
                history.best_hpwl,
                history.current_overflow,
                history.best_overflow,
                history.current_feasible,
                history.best_feasible,
            )
        )


def fixed_summary_dict(
    dataset: str,
    seed: int,
    blocks: BlockData,
    solution: FixedOutlineSolution,
    requested_dead_space_ratio: float | None,
) -> dict:
    score = solution.score
    return {
        "dataset": dataset,
        "seed": int(seed),
        "n_blocks": blocks.n,
        "total_block_area": blocks.total_area,
        "outline_side": float(solution.outline_side),
        "effective_integer_coordinate_limit": int(
            math.floor(solution.outline_side + 1e-12)
        ),
        "outline_area": float(solution.outline_side**2),
        "realized_dead_space_ratio": float(
            solution.outline_side**2 / blocks.total_area - 1.0
        ),
        "requested_dead_space_ratio": requested_dead_space_ratio,
        "packed_width": int(solution.layout.W),
        "packed_height": int(solution.layout.H),
        "offset_x": float(score.offset_x),
        "offset_y": float(score.offset_y),
        "feasible": bool(score.feasible),
        "overflow": float(score.overflow),
        "total_hpwl": float(score.hpwl),
    }


def save_fixed_summary_json(
    dataset: str,
    seed: int,
    blocks: BlockData,
    solution: FixedOutlineSolution,
    path: str | Path,
    requested_dead_space_ratio: float | None,
) -> None:
    obj = fixed_summary_dict(
        dataset, seed, blocks, solution, requested_dead_space_ratio
    )
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_q3_search_csv(result: MinimumOutlineResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["outline_side", "dead_space_ratio", "feasible", "overflow", "hpwl", "attempts"]
        )
        for record in result.records:
            writer.writerow(
                [
                    record.outline_side,
                    record.dead_space_ratio,
                    int(record.feasible),
                    record.overflow,
                    record.hpwl,
                    record.attempts,
                ]
            )
