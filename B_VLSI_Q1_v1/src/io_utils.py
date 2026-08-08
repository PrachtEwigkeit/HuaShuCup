from __future__ import annotations

import csv
import json
from pathlib import Path

from .data import BlockData
from .objective import dead_space_ratio, utilization
from .structures import Layout, SAHistory, Score


def save_layout_csv(blocks: BlockData, layout: Layout, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["block", "x", "y", "width", "height", "rotated"])
        for i, name in enumerate(blocks.names):
            writer.writerow([
                name,
                int(layout.x[i]),
                int(layout.y[i]),
                int(layout.width[i]),
                int(layout.height[i]),
                int(bool(layout.rotated[i])),
            ])


def save_history_csv(history: SAHistory, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "current_area", "best_area", "current_aspect", "best_aspect"])
        writer.writerows(zip(
            history.iteration,
            history.current_area,
            history.best_area,
            history.current_aspect,
            history.best_aspect,
        ))


def save_summary_json(
    dataset: str,
    seed: int,
    blocks: BlockData,
    layout: Layout,
    score: Score,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "dataset": dataset,
        "seed": int(seed),
        "n_blocks": blocks.n,
        "total_block_area": blocks.total_area,
        "W": int(layout.W),
        "H": int(layout.H),
        "area": int(score.area),
        "aspect_ratio": float(score.aspect),
        "utilization": float(utilization(blocks.total_area, layout)),
        "dead_space_ratio": float(dead_space_ratio(blocks.total_area, layout)),
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
