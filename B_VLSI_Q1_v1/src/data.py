from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_BLOCK_RE = re.compile(
    r"^(?P<name>\S+)\s+block\s+4\s+"
    r"\((?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\)\s+"
    r"\((?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)\)\s+"
    r"\((?P<x3>-?\d+)\s*,\s*(?P<y3>-?\d+)\)\s+"
    r"\((?P<x4>-?\d+)\s*,\s*(?P<y4>-?\d+)\)\s*$"
)


@dataclass(frozen=True)
class BlockData:
    names: list[str]
    width: np.ndarray
    height: np.ndarray

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def total_area(self) -> int:
        return int(np.sum(self.width.astype(np.int64) * self.height.astype(np.int64)))

    @property
    def max_packed_width(self) -> int:
        """Skyline 数组安全上界：所有模块最大边之和。"""
        return int(np.sum(np.maximum(self.width, self.height))) + 1


def load_blocks(path: str | Path) -> BlockData:
    """读取题目 .blocks 文件。

    第一问只需要 HardBlock 的矩形宽、高；NumTerminals 在本问中忽略。
    文件中的矩形坐标可能不依赖固定顶点顺序，因此通过 min/max 求宽高。
    """
    path = Path(path)
    names: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    declared_n: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("NumHardBlocks"):
                declared_n = int(line.split(":", 1)[1].strip())
                continue
            if " block 4 " not in f" {line} ":
                continue

            m = _BLOCK_RE.match(line)
            if m is None:
                raise ValueError(f"无法解析 block 行: {line}")

            coords = [
                (int(m.group("x1")), int(m.group("y1"))),
                (int(m.group("x2")), int(m.group("y2"))),
                (int(m.group("x3")), int(m.group("y3"))),
                (int(m.group("x4")), int(m.group("y4"))),
            ]
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w <= 0 or h <= 0:
                raise ValueError(f"非法矩形尺寸: {line}")

            names.append(m.group("name"))
            widths.append(w)
            heights.append(h)

    if declared_n is not None and len(names) != declared_n:
        raise ValueError(
            f"HardBlock 数量不一致: 文件声明 {declared_n}, 实际解析 {len(names)}"
        )
    if not names:
        raise ValueError(f"未在 {path} 中解析到 HardBlock")

    return BlockData(
        names=names,
        width=np.asarray(widths, dtype=np.int32),
        height=np.asarray(heights, dtype=np.int32),
    )
