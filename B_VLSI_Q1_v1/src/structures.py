from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BStarTreeState:
    root: int
    parent: np.ndarray
    left: np.ndarray
    right: np.ndarray
    module_at_node: np.ndarray
    rotated: np.ndarray

    @property
    def n(self) -> int:
        return int(self.parent.size)

    def copy(self) -> "BStarTreeState":
        return BStarTreeState(
            root=int(self.root),
            parent=self.parent.copy(),
            left=self.left.copy(),
            right=self.right.copy(),
            module_at_node=self.module_at_node.copy(),
            rotated=self.rotated.copy(),
        )


@dataclass
class Layout:
    # 以下数组均按 block_id 索引，而不是 tree node 索引
    x: np.ndarray
    y: np.ndarray
    width: np.ndarray
    height: np.ndarray
    rotated: np.ndarray
    W: int
    H: int
    area: int


@dataclass(frozen=True)
class Score:
    area: int
    aspect: float


@dataclass
class SAHistory:
    iteration: list[int]
    current_area: list[int]
    best_area: list[int]
    current_aspect: list[float]
    best_aspect: list[float]
