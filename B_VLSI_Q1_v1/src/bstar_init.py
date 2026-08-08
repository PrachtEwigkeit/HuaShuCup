from __future__ import annotations

import numpy as np

from .data import BlockData
from .structures import BStarTreeState


def _random_topology(n: int, rng: np.random.Generator) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """构造一棵随机合法二叉树。

    使用“空孩子槽”集合逐个插入节点，保证每个节点恰有一个父节点（根除外）。
    """
    if n <= 0:
        raise ValueError("n 必须为正")

    root = 0
    parent = np.full(n, -1, dtype=np.int32)
    left = np.full(n, -1, dtype=np.int32)
    right = np.full(n, -1, dtype=np.int32)

    # (parent_node, side), side: 0=left, 1=right
    open_slots: list[tuple[int, int]] = [(root, 0), (root, 1)]

    for node in range(1, n):
        k = int(rng.integers(len(open_slots)))
        p, side = open_slots.pop(k)
        parent[node] = p
        if side == 0:
            left[p] = node
        else:
            right[p] = node
        open_slots.append((node, 0))
        open_slots.append((node, 1))

    return root, parent, left, right


def create_initial_tree(
    blocks: BlockData,
    rng: np.random.Generator,
    method: str = "area_sorted",
) -> BStarTreeState:
    root, parent, left, right = _random_topology(blocks.n, rng)

    if method == "random":
        module_at_node = rng.permutation(blocks.n).astype(np.int32)
    elif method == "area_sorted":
        area = blocks.width.astype(np.int64) * blocks.height.astype(np.int64)
        # 大模块优先出现在前序较靠前的节点编号中
        module_at_node = np.argsort(-area).astype(np.int32)
    else:
        raise ValueError(f"未知初始化方法: {method}")

    rotated = (rng.random(blocks.n) < 0.5)

    return BStarTreeState(
        root=root,
        parent=parent,
        left=left,
        right=right,
        module_at_node=module_at_node,
        rotated=rotated,
    )
