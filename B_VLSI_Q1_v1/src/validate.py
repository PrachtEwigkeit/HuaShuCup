from __future__ import annotations

import numpy as np

from .data import BlockData
from .structures import BStarTreeState, Layout


def validate_tree(state: BStarTreeState) -> None:
    n = state.n
    if not (0 <= state.root < n):
        raise AssertionError("root 越界")
    if int(state.parent[state.root]) != -1:
        raise AssertionError("root 的 parent 必须为 -1")

    child_count = np.zeros(n, dtype=np.int32)
    for u in range(n):
        for v in (int(state.left[u]), int(state.right[u])):
            if v == -1:
                continue
            if not (0 <= v < n):
                raise AssertionError("child 越界")
            if int(state.parent[v]) != u:
                raise AssertionError("parent/child 指针不一致")
            child_count[v] += 1

    if child_count[state.root] != 0:
        raise AssertionError("root 不能作为其他节点的孩子")
    for u in range(n):
        if u != state.root and child_count[u] != 1:
            raise AssertionError(f"节点 {u} 的父节点数量不是1")

    visited = set()
    stack = [int(state.root)]
    while stack:
        u = int(stack.pop())
        if u in visited:
            raise AssertionError("树中存在环")
        visited.add(u)
        l, r = int(state.left[u]), int(state.right[u])
        if l != -1:
            stack.append(l)
        if r != -1:
            stack.append(r)
    if len(visited) != n:
        raise AssertionError("树不连通")

    mods = np.sort(state.module_at_node)
    if not np.array_equal(mods, np.arange(n, dtype=mods.dtype)):
        raise AssertionError("module_at_node 不是 0..n-1 的排列")


def _rects_overlap(xi, yi, wi, hi, xj, yj, wj, hj) -> bool:
    separated = (
        xi + wi <= xj
        or xj + wj <= xi
        or yi + hi <= yj
        or yj + hj <= yi
    )
    return not separated


def validate_layout(blocks: BlockData, layout: Layout, check_pairs: bool = True) -> None:
    n = blocks.n
    for arr_name in ("x", "y", "width", "height", "rotated"):
        arr = getattr(layout, arr_name)
        if len(arr) != n:
            raise AssertionError(f"layout.{arr_name} 长度错误")

    if np.any(layout.x < 0) or np.any(layout.y < 0):
        raise AssertionError("存在负坐标")

    # 每个模块尺寸只能是原尺寸或旋转尺寸
    for i in range(n):
        actual = (int(layout.width[i]), int(layout.height[i]))
        original = (int(blocks.width[i]), int(blocks.height[i]))
        rotated = (original[1], original[0])
        if actual not in (original, rotated):
            raise AssertionError(f"模块 {blocks.names[i]} 尺寸错误: {actual}")

    W = int(np.max(layout.x.astype(np.int64) + layout.width.astype(np.int64)))
    H = int(np.max(layout.y.astype(np.int64) + layout.height.astype(np.int64)))
    if W != layout.W or H != layout.H or W * H != layout.area:
        raise AssertionError("W/H/area 与模块坐标不一致")

    if layout.area < blocks.total_area:
        raise AssertionError("外包面积小于模块总面积，必有错误")

    if check_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                if _rects_overlap(
                    int(layout.x[i]), int(layout.y[i]), int(layout.width[i]), int(layout.height[i]),
                    int(layout.x[j]), int(layout.y[j]), int(layout.width[j]), int(layout.height[j]),
                ):
                    raise AssertionError(
                        f"检测到重叠: {blocks.names[i]} 与 {blocks.names[j]}"
                    )
