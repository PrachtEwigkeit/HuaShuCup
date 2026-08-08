from __future__ import annotations

import numpy as np

from .data import BlockData
from .structures import BStarTreeState, Layout


def _block_wh(blocks: BlockData, block_id: int, rotated: bool) -> tuple[int, int]:
    if rotated:
        return int(blocks.height[block_id]), int(blocks.width[block_id])
    return int(blocks.width[block_id]), int(blocks.height[block_id])


def _preorder_nodes(state: BStarTreeState) -> list[int]:
    order: list[int] = []
    stack = [int(state.root)]
    seen = set()
    while stack:
        u = int(stack.pop())
        if u in seen:
            raise ValueError("B*-Tree 中检测到环或重复节点")
        seen.add(u)
        order.append(u)
        # 先压 right，再压 left，使 left 先被处理
        r = int(state.right[u])
        l = int(state.left[u])
        if r != -1:
            stack.append(r)
        if l != -1:
            stack.append(l)
    if len(order) != state.n:
        raise ValueError(f"B*-Tree 非连通: 访问 {len(order)} / {state.n} 个节点")
    return order


def pack_bstar(blocks: BlockData, state: BStarTreeState) -> Layout:
    """用 B*-Tree + 整数 skyline 解码为无重叠布局。

    B*-Tree 规则：
      - 左孩子的 x = 父模块 x + 父模块宽度；
      - 右孩子的 x = 父模块 x；
      - y 由当前 skyline 在 [x, x+w) 上的最大高度决定。

    题目尺寸均为整数，因此第一版使用整数栅格 skyline，逻辑直观且便于验证。
    """
    n = blocks.n
    if state.n != n:
        raise ValueError("state 节点数与 block 数不一致")

    skyline = np.zeros(blocks.max_packed_width, dtype=np.int32)

    # node 索引下的临时信息
    node_x = np.zeros(n, dtype=np.int32)
    node_y = np.zeros(n, dtype=np.int32)
    node_w = np.zeros(n, dtype=np.int32)
    node_h = np.zeros(n, dtype=np.int32)

    # block 索引下的最终信息
    x = np.zeros(n, dtype=np.int32)
    y = np.zeros(n, dtype=np.int32)
    width = np.zeros(n, dtype=np.int32)
    height = np.zeros(n, dtype=np.int32)

    order = _preorder_nodes(state)

    for node in order:
        block_id = int(state.module_at_node[node])
        w, h = _block_wh(blocks, block_id, bool(state.rotated[block_id]))
        node_w[node] = w
        node_h[node] = h

        if node == state.root:
            x0 = 0
        else:
            p = int(state.parent[node])
            if p < 0:
                raise ValueError("非根节点缺少父节点")
            if int(state.left[p]) == node:
                x0 = int(node_x[p] + node_w[p])
            elif int(state.right[p]) == node:
                x0 = int(node_x[p])
            else:
                raise ValueError("parent/child 指针不一致")

        x1 = x0 + w
        if x1 > skyline.size:
            raise RuntimeError("skyline 预分配宽度不足；请检查输入或上界计算")

        y0 = int(skyline[x0:x1].max(initial=0))
        skyline[x0:x1] = y0 + h

        node_x[node] = x0
        node_y[node] = y0

        x[block_id] = x0
        y[block_id] = y0
        width[block_id] = w
        height[block_id] = h

    W = int(np.max(x.astype(np.int64) + width.astype(np.int64)))
    H = int(np.max(y.astype(np.int64) + height.astype(np.int64)))
    area = int(W * H)

    return Layout(
        x=x,
        y=y,
        width=width,
        height=height,
        rotated=state.rotated.copy(),
        W=W,
        H=H,
        area=area,
    )
