from __future__ import annotations

import numpy as np

from .structures import BStarTreeState


def rotate(state: BStarTreeState, rng: np.random.Generator) -> None:
    node = int(rng.integers(state.n))
    block_id = int(state.module_at_node[node])
    state.rotated[block_id] = ~state.rotated[block_id]


def swap_modules(state: BStarTreeState, rng: np.random.Generator) -> None:
    u, v = rng.choice(state.n, size=2, replace=False)
    u, v = int(u), int(v)
    tmp = int(state.module_at_node[u])
    state.module_at_node[u] = state.module_at_node[v]
    state.module_at_node[v] = tmp


def _subtree_nodes(state: BStarTreeState, root: int) -> set[int]:
    nodes: set[int] = set()
    stack = [int(root)]
    while stack:
        u = int(stack.pop())
        if u in nodes:
            raise ValueError("树中检测到环")
        nodes.add(u)
        l = int(state.left[u])
        r = int(state.right[u])
        if l != -1:
            stack.append(l)
        if r != -1:
            stack.append(r)
    return nodes


def move_subtree(state: BStarTreeState, rng: np.random.Generator, max_tries: int = 50) -> bool:
    """将一棵非根子树整体剪下，并接到另一节点的空孩子槽。

    返回是否真的执行了移动。第一版保留子树内部拓扑不变，逻辑安全、易验证。
    """
    if state.n <= 1:
        return False

    for _ in range(max_tries):
        u = int(rng.integers(state.n))
        if u == state.root:
            continue

        subtree = _subtree_nodes(state, u)
        candidates: list[tuple[int, int]] = []  # (v, side), 0 left / 1 right

        p_old = int(state.parent[u])
        old_side = 0 if int(state.left[p_old]) == u else 1

        for v in range(state.n):
            if v in subtree:
                continue
            if int(state.left[v]) == -1:
                candidates.append((v, 0))
            if int(state.right[v]) == -1:
                candidates.append((v, 1))
            # 原槽位在剪下后会变空，也允许作为候选，但会造成无变化；后面排除
            if v == p_old:
                if old_side == 0 and int(state.left[v]) == u:
                    candidates.append((v, 0))
                elif old_side == 1 and int(state.right[v]) == u:
                    candidates.append((v, 1))

        candidates = [c for c in candidates if c != (p_old, old_side)]
        if not candidates:
            continue

        v, side = candidates[int(rng.integers(len(candidates)))]

        # 先剪下 u
        if int(state.left[p_old]) == u:
            state.left[p_old] = -1
        elif int(state.right[p_old]) == u:
            state.right[p_old] = -1
        else:
            raise ValueError("parent/child 指针不一致")

        # 确认目标槽在剪下后为空
        if side == 0:
            if int(state.left[v]) != -1:
                # 极少数情况下候选槽状态变化，撤销
                state.left[p_old] = u if old_side == 0 else state.left[p_old]
                state.right[p_old] = u if old_side == 1 else state.right[p_old]
                return False
            state.left[v] = u
        else:
            if int(state.right[v]) != -1:
                state.left[p_old] = u if old_side == 0 else state.left[p_old]
                state.right[p_old] = u if old_side == 1 else state.right[p_old]
                return False
            state.right[v] = u

        state.parent[u] = v
        return True

    return False


def perturb(
    state: BStarTreeState,
    rng: np.random.Generator,
    p_rotate: float,
    p_swap: float,
    p_move: float,
) -> BStarTreeState:
    probs = np.asarray([p_rotate, p_swap, p_move], dtype=float)
    if np.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("邻域概率非法")
    probs = probs / probs.sum()

    cand = state.copy()
    op = int(rng.choice(3, p=probs))
    if op == 0:
        rotate(cand, rng)
    elif op == 1:
        swap_modules(cand, rng)
    else:
        ok = move_subtree(cand, rng)
        if not ok:
            # 退化为 swap，确保候选解产生变化
            swap_modules(cand, rng)
    return cand
