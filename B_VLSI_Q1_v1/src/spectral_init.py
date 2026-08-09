from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import BlockData
from .netlist import NetlistData
from .structures import BStarTreeState


@dataclass(frozen=True)
class SpectralEmbedding:
    x: np.ndarray
    y: np.ndarray
    harmonic_x: np.ndarray
    harmonic_y: np.ndarray
    fiedler_a: np.ndarray
    fiedler_b: np.ndarray


def _normalized_fiedler_vectors(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = adjacency.shape[0]
    if n == 1:
        return np.zeros(1), np.zeros(1)

    degree = adjacency.sum(axis=1)
    safe = np.where(degree > 1e-12, degree, 1.0)
    inv_sqrt = 1.0 / np.sqrt(safe)
    normalized = np.eye(n) - (inv_sqrt[:, None] * adjacency * inv_sqrt[None, :])
    _, vectors = np.linalg.eigh(normalized)
    first = vectors[:, 1] if n >= 2 else vectors[:, 0]
    second = vectors[:, 2] if n >= 3 else np.arange(n, dtype=float)
    return first.astype(float), second.astype(float)


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values.astype(float) - float(np.mean(values))
    scale = float(np.std(centered))
    if scale < 1e-12:
        return np.zeros_like(centered)
    return centered / scale


def _clip_embedding_axis(
    values: np.ndarray,
    blocks: BlockData,
    outline_side: float,
) -> np.ndarray:
    half_min = np.minimum(blocks.width, blocks.height).astype(float) / 2.0
    return np.clip(values, half_min, outline_side - half_min)


def _reweighted_axis_solve(
    current: np.ndarray,
    terminal_axis: np.ndarray,
    netlist: NetlistData,
    outline_side: float,
    regularization: float,
) -> np.ndarray:
    """用网络当前包围盒的极端引脚构造一轮 Bound-to-Bound IRLS。"""

    n = netlist.n_blocks
    adjacency = np.zeros((n, n), dtype=np.float64)
    anchor_degree = np.zeros(n, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    all_axis = np.concatenate([current.astype(float, copy=False), terminal_axis])

    for net in netlist.nets:
        degree = len(net)
        if degree <= 1:
            continue
        vertices = [int(v) for v in net]
        block_vertices = [v for v in vertices if v < n]
        terminal_vertices = [v - n for v in vertices if v >= n]
        if not block_vertices:
            continue

        values = np.asarray([all_axis[v] for v in vertices], dtype=float)
        extremes = {vertices[int(np.argmin(values))], vertices[int(np.argmax(values))]}
        base = 1.0 / float(degree - 1)

        # 保留较弱的全端口锚定项，避免极端点切换时布局方向剧烈翻转。
        for i in block_vertices:
            for terminal in terminal_vertices:
                weight = 0.15 * base
                anchor_degree[i] += weight
                rhs[i] += weight * float(terminal_axis[terminal])

        for i in block_vertices:
            for target in extremes:
                if target == i:
                    continue
                distance = max(abs(float(all_axis[i] - all_axis[target])), 1.0)
                weight = base / distance
                if target < n:
                    adjacency[i, target] += weight
                    adjacency[target, i] += weight
                else:
                    terminal = target - n
                    anchor_degree[i] += weight
                    rhs[i] += weight * float(terminal_axis[terminal])

    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    scale = max(float(np.mean(np.diag(laplacian) + anchor_degree)), 1.0)
    ridge = regularization * scale
    system = laplacian + np.diag(anchor_degree) + ridge * np.eye(n)
    center = outline_side / 2.0
    solved = np.linalg.solve(system, rhs + ridge * center)
    # 阻尼可减少连续两轮极端引脚发生交换时的振荡。
    return 0.5 * current + 0.5 * solved


def anchored_spectral_embedding(
    blocks: BlockData,
    netlist: NetlistData,
    outline_side: float,
    spread_strength: float = 0.16,
    regularization: float = 1e-3,
    reweight_iterations: int = 0,
) -> SpectralEmbedding:
    """计算端口锚定的谱坐标。

    超边使用度数归一化 clique expansion。先求固定端口约束下的调和坐标，
    再叠加两个归一化 Fiedler 方向，避免强连接模块完全塌缩。
    """

    n = blocks.n
    adjacency = np.zeros((n, n), dtype=np.float64)
    anchor_degree = np.zeros(n, dtype=np.float64)
    rhs_x = np.zeros(n, dtype=np.float64)
    rhs_y = np.zeros(n, dtype=np.float64)

    for net in netlist.nets:
        degree = len(net)
        if degree <= 1:
            continue
        weight = 1.0 / float(degree - 1)
        block_vertices = [int(v) for v in net if int(v) < n]
        terminal_vertices = [int(v) - n for v in net if int(v) >= n]

        for pos, i in enumerate(block_vertices):
            for j in block_vertices[pos + 1 :]:
                adjacency[i, j] += weight
                adjacency[j, i] += weight
            for terminal in terminal_vertices:
                anchor_degree[i] += weight
                rhs_x[i] += weight * float(netlist.terminal_x[terminal])
                rhs_y[i] += weight * float(netlist.terminal_y[terminal])

    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    scale = max(float(np.mean(np.diag(laplacian) + anchor_degree)), 1.0)
    ridge = regularization * scale
    system = laplacian + np.diag(anchor_degree) + ridge * np.eye(n)
    center = outline_side / 2.0

    harmonic_x = np.linalg.solve(system, rhs_x + ridge * center)
    harmonic_y = np.linalg.solve(system, rhs_y + ridge * center)

    for _ in range(max(0, int(reweight_iterations))):
        harmonic_x = _reweighted_axis_solve(
            harmonic_x,
            netlist.terminal_x,
            netlist,
            outline_side,
            regularization,
        )
        harmonic_y = _reweighted_axis_solve(
            harmonic_y,
            netlist.terminal_y,
            netlist,
            outline_side,
            regularization,
        )

    fiedler_a, fiedler_b = _normalized_fiedler_vectors(adjacency)
    fiedler_a = _standardize(fiedler_a)
    fiedler_b = _standardize(fiedler_b)

    # 用调和坐标决定特征向量的方向，避免特征向量符号不确定导致结果翻转。
    if np.dot(fiedler_a, harmonic_x - np.mean(harmonic_x)) < 0:
        fiedler_a = -fiedler_a
    if np.dot(fiedler_b, harmonic_y - np.mean(harmonic_y)) < 0:
        fiedler_b = -fiedler_b

    x = harmonic_x + spread_strength * outline_side * fiedler_a
    y = harmonic_y + spread_strength * outline_side * fiedler_b

    x = _clip_embedding_axis(x, blocks, outline_side)
    y = _clip_embedding_axis(y, blocks, outline_side)
    return SpectralEmbedding(
        x=x,
        y=y,
        harmonic_x=harmonic_x,
        harmonic_y=harmonic_y,
        fiedler_a=fiedler_a,
        fiedler_b=fiedler_b,
    )


def spectral_embedding_variant(
    base: SpectralEmbedding,
    blocks: BlockData,
    outline_side: float,
    spread_strength: float,
    axis_mode: int,
) -> SpectralEmbedding:
    """共享调和解和特征向量，生成交换方向/翻转符号的谱坐标候选。"""

    mode = int(axis_mode) % 8
    swap_axes = mode >= 4
    sign_mode = mode % 4
    sign_x = -1.0 if sign_mode & 1 else 1.0
    sign_y = -1.0 if sign_mode & 2 else 1.0
    fx, fy = (
        (base.fiedler_b, base.fiedler_a)
        if swap_axes
        else (base.fiedler_a, base.fiedler_b)
    )
    x = base.harmonic_x + sign_x * spread_strength * outline_side * fx
    y = base.harmonic_y + sign_y * spread_strength * outline_side * fy
    return SpectralEmbedding(
        x=_clip_embedding_axis(x, blocks, outline_side),
        y=_clip_embedding_axis(y, blocks, outline_side),
        harmonic_x=base.harmonic_x,
        harmonic_y=base.harmonic_y,
        fiedler_a=base.fiedler_a,
        fiedler_b=base.fiedler_b,
    )


def _choose_rows(
    blocks: BlockData,
    embedding: SpectralEmbedding,
    outline_side: int,
    rng: np.random.Generator,
    variant: int,
) -> tuple[list[list[int]], np.ndarray]:
    """高度递减 best-fit shelf，并用谱坐标决定同行和行间次序。

    先将矩形旋转为横向，按行高递减处理，可避免局部装满行宽却产生大量
    行高浪费。谱坐标只参与可行行的选择与行内次序，不牺牲轮廓可行性。
    """

    n = blocks.n
    rotated = blocks.width < blocks.height
    item_width = np.where(rotated, blocks.height, blocks.width).astype(np.int32)
    item_height = np.where(rotated, blocks.width, blocks.height).astype(np.int32)
    rows: list[list[int]] = []
    row_width: list[int] = []
    row_height: list[int] = []
    row_y_sum: list[float] = []

    jitter = rng.normal(0.0, 1e-3, n)
    order = np.lexsort((embedding.y + jitter, -item_height.astype(float)))
    spectral_weights = (0.0, 0.20, 0.45, 0.80, 1.30, 2.20, 3.60, 6.00)
    spectral_weight = spectral_weights[variant % len(spectral_weights)]

    for raw_block in order:
        block = int(raw_block)
        w = int(item_width[block])
        h = int(item_height[block])
        if w > outline_side:
            raise ValueError(f"模块 {blocks.names[block]} 无法放入边长 {outline_side} 的轮廓")

        choices: list[tuple[float, int]] = []
        for row_id, width in enumerate(row_width):
            if width + w > outline_side:
                continue
            residual = (outline_side - width - w) / max(1.0, outline_side)
            mean_y = row_y_sum[row_id] / max(1, len(rows[row_id]))
            spectral_cost = abs(mean_y - float(embedding.y[block])) / max(
                1.0, outline_side
            )
            choices.append((residual + spectral_weight * spectral_cost, row_id))

        if choices:
            _, row_id = min(choices)
            rows[row_id].append(block)
            row_width[row_id] += w
            row_y_sum[row_id] += float(embedding.y[block])
        else:
            rows.append([block])
            row_width.append(w)
            row_height.append(h)
            row_y_sum.append(float(embedding.y[block]))

    # 在不破坏行宽和总高度可行性的前提下，用跨行交换恢复谱 y 邻近性。
    packed_rows = sorted(
        zip(rows, row_width, row_height, row_y_sum),
        key=lambda item: float(np.mean(embedding.y[item[0]])),
    )
    rows = [item[0] for item in packed_rows]
    row_width = [item[1] for item in packed_rows]
    row_height = [item[2] for item in packed_rows]
    row_y_sum = [item[3] for item in packed_rows]
    refinement_moves = (variant % 8) * max(250, 8 * n)
    for _ in range(refinement_moves):
        if len(rows) < 2:
            break
        a, b = rng.choice(len(rows), size=2, replace=False)
        a, b = int(a), int(b)
        ia = int(rng.integers(len(rows[a])))
        ib = int(rng.integers(len(rows[b])))
        block_a = rows[a][ia]
        block_b = rows[b][ib]
        if block_a == block_b:
            continue

        new_width_a = row_width[a] - int(item_width[block_a]) + int(item_width[block_b])
        new_width_b = row_width[b] - int(item_width[block_b]) + int(item_width[block_a])
        if new_width_a > outline_side or new_width_b > outline_side:
            continue

        centers: list[float] = []
        base = 0.0
        for height in row_height:
            centers.append(base + height / 2.0)
            base += height
        old_cost = abs(float(embedding.y[block_a]) - centers[a]) + abs(
            float(embedding.y[block_b]) - centers[b]
        )
        new_cost = abs(float(embedding.y[block_b]) - centers[a]) + abs(
            float(embedding.y[block_a]) - centers[b]
        )
        if new_cost + 1e-12 >= old_cost:
            continue

        trial_a = rows[a].copy()
        trial_b = rows[b].copy()
        trial_a[ia], trial_b[ib] = block_b, block_a
        new_height_a = max(int(item_height[i]) for i in trial_a)
        new_height_b = max(int(item_height[i]) for i in trial_b)
        new_total_height = (
            sum(row_height)
            - row_height[a]
            - row_height[b]
            + new_height_a
            + new_height_b
        )
        if new_total_height > outline_side:
            continue

        rows[a], rows[b] = trial_a, trial_b
        row_width[a], row_width[b] = new_width_a, new_width_b
        row_height[a], row_height[b] = new_height_a, new_height_b
        row_y_sum[a] = float(np.sum(embedding.y[rows[a]]))
        row_y_sum[b] = float(np.sum(embedding.y[rows[b]]))

    # 行按谱 y 排列；每行让最高模块成为根，其余模块按谱 x 排列。
    rows.sort(key=lambda row: float(np.mean(embedding.y[row])))
    for idx, row in enumerate(rows):
        heights = [
            int(blocks.width[b] if rotated[b] else blocks.height[b]) for b in row
        ]
        root_pos = int(np.argmax(heights))
        root_block = row[root_pos]
        rest = row[:root_pos] + row[root_pos + 1 :]
        rest.sort(key=lambda b: float(embedding.x[b]))
        rows[idx] = [root_block] + rest

    return rows, rotated


def create_spectral_shelf_tree(
    blocks: BlockData,
    embedding: SpectralEmbedding,
    outline_side: int,
    rng: np.random.Generator,
    variant: int = 0,
) -> BStarTreeState:
    rows, rotated = _choose_rows(blocks, embedding, outline_side, rng, variant)
    n = blocks.n
    parent = np.full(n, -1, dtype=np.int32)
    left = np.full(n, -1, dtype=np.int32)
    right = np.full(n, -1, dtype=np.int32)
    module_at_node = np.empty(n, dtype=np.int32)

    next_node = 0
    row_roots: list[int] = []
    for row in rows:
        node_ids = list(range(next_node, next_node + len(row)))
        next_node += len(row)
        row_roots.append(node_ids[0])
        for node, block in zip(node_ids, row):
            module_at_node[node] = int(block)
        for u, v in zip(node_ids, node_ids[1:]):
            left[u] = v
            parent[v] = u

    for upper, lower in zip(row_roots[1:], row_roots[:-1]):
        right[lower] = upper
        parent[upper] = lower

    return BStarTreeState(
        root=int(row_roots[0]),
        parent=parent,
        left=left,
        right=right,
        module_at_node=module_at_node,
        rotated=rotated,
    )


def create_spectral_corner_tree(
    blocks: BlockData,
    embedding: SpectralEmbedding,
    outline_side: int,
    rng: np.random.Generator,
    variant: int = 0,
) -> BStarTreeState:
    """按谱目标坐标选择 B*-Tree 空孩子槽，形成角点式候选拓扑。"""

    n = blocks.n
    if n == 0:
        raise ValueError("模块集合不能为空")
    if variant % 3 == 0:
        rotated = blocks.width < blocks.height
    elif variant % 3 == 1:
        rotated = blocks.width > blocks.height
    else:
        rotated = (rng.random(n) < 0.5)

    actual_w = np.where(rotated, blocks.height, blocks.width).astype(float)
    actual_h = np.where(rotated, blocks.width, blocks.height).astype(float)
    normalized_x = embedding.x / max(1.0, float(outline_side))
    normalized_y = embedding.y / max(1.0, float(outline_side))
    jitter = rng.normal(0.0, 1e-4, n)
    order = np.lexsort((normalized_x + jitter, normalized_y + jitter))

    parent = np.full(n, -1, dtype=np.int32)
    left = np.full(n, -1, dtype=np.int32)
    right = np.full(n, -1, dtype=np.int32)
    module_at_node = np.empty(n, dtype=np.int32)
    module_at_node[0] = int(order[0])
    placed_nodes = [0]

    for node, raw_block in enumerate(order[1:], start=1):
        block = int(raw_block)
        choices: list[tuple[float, int, int]] = []
        for candidate_parent in placed_nodes:
            parent_block = int(module_at_node[candidate_parent])
            if int(left[candidate_parent]) == -1:
                expected_x = embedding.x[parent_block] + (
                    actual_w[parent_block] + actual_w[block]
                ) / 2.0
                expected_y = embedding.y[parent_block]
                score = (
                    abs(float(embedding.x[block] - expected_x))
                    + 0.6 * abs(float(embedding.y[block] - expected_y))
                ) / max(1.0, outline_side)
                choices.append((score, candidate_parent, 0))
            if int(right[candidate_parent]) == -1:
                expected_x = embedding.x[parent_block]
                expected_y = embedding.y[parent_block] + (
                    actual_h[parent_block] + actual_h[block]
                ) / 2.0
                score = (
                    0.6 * abs(float(embedding.x[block] - expected_x))
                    + abs(float(embedding.y[block] - expected_y))
                ) / max(1.0, outline_side)
                choices.append((score, candidate_parent, 1))
        _, chosen_parent, side = min(choices, key=lambda item: item[0])
        module_at_node[node] = block
        parent[node] = chosen_parent
        if side == 0:
            left[chosen_parent] = node
        else:
            right[chosen_parent] = node
        placed_nodes.append(node)

    return BStarTreeState(
        root=0,
        parent=parent,
        left=left,
        right=right,
        module_at_node=module_at_node,
        rotated=rotated,
    )
