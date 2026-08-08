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


def anchored_spectral_embedding(
    blocks: BlockData,
    netlist: NetlistData,
    outline_side: float,
    spread_strength: float = 0.16,
    regularization: float = 1e-3,
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

    half_min = np.minimum(blocks.width, blocks.height).astype(float) / 2.0
    x = np.clip(x, half_min, outline_side - half_min)
    y = np.clip(y, half_min, outline_side - half_min)
    return SpectralEmbedding(x=x, y=y, harmonic_x=harmonic_x, harmonic_y=harmonic_y)


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
