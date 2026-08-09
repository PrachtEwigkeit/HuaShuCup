from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bstar_pack import pack_bstar
from .data import BlockData
from .elastic_lp import optimize_fixed_topology_lp
from .netlist import NetlistData
from .operators import perturb
from .spectral_init import (
    SpectralEmbedding,
    anchored_spectral_embedding,
    create_spectral_corner_tree,
    create_spectral_shelf_tree,
    spectral_embedding_variant,
)
from .structures import BStarTreeState, Layout


@dataclass(frozen=True)
class FixedOutlineScore:
    feasible: bool
    overflow: float
    hpwl: float
    offset_x: float
    offset_y: float
    packed_width: float
    packed_height: float


@dataclass
class FixedSAHistory:
    iteration: list[int]
    current_hpwl: list[float]
    best_hpwl: list[float]
    current_overflow: list[float]
    best_overflow: list[float]
    current_feasible: list[int]
    best_feasible: list[int]


@dataclass(frozen=True)
class FixedSAConfig:
    initial_temperature: float = 0.20
    final_temperature: float = 0.001
    alpha: float = 0.91
    moves_per_temp_factor: float = 1.5
    max_stagnant_levels: int = 8
    outline_penalty: float = 80.0
    infeasible_hpwl_weight: float = 0.02
    p_rotate: float = 0.20
    p_swap: float = 0.20
    p_move: float = 0.10
    p_guided: float = 0.35
    p_row: float = 0.15
    adaptive_operators: bool = True


@dataclass(frozen=True)
class FixedOutlineSolution:
    state: BStarTreeState
    layout: Layout
    score: FixedOutlineScore
    history: FixedSAHistory
    embedding: SpectralEmbedding
    outline_side: float
    lp_applied: bool = False
    hpwl_before_lp: float | None = None


class WirelengthEvaluator:
    """把低度超图填充为矩阵，以向量化计算平移断点和 HPWL。"""

    def __init__(self, netlist: NetlistData):
        max_degree = max(len(net) for net in netlist.nets)
        shape = (netlist.n_nets, max_degree)
        vertices = np.zeros(shape, dtype=np.int32)
        pin_mask = np.zeros(shape, dtype=bool)
        for row, net in enumerate(netlist.nets):
            vertices[row, : len(net)] = net
            pin_mask[row, : len(net)] = True
        self.netlist = netlist
        self.vertices = vertices
        self.pin_mask = pin_mask
        self.block_mask = pin_mask & (vertices < netlist.n_blocks)
        self.terminal_mask = pin_mask & (vertices >= netlist.n_blocks)
        self.mixed_mask = self.block_mask.any(axis=1) & self.terminal_mask.any(axis=1)

    def _gather_axis(
        self, block_axis: np.ndarray, terminal_axis: np.ndarray
    ) -> np.ndarray:
        all_axis = np.concatenate(
            [block_axis.astype(np.float64, copy=False), terminal_axis]
        )
        return all_axis[self.vertices]

    def optimal_axis_translation(
        self,
        block_axis: np.ndarray,
        terminal_axis: np.ndarray,
        slack: float,
    ) -> float:
        if slack <= 0 or not np.any(self.mixed_mask):
            return 0.0
        values = self._gather_axis(block_axis, terminal_axis)
        block_max = np.max(np.where(self.block_mask, values, -np.inf), axis=1)
        block_min = np.min(np.where(self.block_mask, values, np.inf), axis=1)
        terminal_max = np.max(
            np.where(self.terminal_mask, values, -np.inf), axis=1
        )
        terminal_min = np.min(
            np.where(self.terminal_mask, values, np.inf), axis=1
        )
        mixed = self.mixed_mask
        events = np.concatenate(
            [
                terminal_max[mixed] - block_max[mixed],
                terminal_min[mixed] - block_min[mixed],
            ]
        )
        optimum = float(np.median(events))
        return float(np.clip(optimum, 0.0, slack))

    def net_hpwl(
        self,
        block_x: np.ndarray,
        block_y: np.ndarray,
        offset_x: float,
        offset_y: float,
    ) -> np.ndarray:
        x = self._gather_axis(
            block_x + float(offset_x), self.netlist.terminal_x
        )
        y = self._gather_axis(
            block_y + float(offset_y), self.netlist.terminal_y
        )
        x_max = np.max(np.where(self.pin_mask, x, -np.inf), axis=1)
        x_min = np.min(np.where(self.pin_mask, x, np.inf), axis=1)
        y_max = np.max(np.where(self.pin_mask, y, -np.inf), axis=1)
        y_min = np.min(np.where(self.pin_mask, y, np.inf), axis=1)
        return (x_max - x_min) + (y_max - y_min)

    def total_hpwl(
        self,
        block_x: np.ndarray,
        block_y: np.ndarray,
        offset_x: float,
        offset_y: float,
    ) -> float:
        return float(np.sum(self.net_hpwl(block_x, block_y, offset_x, offset_y)))


def _block_centers(layout: Layout) -> tuple[np.ndarray, np.ndarray]:
    return (
        layout.x.astype(np.float64) + layout.width.astype(np.float64) / 2.0,
        layout.y.astype(np.float64) + layout.height.astype(np.float64) / 2.0,
    )


def optimal_translation(
    layout: Layout,
    netlist: NetlistData,
    outline_side: float,
    evaluator: WirelengthEvaluator | None = None,
) -> tuple[float, float]:
    """求给定紧凑布局在固定轮廓内使 HPWL 最小的整体平移。

    对每个坐标轴，总 HPWL 是平移量的凸分段线性函数；每个同时含模块和
    固定端口的网络贡献两个斜率断点，因此中位断点给出全局最优解。
    """

    slack_x = float(outline_side - layout.W)
    slack_y = float(outline_side - layout.H)
    if slack_x < 0 or slack_y < 0:
        return 0.0, 0.0
    center_x, center_y = _block_centers(layout)
    evaluator = evaluator or WirelengthEvaluator(netlist)
    return (
        evaluator.optimal_axis_translation(center_x, netlist.terminal_x, slack_x),
        evaluator.optimal_axis_translation(center_y, netlist.terminal_y, slack_y),
    )


def total_hpwl(
    layout: Layout,
    netlist: NetlistData,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    evaluator: WirelengthEvaluator | None = None,
) -> float:
    center_x, center_y = _block_centers(layout)
    evaluator = evaluator or WirelengthEvaluator(netlist)
    return evaluator.total_hpwl(center_x, center_y, offset_x, offset_y)


def evaluate_fixed_outline(
    layout: Layout,
    netlist: NetlistData,
    outline_side: float,
    evaluator: WirelengthEvaluator | None = None,
) -> FixedOutlineScore:
    overflow_x = max(0.0, float(layout.W) - float(outline_side))
    overflow_y = max(0.0, float(layout.H) - float(outline_side))
    overflow = (overflow_x + overflow_y) / max(1.0, float(outline_side))
    feasible = overflow_x <= 1e-12 and overflow_y <= 1e-12
    evaluator = evaluator or WirelengthEvaluator(netlist)
    dx, dy = (
        optimal_translation(layout, netlist, outline_side, evaluator)
        if feasible
        else (0.0, 0.0)
    )
    hpwl = total_hpwl(layout, netlist, dx, dy, evaluator)
    return FixedOutlineScore(
        feasible=feasible,
        overflow=float(overflow),
        hpwl=float(hpwl),
        offset_x=float(dx),
        offset_y=float(dy),
        packed_width=float(layout.W),
        packed_height=float(layout.H),
    )


def better_fixed(a: FixedOutlineScore, b: FixedOutlineScore) -> bool:
    if a.feasible != b.feasible:
        return a.feasible
    if not a.feasible and not math.isclose(a.overflow, b.overflow):
        return a.overflow < b.overflow
    if not math.isclose(a.hpwl, b.hpwl):
        return a.hpwl < b.hpwl
    return (a.packed_width * a.packed_height) < (b.packed_width * b.packed_height)


def _energy(score: FixedOutlineScore, hpwl_scale: float, config: FixedSAConfig) -> float:
    normalized_hpwl = score.hpwl / max(1.0, hpwl_scale)
    if score.feasible:
        return normalized_hpwl
    return (
        2.0
        + config.outline_penalty * score.overflow
        + config.infeasible_hpwl_weight * normalized_hpwl
    )


def _guided_swap(
    state: BStarTreeState,
    layout: Layout,
    score: FixedOutlineScore,
    netlist: NetlistData,
    block_weights: np.ndarray,
    rng: np.random.Generator,
) -> BStarTreeState:
    """将高连接模块交换到其相邻引脚坐标中位数附近。"""

    candidate = state.copy()
    block = int(rng.choice(state.n, p=block_weights))
    other_x: list[float] = []
    other_y: list[float] = []
    center_x, center_y = _block_centers(layout)
    center_x += score.offset_x
    center_y += score.offset_y

    for net_id in netlist.block_to_nets[block]:
        for vertex in netlist.nets[int(net_id)]:
            vertex = int(vertex)
            if vertex == block:
                continue
            if vertex < netlist.n_blocks:
                other_x.append(float(center_x[vertex]))
                other_y.append(float(center_y[vertex]))
            else:
                terminal = vertex - netlist.n_blocks
                other_x.append(float(netlist.terminal_x[terminal]))
                other_y.append(float(netlist.terminal_y[terminal]))

    if not other_x:
        return candidate

    target_x = float(np.median(other_x))
    target_y = float(np.median(other_y))
    distance = np.abs(center_x - target_x) + np.abs(center_y - target_y)
    distance[block] = np.inf
    partner = int(np.argmin(distance))

    node_of_module = np.empty(state.n, dtype=np.int32)
    node_of_module[state.module_at_node] = np.arange(state.n, dtype=np.int32)
    u = int(node_of_module[block])
    v = int(node_of_module[partner])
    candidate.module_at_node[u], candidate.module_at_node[v] = (
        candidate.module_at_node[v],
        candidate.module_at_node[u],
    )
    return candidate


def _critical_net_guided_swap(
    state: BStarTreeState,
    layout: Layout,
    score: FixedOutlineScore,
    netlist: NetlistData,
    evaluator: WirelengthEvaluator,
    rng: np.random.Generator,
) -> BStarTreeState:
    """优先处理当前 HPWL 大的网络，并选择尺寸相近的交换对象。"""

    candidate = state.copy()
    center_x, center_y = _block_centers(layout)
    center_x += score.offset_x
    center_y += score.offset_y
    net_cost = evaluator.net_hpwl(center_x, center_y, 0.0, 0.0)
    has_block = evaluator.block_mask.any(axis=1)
    weights = np.where(has_block, np.maximum(net_cost, 0.0) ** 1.5, 0.0)
    if float(weights.sum()) <= 1e-12:
        return candidate
    net_id = int(rng.choice(netlist.n_nets, p=weights / weights.sum()))

    pin_x: list[float] = []
    pin_y: list[float] = []
    blocks_in_net: list[int] = []
    for raw_vertex in netlist.nets[net_id]:
        vertex = int(raw_vertex)
        if vertex < netlist.n_blocks:
            blocks_in_net.append(vertex)
            pin_x.append(float(center_x[vertex]))
            pin_y.append(float(center_y[vertex]))
        else:
            terminal = vertex - netlist.n_blocks
            pin_x.append(float(netlist.terminal_x[terminal]))
            pin_y.append(float(netlist.terminal_y[terminal]))
    if not blocks_in_net:
        return candidate

    target_x = float(np.median(pin_x))
    target_y = float(np.median(pin_y))
    block = max(
        blocks_in_net,
        key=lambda item: abs(float(center_x[item]) - target_x)
        + abs(float(center_y[item]) - target_y),
    )
    position_cost = np.abs(center_x - target_x) + np.abs(center_y - target_y)
    source_area = float(layout.width[block] * layout.height[block])
    area = layout.width.astype(float) * layout.height.astype(float)
    size_cost = np.abs(area - source_area) / np.maximum(area, source_area)
    source_ratio = float(layout.width[block]) / max(1.0, float(layout.height[block]))
    ratio = layout.width.astype(float) / np.maximum(1.0, layout.height.astype(float))
    shape_cost = np.abs(np.log(np.maximum(ratio, 1e-12) / source_ratio))
    scale = max(float(layout.W + layout.H), 1.0)
    combined = position_cost / scale + 0.45 * size_cost + 0.15 * shape_cost
    combined[block] = np.inf
    partner = int(np.argmin(combined))

    node_of_module = np.empty(state.n, dtype=np.int32)
    node_of_module[state.module_at_node] = np.arange(state.n, dtype=np.int32)
    u = int(node_of_module[block])
    v = int(node_of_module[partner])
    candidate.module_at_node[u], candidate.module_at_node[v] = (
        candidate.module_at_node[v],
        candidate.module_at_node[u],
    )
    return candidate


def _row_aware_swap(
    state: BStarTreeState,
    layout: Layout,
    rng: np.random.Generator,
) -> BStarTreeState:
    """执行行内相邻交换，或执行尺寸平衡的小扰动交换。"""

    candidate = state.copy()
    if state.n <= 1:
        return candidate
    left_edges = [
        (u, int(state.left[u]))
        for u in range(state.n)
        if int(state.left[u]) != -1
    ]
    if left_edges and rng.random() < 0.65:
        u, v = left_edges[int(rng.integers(len(left_edges)))]
    else:
        u = int(rng.integers(state.n))
        module_u = int(state.module_at_node[u])
        sample_size = min(state.n - 1, 24)
        nodes = np.asarray(
            [node for node in range(state.n) if node != u], dtype=np.int32
        )
        if len(nodes) > sample_size:
            nodes = rng.choice(nodes, size=sample_size, replace=False)
        modules = state.module_at_node[nodes]
        dw = np.abs(
            layout.width[modules].astype(float) - float(layout.width[module_u])
        )
        dh = np.abs(
            layout.height[modules].astype(float) - float(layout.height[module_u])
        )
        v = int(nodes[int(np.argmin(dw + dh))])

    candidate.module_at_node[u], candidate.module_at_node[v] = (
        candidate.module_at_node[v],
        candidate.module_at_node[u],
    )
    return candidate


def simulated_annealing_fixed(
    blocks: BlockData,
    netlist: NetlistData,
    init_state: BStarTreeState,
    outline_side: float,
    config: FixedSAConfig,
    rng: np.random.Generator,
    evaluator: WirelengthEvaluator | None = None,
) -> tuple[BStarTreeState, Layout, FixedOutlineScore, FixedSAHistory]:
    evaluator = evaluator or WirelengthEvaluator(netlist)
    current = init_state.copy()
    current_layout = pack_bstar(blocks, current)
    current_score = evaluate_fixed_outline(
        current_layout, netlist, outline_side, evaluator
    )
    hpwl_scale = max(1.0, current_score.hpwl)

    best = current.copy()
    best_layout = current_layout
    best_score = current_score
    history = FixedSAHistory([], [], [], [], [], [], [])

    operator_names = ("rotate", "swap", "move", "critical", "row")
    base_probabilities = np.asarray(
        [config.p_rotate, config.p_swap, config.p_move, config.p_guided, config.p_row],
        dtype=float,
    )
    if np.any(base_probabilities < 0) or float(base_probabilities.sum()) <= 0:
        raise ValueError("邻域概率非法")
    operator_probabilities = base_probabilities / base_probabilities.sum()

    iteration = 0
    temperature = float(config.initial_temperature)
    moves_per_temp = max(1, int(round(config.moves_per_temp_factor * blocks.n)))
    stagnant_levels = 0

    while (
        temperature > config.final_temperature
        and stagnant_levels < config.max_stagnant_levels
    ):
        improved_this_level = False
        attempts = np.zeros(len(operator_names), dtype=float)
        accepted = np.zeros(len(operator_names), dtype=float)
        improvements = np.zeros(len(operator_names), dtype=float)
        for _ in range(moves_per_temp):
            iteration += 1
            op = int(rng.choice(len(operator_names), p=operator_probabilities))
            attempts[op] += 1.0
            if operator_names[op] == "critical":
                candidate = _critical_net_guided_swap(
                    current,
                    current_layout,
                    current_score,
                    netlist,
                    evaluator,
                    rng,
                )
            elif operator_names[op] == "row":
                candidate = _row_aware_swap(current, current_layout, rng)
            elif operator_names[op] == "rotate":
                candidate = perturb(current, rng, 1.0, 0.0, 0.0)
            elif operator_names[op] == "swap":
                candidate = perturb(current, rng, 0.0, 1.0, 0.0)
            else:
                candidate = perturb(current, rng, 0.0, 0.0, 1.0)

            candidate_layout = pack_bstar(blocks, candidate)
            candidate_score = evaluate_fixed_outline(
                candidate_layout, netlist, outline_side, evaluator
            )
            delta = _energy(candidate_score, hpwl_scale, config) - _energy(
                current_score, hpwl_scale, config
            )
            if delta <= 0:
                accept = True
            else:
                exponent = -delta / max(temperature, 1e-15)
                accept = rng.random() < math.exp(max(exponent, -700.0))

            if accept:
                accepted[op] += 1.0
                current = candidate
                current_layout = candidate_layout
                current_score = candidate_score

            if better_fixed(current_score, best_score):
                best = current.copy()
                best_layout = current_layout
                best_score = current_score
                improved_this_level = True
                improvements[op] += 1.0

            history.iteration.append(iteration)
            history.current_hpwl.append(current_score.hpwl)
            history.best_hpwl.append(best_score.hpwl)
            history.current_overflow.append(current_score.overflow)
            history.best_overflow.append(best_score.overflow)
            history.current_feasible.append(int(current_score.feasible))
            history.best_feasible.append(int(best_score.feasible))

        if config.adaptive_operators:
            denominator = np.maximum(attempts, 1.0)
            quality = (
                1.0
                + 0.25 * accepted / denominator
                + 4.0 * improvements / denominator
            )
            learned = base_probabilities * quality
            learned /= learned.sum()
            operator_probabilities = 0.5 * operator_probabilities + 0.5 * learned
            operator_probabilities /= operator_probabilities.sum()

        stagnant_levels = 0 if improved_this_level else stagnant_levels + 1
        temperature *= config.alpha

    return best, best_layout, best_score, history


def solve_fixed_outline(
    blocks: BlockData,
    netlist: NetlistData,
    outline_side: float,
    config: FixedSAConfig,
    rng: np.random.Generator,
    spectral_variants: int = 6,
    spread_strength: float = 0.16,
    spread_strengths: tuple[float, ...] | None = None,
    axis_variants: int = 8,
    reweight_iterations: int = 1,
    enable_lp_refine: bool = True,
    lp_time_limit_seconds: float = 20.0,
    warm_states: tuple[BStarTreeState, ...] = (),
) -> FixedOutlineSolution:
    evaluator = WirelengthEvaluator(netlist)
    packing_side = int(math.floor(float(outline_side) + 1e-12))
    classic_embedding = anchored_spectral_embedding(
        blocks,
        netlist,
        outline_side=float(outline_side),
        spread_strength=spread_strength,
        reweight_iterations=0,
    )
    embedding = (
        anchored_spectral_embedding(
            blocks,
            netlist,
            outline_side=float(outline_side),
            spread_strength=spread_strength,
            reweight_iterations=reweight_iterations,
        )
        if reweight_iterations > 0
        else classic_embedding
    )

    candidates: list[BStarTreeState] = [state.copy() for state in warm_states]
    strengths = tuple(spread_strengths or (spread_strength,))
    if not strengths or any(value < 0 for value in strengths):
        raise ValueError("谱展开强度必须非负")
    total_variants = max(1, int(spectral_variants))
    classic_count = min(8, total_variants)
    for variant in range(classic_count):
        candidates.append(
            create_spectral_shelf_tree(
                blocks, classic_embedding, packing_side, rng, variant=variant
            )
        )
    for variant in range(classic_count, total_variants):
        extra = variant - classic_count
        axis_mode = extra % max(1, int(axis_variants))
        strength = strengths[
            (extra // max(1, int(axis_variants))) % len(strengths)
        ]
        variant_embedding = spectral_embedding_variant(
            embedding,
            blocks,
            float(outline_side),
            float(strength),
            axis_mode,
        )
        constructor = (
            create_spectral_corner_tree
            if extra % 4 == 3
            else create_spectral_shelf_tree
        )
        candidates.append(
            constructor(
                blocks, variant_embedding, packing_side, rng, variant=variant
            )
        )

    init_state = candidates[0]
    init_layout = pack_bstar(blocks, init_state)
    init_score = evaluate_fixed_outline(
        init_layout, netlist, outline_side, evaluator
    )
    for candidate in candidates[1:]:
        layout = pack_bstar(blocks, candidate)
        score = evaluate_fixed_outline(layout, netlist, outline_side, evaluator)
        if better_fixed(score, init_score):
            init_state, init_layout, init_score = candidate, layout, score

    best_state, best_layout, best_score, history = simulated_annealing_fixed(
        blocks, netlist, init_state, outline_side, config, rng, evaluator
    )
    hpwl_before_lp = float(best_score.hpwl)
    lp_applied = False
    if enable_lp_refine:
        lp_result = optimize_fixed_topology_lp(
            best_layout,
            netlist,
            outline_side,
            time_limit_seconds=lp_time_limit_seconds,
        )
        if lp_result.success:
            lp_score = evaluate_fixed_outline(
                lp_result.layout, netlist, outline_side, evaluator
            )
            if lp_score.feasible and better_fixed(lp_score, best_score):
                best_layout = lp_result.layout
                best_score = lp_score
                lp_applied = True
    return FixedOutlineSolution(
        state=best_state,
        layout=best_layout,
        score=best_score,
        history=history,
        embedding=embedding,
        outline_side=float(outline_side),
        lp_applied=lp_applied,
        hpwl_before_lp=hpwl_before_lp,
    )
