from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bstar_pack import pack_bstar
from .data import BlockData
from .netlist import NetlistData
from .operators import perturb
from .spectral_init import (
    SpectralEmbedding,
    anchored_spectral_embedding,
    create_spectral_shelf_tree,
)
from .structures import BStarTreeState, Layout


@dataclass(frozen=True)
class FixedOutlineScore:
    feasible: bool
    overflow: float
    hpwl: float
    offset_x: float
    offset_y: float
    packed_width: int
    packed_height: int


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
    moves_per_temp_factor: float = 2.0
    max_stagnant_levels: int = 24
    outline_penalty: float = 80.0
    infeasible_hpwl_weight: float = 0.02
    p_rotate: float = 0.24
    p_swap: float = 0.25
    p_move: float = 0.31
    p_guided: float = 0.20


@dataclass(frozen=True)
class FixedOutlineSolution:
    state: BStarTreeState
    layout: Layout
    score: FixedOutlineScore
    history: FixedSAHistory
    embedding: SpectralEmbedding
    outline_side: float


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

    def total_hpwl(
        self,
        block_x: np.ndarray,
        block_y: np.ndarray,
        offset_x: float,
        offset_y: float,
    ) -> float:
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
        return float(np.sum((x_max - x_min) + (y_max - y_min)))


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
        packed_width=int(layout.W),
        packed_height=int(layout.H),
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

    degree = np.asarray([len(ids) for ids in netlist.block_to_nets], dtype=float)
    degree = np.maximum(degree, 1.0)
    block_weights = degree / degree.sum()

    random_total = config.p_rotate + config.p_swap + config.p_move
    if random_total <= 0 or config.p_guided < 0:
        raise ValueError("邻域概率非法")

    iteration = 0
    temperature = float(config.initial_temperature)
    moves_per_temp = max(1, int(round(config.moves_per_temp_factor * blocks.n)))
    stagnant_levels = 0

    while (
        temperature > config.final_temperature
        and stagnant_levels < config.max_stagnant_levels
    ):
        improved_this_level = False
        for _ in range(moves_per_temp):
            iteration += 1
            if rng.random() < config.p_guided:
                candidate = _guided_swap(
                    current,
                    current_layout,
                    current_score,
                    netlist,
                    block_weights,
                    rng,
                )
            else:
                candidate = perturb(
                    current,
                    rng,
                    config.p_rotate / random_total,
                    config.p_swap / random_total,
                    config.p_move / random_total,
                )

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
                current = candidate
                current_layout = candidate_layout
                current_score = candidate_score

            if better_fixed(current_score, best_score):
                best = current.copy()
                best_layout = current_layout
                best_score = current_score
                improved_this_level = True

            history.iteration.append(iteration)
            history.current_hpwl.append(current_score.hpwl)
            history.best_hpwl.append(best_score.hpwl)
            history.current_overflow.append(current_score.overflow)
            history.best_overflow.append(best_score.overflow)
            history.current_feasible.append(int(current_score.feasible))
            history.best_feasible.append(int(best_score.feasible))

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
    warm_states: tuple[BStarTreeState, ...] = (),
) -> FixedOutlineSolution:
    evaluator = WirelengthEvaluator(netlist)
    packing_side = int(math.floor(float(outline_side) + 1e-12))
    embedding = anchored_spectral_embedding(
        blocks,
        netlist,
        outline_side=float(outline_side),
        spread_strength=spread_strength,
    )

    candidates: list[BStarTreeState] = [state.copy() for state in warm_states]
    for variant in range(max(1, spectral_variants)):
        candidates.append(
            create_spectral_shelf_tree(
                blocks, embedding, packing_side, rng, variant=variant
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
    return FixedOutlineSolution(
        state=best_state,
        layout=best_layout,
        score=best_score,
        history=history,
        embedding=embedding,
        outline_side=float(outline_side),
    )
