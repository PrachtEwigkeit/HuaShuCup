from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import BlockData
from .fixed_outline import (
    FixedOutlineSolution,
    FixedSAConfig,
    better_fixed,
    solve_fixed_outline,
)
from .netlist import NetlistData


@dataclass(frozen=True)
class OutlineSearchRecord:
    outline_side: int
    dead_space_ratio: float
    feasible: bool
    overflow: float
    hpwl: float
    attempts: int


@dataclass(frozen=True)
class MinimumOutlineResult:
    solution: FixedOutlineSolution
    dead_space_ratio: float
    lower_bound_side: int
    first_failed_side: int | None
    records: tuple[OutlineSearchRecord, ...]


def integer_side_lower_bound(
    blocks: BlockData,
    netlist: NetlistData,
    terminals_inside_outline: bool = False,
) -> int:
    """正方形整数边长下界。

    赛题正式口径下 Terminal 只参与 HPWL，不参与面积、重叠或轮廓约束，
    因此默认不使用端口坐标。保留可选参数仅用于其他数据口径。
    """

    side = int(math.ceil(math.sqrt(blocks.total_area)))
    side = max(side, int(np.max(np.maximum(blocks.width, blocks.height))))
    if terminals_inside_outline:
        max_terminal = max(
            float(np.max(netlist.terminal_x)),
            float(np.max(netlist.terminal_y)),
        )
        min_terminal = min(
            float(np.min(netlist.terminal_x)),
            float(np.min(netlist.terminal_y)),
        )
        if min_terminal < 0:
            raise ValueError("固定端口存在负坐标，当前正方形轮廓模型无法包含")
        side = max(side, int(math.ceil(max_terminal)))
    return side


def _record(
    side: int,
    blocks: BlockData,
    solution: FixedOutlineSolution,
    attempts: int,
) -> OutlineSearchRecord:
    return OutlineSearchRecord(
        outline_side=int(side),
        dead_space_ratio=float(side**2 / blocks.total_area - 1.0),
        feasible=bool(solution.score.feasible),
        overflow=float(solution.score.overflow),
        hpwl=float(solution.score.hpwl),
        attempts=int(attempts),
    )


def search_minimum_outline(
    blocks: BlockData,
    netlist: NetlistData,
    initial_solution: FixedOutlineSolution,
    config: FixedSAConfig,
    rng: np.random.Generator,
    spectral_variants: int = 8,
    attempts_per_side: int = 1,
    boundary_attempts: int = 3,
    boundary_lp_enabled: bool = False,
    refine_attempts: int = 1,
    terminals_inside_outline: bool = False,
    probe_lower_bound: bool = True,
    spread_strengths: tuple[float, ...] = (0.08, 0.12, 0.16, 0.20, 0.24),
    axis_variants: int = 8,
    reweight_iterations: int = 1,
    lp_time_limit_seconds: float = 20.0,
) -> MinimumOutlineResult:
    """以下界直接探测为主、启发式二分为后的最小轮廓搜索。"""

    if not initial_solution.score.feasible:
        raise ValueError("问题3需要从一个问题2可行解开始")
    if attempts_per_side <= 0 or boundary_attempts <= 0 or refine_attempts < 0:
        raise ValueError("搜索尝试次数必须为正")

    lower = integer_side_lower_bound(
        blocks, netlist, terminals_inside_outline=terminals_inside_outline
    )
    # 精修使用独立随机流，保证增加/减少边长探测次数不会改变同边长 HPWL 结果。
    refine_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
    initial_side = int(math.floor(float(initial_solution.outline_side) + 1e-12))
    current = initial_solution
    records: list[OutlineSearchRecord] = [
        _record(initial_side, blocks, initial_solution, 1)
    ]
    first_failed_side: int | None = None

    def solve_side(
        side: int,
        warm: FixedOutlineSolution,
        attempts_limit: int,
        enable_lp: bool = False,
    ) -> tuple[FixedOutlineSolution, int]:
        best: FixedOutlineSolution | None = None
        used = 0
        for attempt in range(attempts_limit):
            used += 1
            solution = solve_fixed_outline(
                blocks,
                netlist,
                side,
                config,
                rng,
                spectral_variants=spectral_variants,
                spread_strengths=spread_strengths,
                axis_variants=axis_variants,
                reweight_iterations=reweight_iterations,
                enable_lp_refine=enable_lp,
                lp_time_limit_seconds=lp_time_limit_seconds,
                warm_states=(warm.state,) if attempt == 0 else (),
            )
            if best is None or better_fixed(solution.score, best.score):
                best = solution
            # 最小边长的可行性已经确定，剩余预算留给定边长 HPWL 精修。
            if solution.score.feasible:
                break
        assert best is not None
        return best, used

    hit_lower_bound = initial_side == lower
    if probe_lower_bound and lower < initial_side:
        lower_solution, used = solve_side(lower, current, attempts_per_side)
        records.append(_record(lower, blocks, lower_solution, used))
        if lower_solution.score.feasible:
            current = lower_solution
            hit_lower_bound = True
        else:
            first_failed_side = lower

    if not hit_lower_bound:
        # SA 失败不是严格不可行证明，因此这里称为启发式二分。
        low_failed = lower
        high_feasible = initial_side
        high_solution = initial_solution
        while high_feasible - low_failed > 1:
            side = (low_failed + high_feasible) // 2
            candidate, used = solve_side(side, high_solution, attempts_per_side)
            records.append(_record(side, blocks, candidate, used))
            if candidate.score.feasible:
                high_feasible = side
                high_solution = candidate
            else:
                low_failed = side
                first_failed_side = side
        current = high_solution

        # 只在二分边界附近增加预算，避免所有中间边长都重复运行。
        while int(current.outline_side) > lower:
            boundary_side = int(current.outline_side) - 1
            rescued, used = solve_side(
                boundary_side,
                current,
                boundary_attempts,
                enable_lp=boundary_lp_enabled,
            )
            records.append(_record(boundary_side, blocks, rescued, used))
            if not rescued.score.feasible:
                first_failed_side = boundary_side
                break
            current = rescued

    # 固定已找到的最小边长，用临界网络局部搜索和 LP 空白重分配优化 HPWL。
    for _ in range(refine_attempts):
        refined = solve_fixed_outline(
            blocks,
            netlist,
            int(current.outline_side),
            config,
            refine_rng,
            spectral_variants=spectral_variants,
            spread_strengths=spread_strengths,
            axis_variants=axis_variants,
            reweight_iterations=reweight_iterations,
            enable_lp_refine=True,
            lp_time_limit_seconds=lp_time_limit_seconds,
            warm_states=(current.state,),
        )
        if refined.score.feasible and better_fixed(refined.score, current.score):
            current = refined

    side = int(current.outline_side)
    dead_ratio = side**2 / blocks.total_area - 1.0
    return MinimumOutlineResult(
        solution=current,
        dead_space_ratio=float(dead_ratio),
        lower_bound_side=lower,
        first_failed_side=first_failed_side,
        records=tuple(records),
    )
