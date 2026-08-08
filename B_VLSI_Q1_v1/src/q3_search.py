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
    terminals_inside_outline: bool = True,
) -> int:
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


def search_minimum_outline(
    blocks: BlockData,
    netlist: NetlistData,
    initial_solution: FixedOutlineSolution,
    config: FixedSAConfig,
    rng: np.random.Generator,
    spectral_variants: int = 4,
    attempts_per_side: int = 2,
    refine_attempts: int = 2,
    terminals_inside_outline: bool = True,
    probe_lower_bound: bool = True,
) -> MinimumOutlineResult:
    if not initial_solution.score.feasible:
        raise ValueError("问题3需要从一个问题2可行解开始")
    if attempts_per_side <= 0 or refine_attempts < 0:
        raise ValueError("搜索尝试次数必须为正")

    lower = integer_side_lower_bound(
        blocks, netlist, terminals_inside_outline=terminals_inside_outline
    )
    current = initial_solution
    records: list[OutlineSearchRecord] = [
        OutlineSearchRecord(
            outline_side=int(current.outline_side),
            dead_space_ratio=int(current.outline_side) ** 2 / blocks.total_area - 1.0,
            feasible=True,
            overflow=current.score.overflow,
            hpwl=current.score.hpwl,
            attempts=1,
        )
    ]
    first_failed_side: int | None = None

    # 先直接探测解析下界；若可行，则下界和构造解共同给出最小值证明。
    hit_lower_bound = int(current.outline_side) == lower
    if probe_lower_bound and lower < int(current.outline_side):
        lower_best: FixedOutlineSolution | None = None
        for attempt in range(attempts_per_side):
            warm = (current.state,) if attempt == 0 else ()
            candidate = solve_fixed_outline(
                blocks,
                netlist,
                lower,
                config,
                rng,
                spectral_variants=spectral_variants,
                warm_states=warm,
            )
            if lower_best is None or better_fixed(candidate.score, lower_best.score):
                lower_best = candidate
        assert lower_best is not None
        if lower_best.score.feasible:
            current = lower_best
            hit_lower_bound = True
            records.append(
                OutlineSearchRecord(
                    outline_side=lower,
                    dead_space_ratio=lower**2 / blocks.total_area - 1.0,
                    feasible=True,
                    overflow=lower_best.score.overflow,
                    hpwl=lower_best.score.hpwl,
                    attempts=attempts_per_side,
                )
            )

    search_start = int(initial_solution.outline_side) - 1
    sides_to_search = range(search_start, lower - 1, -1) if not hit_lower_bound else ()
    for side in sides_to_search:
        best_attempt: FixedOutlineSolution | None = None
        for attempt in range(attempts_per_side):
            warm = (current.state,) if attempt == 0 else ()
            solution = solve_fixed_outline(
                blocks,
                netlist,
                side,
                config,
                rng,
                spectral_variants=spectral_variants,
                warm_states=warm,
            )
            if best_attempt is None or better_fixed(
                solution.score, best_attempt.score
            ):
                best_attempt = solution
            if solution.score.feasible:
                # 其余尝试仍可改善同一最小轮廓下的 HPWL。
                current = solution

        assert best_attempt is not None
        if best_attempt.score.feasible and better_fixed(
            best_attempt.score, current.score
        ):
            current = best_attempt

        records.append(
            OutlineSearchRecord(
                outline_side=side,
                dead_space_ratio=side**2 / blocks.total_area - 1.0,
                feasible=best_attempt.score.feasible,
                overflow=best_attempt.score.overflow,
                hpwl=best_attempt.score.hpwl,
                attempts=attempts_per_side,
            )
        )
        if not best_attempt.score.feasible:
            first_failed_side = side
            break
        current = best_attempt if better_fixed(best_attempt.score, current.score) else current

    # 固定最小可行边长，仅优化该边长下的 HPWL，不允许用更大轮廓换线长。
    for _ in range(refine_attempts):
        refined = solve_fixed_outline(
            blocks,
            netlist,
            int(current.outline_side),
            config,
            rng,
            spectral_variants=spectral_variants,
            warm_states=(current.state,),
        )
        if refined.score.feasible and better_fixed(refined.score, current.score):
            current = refined

    dead_ratio = int(current.outline_side) ** 2 / blocks.total_area - 1.0
    return MinimumOutlineResult(
        solution=current,
        dead_space_ratio=float(dead_ratio),
        lower_bound_side=lower,
        first_failed_side=first_failed_side,
        records=tuple(records),
    )
