from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_q2 import (
    build_fixed_sa_config,
    load_config,
    requested_outline_side,
    solve_dataset,
)
from src.bstar_pack import pack_bstar
from src.data import load_blocks
from src.fixed_io import (
    fixed_summary_dict,
    load_bstar_state,
    save_bstar_state,
    save_fixed_history_csv,
    save_fixed_layout_csv,
    save_q3_search_csv,
)
from src.fixed_outline import (
    FixedOutlineSolution,
    FixedSAHistory,
    WirelengthEvaluator,
    evaluate_fixed_outline,
)
from src.netlist import load_netlist
from src.q3_search import search_minimum_outline
from src.spectral_init import anchored_spectral_embedding
from src.validate import validate_layout, validate_tree
from src.visualize import plot_fixed_convergence, plot_fixed_layout


def _load_or_solve_q2_initial(
    dataset: str,
    seed: int,
    config_path: Path,
) -> tuple[FixedOutlineSolution, object, object, dict, bool]:
    cfg = load_config(config_path)
    state_path = ROOT / "results" / "q2" / dataset / f"seed_{seed}" / "state.npz"
    if not state_path.exists():
        solution, blocks, netlist, cfg = solve_dataset(dataset, seed, config_path)
        return solution, blocks, netlist, cfg, False

    raw = ROOT / "data" / "raw"
    blocks = load_blocks(raw / f"{dataset}.blocks")
    netlist = load_netlist(raw / f"{dataset}.nets", raw / f"{dataset}.pl", blocks)
    state = load_bstar_state(state_path)
    validate_tree(state)
    layout = pack_bstar(blocks, state)
    side = requested_outline_side(
        blocks.total_area, float(cfg.get("dead_space_ratio", 0.15))
    )
    score = evaluate_fixed_outline(layout, netlist, side, WirelengthEvaluator(netlist))
    spectral = cfg.get("spectral", {})
    embedding = anchored_spectral_embedding(
        blocks,
        netlist,
        side,
        spread_strength=float(spectral.get("spread_strength", 0.16)),
        reweight_iterations=int(spectral.get("reweight_iterations", 1)),
    )
    solution = FixedOutlineSolution(
        state=state,
        layout=layout,
        score=score,
        history=FixedSAHistory([], [], [], [], [], [], []),
        embedding=embedding,
        outline_side=side,
    )
    return solution, blocks, netlist, cfg, True


def run(dataset: str, seed: int, config_path: Path) -> Path:
    pipeline_start = time.perf_counter()
    q2_start = time.perf_counter()
    initial, blocks, netlist, cfg, reused_q2_state = _load_or_solve_q2_initial(
        dataset, seed, config_path
    )
    q2_initial_solution_time_seconds = time.perf_counter() - q2_start
    if not initial.score.feasible:
        raise RuntimeError("问题2初始轮廓不可行，无法启动问题3轮廓搜索")

    integer_high = int(math.floor(initial.outline_side + 1e-12))
    # Q3 的 warm start 使用可复现的紧致 B*-Tree 解码，而不是 Q2 的 LP 浮动坐标。
    integer_layout = pack_bstar(blocks, initial.state)
    integer_score = evaluate_fixed_outline(
        integer_layout,
        netlist,
        integer_high,
        WirelengthEvaluator(netlist),
    )
    if not integer_score.feasible:
        raise RuntimeError("问题2状态无法映射到问题3的整数初始边长")
    initial = replace(
        initial,
        layout=integer_layout,
        score=integer_score,
        outline_side=float(integer_high),
        lp_applied=False,
        hpwl_before_lp=None,
    )

    search_cfg = cfg.get("q3_search", {})
    spectral_cfg = cfg.get("spectral", {})
    lp_cfg = cfg.get("elastic_lp", {})
    spread_strengths = tuple(
        float(value)
        for value in spectral_cfg.get(
            "spread_strength_candidates",
            [spectral_cfg.get("spread_strength", 0.16)],
        )
    )
    terminals_inside_outline = bool(
        search_cfg.get("terminals_inside_outline", False)
    )

    q3_start = time.perf_counter()
    result = search_minimum_outline(
        blocks,
        netlist,
        initial,
        build_fixed_sa_config(cfg),
        np.random.default_rng(seed + 100003),
        spectral_variants=int(
            search_cfg.get("spectral_variants", spectral_cfg.get("variants", 8))
        ),
        attempts_per_side=int(search_cfg.get("attempts_per_side", 1)),
        boundary_attempts=int(search_cfg.get("boundary_attempts", 3)),
        boundary_lp_enabled=bool(search_cfg.get("boundary_lp_enabled", False)),
        refine_attempts=int(search_cfg.get("refine_attempts", 1)),
        terminals_inside_outline=terminals_inside_outline,
        probe_lower_bound=bool(search_cfg.get("probe_lower_bound", True)),
        spread_strengths=spread_strengths,
        axis_variants=int(spectral_cfg.get("axis_variants", 8)),
        reweight_iterations=int(spectral_cfg.get("reweight_iterations", 1)),
        lp_time_limit_seconds=float(lp_cfg.get("time_limit_seconds", 20.0)),
    )
    q3_outline_search_time_seconds = time.perf_counter() - q3_start
    total_pipeline_time_seconds = time.perf_counter() - pipeline_start
    solution = result.solution
    validate_tree(solution.state)
    validate_layout(blocks, solution.layout, check_pairs=True)

    out_dir = ROOT / "results" / "q3" / dataset / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fixed_layout_csv(blocks, solution, out_dir / "layout.csv")
    save_bstar_state(solution.state, out_dir / "state.npz")
    save_fixed_history_csv(solution.history, out_dir / "history.csv")
    save_q3_search_csv(result, out_dir / "outline_search.csv")
    summary = fixed_summary_dict(dataset, seed, blocks, solution, None)
    summary.update(
        {
            "minimum_dead_space_ratio": result.dead_space_ratio,
            "lower_bound_side": result.lower_bound_side,
            "first_failed_side": result.first_failed_side,
            "terminals_inside_outline": terminals_inside_outline,
            "minimum_is_heuristic": int(solution.outline_side)
            > result.lower_bound_side,
            "minimum_certified_by_lower_bound": int(solution.outline_side)
            == result.lower_bound_side,
            "q2_state_reused": reused_q2_state,
            "q2_initial_solution_time_seconds": q2_initial_solution_time_seconds,
            "q3_outline_search_time_seconds": q3_outline_search_time_seconds,
            "total_pipeline_time_seconds": total_pipeline_time_seconds,
        }
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_fixed_layout(
        blocks, solution, out_dir / "layout.png", annotate=(blocks.n <= 120)
    )
    plot_fixed_convergence(solution.history, out_dir / "convergence.png")

    print("=" * 64)
    print(f"dataset       : {dataset}")
    print(f"seed          : {seed}")
    print(f"minimum side  : {int(solution.outline_side)}")
    print(f"dead ratio    : {result.dead_space_ratio:.8f}")
    print(f"packed        : {solution.layout.W:.3f} x {solution.layout.H:.3f}")
    print(f"total HPWL    : {solution.score.hpwl:.3f}")
    print(f"q2 state reuse: {reused_q2_state}")
    print(f"q2 warm start : {q2_initial_solution_time_seconds:.3f} s")
    print(f"q3 search     : {q3_outline_search_time_seconds:.3f} s")
    print(f"total pipeline: {total_pipeline_time_seconds:.3f} s")
    print(f"failed side   : {result.first_failed_side}")
    print(f"output        : {out_dir}")
    print("=" * 64)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="华数杯 B 题问题3：最小死区比例搜索")
    parser.add_argument("--dataset", choices=["n100", "n200", "n300"], default="n100")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "q2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    run(args.dataset, seed, args.config)


if __name__ == "__main__":
    main()
