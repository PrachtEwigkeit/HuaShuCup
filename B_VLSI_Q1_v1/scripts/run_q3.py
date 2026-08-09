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

from scripts.run_q2 import build_fixed_sa_config, load_config, solve_dataset
from src.fixed_io import (
    fixed_summary_dict,
    save_fixed_history_csv,
    save_fixed_layout_csv,
    save_q3_search_csv,
)
from src.fixed_outline import WirelengthEvaluator, evaluate_fixed_outline
from src.q3_search import search_minimum_outline
from src.validate import validate_layout, validate_tree
from src.visualize import plot_fixed_convergence, plot_fixed_layout


def run(dataset: str, seed: int, config_path: Path) -> Path:
    pipeline_start = time.perf_counter()
    q2_start = time.perf_counter()
    initial, blocks, netlist, cfg = solve_dataset(dataset, seed, config_path)
    q2_initial_solution_time_seconds = time.perf_counter() - q2_start
    if not initial.score.feasible:
        raise RuntimeError("问题2初始轮廓尚不可行，无法启动问题3缩边搜索")

    # 问题2使用公式给出的实数轮廓；问题3的候选模块包围边长是整数。
    integer_high = int(math.floor(initial.outline_side + 1e-12))
    integer_score = evaluate_fixed_outline(
        initial.layout,
        netlist,
        integer_high,
        WirelengthEvaluator(netlist),
    )
    if not integer_score.feasible:
        raise RuntimeError("问题2解无法映射到问题3的整数初始边长")
    initial = replace(
        initial,
        score=integer_score,
        outline_side=float(integer_high),
    )

    search_cfg = cfg.get("q3_search", {})
    spectral_cfg = cfg.get("spectral", {})
    q3_start = time.perf_counter()
    result = search_minimum_outline(
        blocks,
        netlist,
        initial,
        build_fixed_sa_config(cfg),
        np.random.default_rng(seed + 100003),
        spectral_variants=int(search_cfg.get("spectral_variants", spectral_cfg.get("variants", 4))),
        attempts_per_side=int(search_cfg.get("attempts_per_side", 2)),
        refine_attempts=int(search_cfg.get("refine_attempts", 2)),
        terminals_inside_outline=bool(search_cfg.get("terminals_inside_outline", True)),
        probe_lower_bound=bool(search_cfg.get("probe_lower_bound", True)),
    )
    q3_outline_search_time_seconds = time.perf_counter() - q3_start
    total_pipeline_time_seconds = time.perf_counter() - pipeline_start
    solution = result.solution
    validate_tree(solution.state)
    validate_layout(blocks, solution.layout, check_pairs=True)

    out_dir = ROOT / "results" / "q3" / dataset / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fixed_layout_csv(blocks, solution, out_dir / "layout.csv")
    save_fixed_history_csv(solution.history, out_dir / "history.csv")
    save_q3_search_csv(result, out_dir / "outline_search.csv")
    summary = fixed_summary_dict(dataset, seed, blocks, solution, None)
    summary.update(
        {
            "minimum_dead_space_ratio": result.dead_space_ratio,
            "lower_bound_side": result.lower_bound_side,
            "first_failed_side": result.first_failed_side,
            "terminals_inside_outline": bool(
                search_cfg.get("terminals_inside_outline", True)
            ),
            "minimum_is_heuristic": int(solution.outline_side) > result.lower_bound_side,
            "minimum_certified_by_lower_bound": (
                int(solution.outline_side) == result.lower_bound_side
            ),
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
    print(f"packed        : {solution.layout.W} x {solution.layout.H}")
    print(f"total HPWL    : {solution.score.hpwl:.3f}")
    print(f"q2 warm start : {q2_initial_solution_time_seconds:.3f} s")
    print(f"q3 search     : {q3_outline_search_time_seconds:.3f} s")
    print(f"total pipeline: {total_pipeline_time_seconds:.3f} s")
    print(f"failed side   : {result.first_failed_side}")
    print(f"output        : {out_dir}")
    print("=" * 64)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="华数杯 B题 问题3：最小死区比延续搜索")
    parser.add_argument("--dataset", choices=["n100", "n200", "n300"], default="n100")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "q2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    run(args.dataset, seed, args.config)


if __name__ == "__main__":
    main()
