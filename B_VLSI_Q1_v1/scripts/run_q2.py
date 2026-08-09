from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_blocks
from src.fixed_io import (
    save_fixed_history_csv,
    save_fixed_layout_csv,
    save_fixed_summary_json,
)
from src.fixed_outline import FixedOutlineSolution, FixedSAConfig, solve_fixed_outline
from src.netlist import load_netlist
from src.validate import validate_layout, validate_tree
from src.visualize import plot_fixed_convergence, plot_fixed_layout


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_fixed_sa_config(cfg: dict) -> FixedSAConfig:
    sa = cfg["sa"]
    op = cfg["operators"]
    return FixedSAConfig(
        initial_temperature=float(sa["initial_temperature"]),
        final_temperature=float(sa["final_temperature"]),
        alpha=float(sa["alpha"]),
        moves_per_temp_factor=float(sa["moves_per_temp_factor"]),
        max_stagnant_levels=int(sa["max_stagnant_levels"]),
        outline_penalty=float(sa.get("outline_penalty", 80.0)),
        infeasible_hpwl_weight=float(sa.get("infeasible_hpwl_weight", 0.02)),
        p_rotate=float(op["rotate"]),
        p_swap=float(op["swap"]),
        p_move=float(op["move"]),
        p_guided=float(op.get("guided", 0.20)),
    )


def requested_outline_side(total_area: int, dead_space_ratio: float) -> float:
    return float(math.sqrt(total_area * (1.0 + dead_space_ratio)))


def solve_dataset(
    dataset: str,
    seed: int,
    config_path: Path,
) -> tuple[FixedOutlineSolution, object, object, dict]:
    cfg = load_config(config_path)
    raw = ROOT / "data" / "raw"
    blocks = load_blocks(raw / f"{dataset}.blocks")
    netlist = load_netlist(raw / f"{dataset}.nets", raw / f"{dataset}.pl", blocks)
    dead_ratio = float(cfg.get("dead_space_ratio", 0.15))
    side = requested_outline_side(blocks.total_area, dead_ratio)
    spectral = cfg.get("spectral", {})

    solution = solve_fixed_outline(
        blocks,
        netlist,
        side,
        build_fixed_sa_config(cfg),
        np.random.default_rng(seed),
        spectral_variants=int(spectral.get("variants", 6)),
        spread_strength=float(spectral.get("spread_strength", 0.16)),
    )
    return solution, blocks, netlist, cfg


def write_outputs(
    dataset: str,
    seed: int,
    cfg: dict,
    blocks,
    solution: FixedOutlineSolution,
    out_dir: Path,
    solve_time_seconds: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fixed_layout_csv(blocks, solution, out_dir / "layout.csv")
    save_fixed_history_csv(solution.history, out_dir / "history.csv")
    save_fixed_summary_json(
        dataset,
        seed,
        blocks,
        solution,
        out_dir / "summary.json",
        requested_dead_space_ratio=float(cfg.get("dead_space_ratio", 0.15)),
        timings={"solve_time_seconds": solve_time_seconds},
    )
    plot_fixed_layout(
        blocks, solution, out_dir / "layout.png", annotate=(blocks.n <= 120)
    )
    plot_fixed_convergence(solution.history, out_dir / "convergence.png")


def run(dataset: str, seed: int, config_path: Path) -> Path:
    solve_start = time.perf_counter()
    solution, blocks, _, cfg = solve_dataset(dataset, seed, config_path)
    solve_time_seconds = time.perf_counter() - solve_start
    validate_tree(solution.state)
    validate_layout(blocks, solution.layout, check_pairs=True)
    out_dir = ROOT / "results" / "q2" / dataset / f"seed_{seed}"
    write_outputs(
        dataset,
        seed,
        cfg,
        blocks,
        solution,
        out_dir,
        solve_time_seconds,
    )

    print("=" * 64)
    print(f"dataset       : {dataset}")
    print(f"seed          : {seed}")
    print(f"outline       : {solution.outline_side:.6f} x {solution.outline_side:.6f}")
    print(f"packed        : {solution.layout.W} x {solution.layout.H}")
    print(f"translation   : ({solution.score.offset_x:.3f}, {solution.score.offset_y:.3f})")
    print(f"feasible      : {solution.score.feasible}")
    print(f"total HPWL    : {solution.score.hpwl:.3f}")
    print(f"solve time    : {solve_time_seconds:.3f} s")
    print(f"output        : {out_dir}")
    print("=" * 64)
    if not solution.score.feasible:
        raise RuntimeError("在当前搜索预算内未找到问题2可行解，请增大 SA 搜索量")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="华数杯 B题 问题2：谱初始化+B*-Tree+SA")
    parser.add_argument("--dataset", choices=["n100", "n200", "n300"], default="n100")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "q2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    run(args.dataset, seed, args.config)


if __name__ == "__main__":
    main()
