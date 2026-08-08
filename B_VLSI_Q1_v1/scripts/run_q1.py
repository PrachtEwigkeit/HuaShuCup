from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annealing import SAConfig, simulated_annealing
from src.bstar_init import create_initial_tree
from src.data import load_blocks
from src.io_utils import save_history_csv, save_layout_csv, save_summary_json
from src.validate import validate_layout, validate_tree
from src.visualize import plot_convergence, plot_layout


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_sa_config(cfg: dict) -> SAConfig:
    sa = cfg["sa"]
    op = cfg["operators"]
    return SAConfig(
        initial_temperature=float(sa["initial_temperature"]),
        final_temperature=float(sa["final_temperature"]),
        alpha=float(sa["alpha"]),
        moves_per_temp_factor=float(sa["moves_per_temp_factor"]),
        max_stagnant_levels=int(sa["max_stagnant_levels"]),
        shape_scale=float(sa.get("shape_scale", 0.01)),
        p_rotate=float(op["rotate"]),
        p_swap=float(op["swap"]),
        p_move=float(op["move"]),
    )


def run(dataset: str, seed: int, config_path: Path) -> Path:
    cfg = load_config(config_path)
    blocks_path = ROOT / "data" / "raw" / f"{dataset}.blocks"
    blocks = load_blocks(blocks_path)

    rng = np.random.default_rng(seed)
    init_method = cfg.get("init", {}).get("method", "area_sorted")
    init_state = create_initial_tree(blocks, rng, method=init_method)
    validate_tree(init_state)

    sa_cfg = build_sa_config(cfg)
    best_state, best_layout, best_score, history = simulated_annealing(
        blocks, init_state, sa_cfg, rng
    )

    validate_tree(best_state)
    validate_layout(blocks, best_layout, check_pairs=True)

    out_dir = ROOT / "results" / "q1" / dataset / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_layout_csv(blocks, best_layout, out_dir / "layout.csv")
    save_history_csv(history, out_dir / "history.csv")
    save_summary_json(dataset, seed, blocks, best_layout, best_score, out_dir / "summary.json")
    plot_layout(blocks, best_layout, out_dir / "layout.png", annotate=(blocks.n <= 120))
    plot_convergence(history, out_dir / "convergence.png")

    print("=" * 60)
    print(f"dataset     : {dataset}")
    print(f"seed        : {seed}")
    print(f"n_blocks    : {blocks.n}")
    print(f"block area  : {blocks.total_area}")
    print(f"W x H       : {best_layout.W} x {best_layout.H}")
    print(f"area        : {best_score.area}")
    print(f"aspect      : {best_score.aspect:.6f}")
    print(f"output      : {out_dir}")
    print("=" * 60)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="华数杯 B题 第一问：B*-Tree + SA 第一版")
    parser.add_argument("--dataset", choices=["n100", "n200", "n300"], default="n100")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "q1.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    run(args.dataset, seed, args.config)


if __name__ == "__main__":
    main()
