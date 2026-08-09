from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_q2 import load_config, run as run_q2
from scripts.run_q3 import run as run_q3


DATASETS = ("n100", "n200", "n300")


def _load_summary(question: str, dataset: str, seed: int) -> dict:
    path = ROOT / "results" / question / dataset / f"seed_{seed}" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def write_runtime_summary(seed: int) -> Path:
    path = ROOT / "results" / "solve_times.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "dataset",
                "seed",
                "solve_time_seconds",
                "q2_initial_solution_time_seconds",
                "q3_outline_search_time_seconds",
                "total_pipeline_time_seconds",
            ],
        )
        writer.writeheader()
        for dataset in DATASETS:
            q2 = _load_summary("q2", dataset, seed)
            writer.writerow(
                {
                    "question": "q2",
                    "dataset": dataset,
                    "seed": seed,
                    "solve_time_seconds": q2["solve_time_seconds"],
                    "q2_initial_solution_time_seconds": "",
                    "q3_outline_search_time_seconds": "",
                    "total_pipeline_time_seconds": q2["solve_time_seconds"],
                }
            )
            q3 = _load_summary("q3", dataset, seed)
            writer.writerow(
                {
                    "question": "q3",
                    "dataset": dataset,
                    "seed": seed,
                    "solve_time_seconds": q3["q3_outline_search_time_seconds"],
                    "q2_initial_solution_time_seconds": q3[
                        "q2_initial_solution_time_seconds"
                    ],
                    "q3_outline_search_time_seconds": q3[
                        "q3_outline_search_time_seconds"
                    ],
                    "total_pipeline_time_seconds": q3["total_pipeline_time_seconds"],
                }
            )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="依次运行问题2和问题3的全部数据集，并汇总求解时间"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "q2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)

    for dataset in DATASETS:
        run_q2(dataset, seed, args.config)
    for dataset in DATASETS:
        run_q3(dataset, seed, args.config)

    summary_path = write_runtime_summary(seed)
    print(f"runtime summary: {summary_path}")


if __name__ == "__main__":
    main()
