from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_q1 import run


def main() -> None:
    parser = argparse.ArgumentParser(description="依次运行 n100/n200/n300 的多个随机种子")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "q1.yaml")
    parser.add_argument("--datasets", nargs="+", default=["n100", "n200", "n300"])
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(s) for s in cfg["multi_start"]["seeds"]]

    rows = []
    for dataset in args.datasets:
        for seed in seeds:
            out = run(dataset, seed, args.config)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            rows.append(summary)

    summary_path = ROOT / "results" / "q1" / "all_runs.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "seed", "n_blocks", "total_block_area", "W", "H",
        "area", "aspect_ratio", "utilization", "dead_space_ratio",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"汇总结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
