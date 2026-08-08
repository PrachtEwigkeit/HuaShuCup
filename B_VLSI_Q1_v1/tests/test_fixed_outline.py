from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import BlockData, load_blocks
from src.fixed_outline import evaluate_fixed_outline, optimal_translation, total_hpwl
from src.netlist import NetlistData, load_netlist
from src.q3_search import integer_side_lower_bound
from src.structures import Layout


class TestFixedOutline(unittest.TestCase):
    def test_exact_translation_for_terminal_net(self):
        layout = Layout(
            x=np.asarray([0], dtype=np.int32),
            y=np.asarray([0], dtype=np.int32),
            width=np.asarray([2], dtype=np.int32),
            height=np.asarray([2], dtype=np.int32),
            rotated=np.asarray([False]),
            W=2,
            H=2,
            area=4,
        )
        netlist = NetlistData(
            nets=(np.asarray([0, 1], dtype=np.int32),),
            terminal_names=("p1",),
            terminal_x=np.asarray([9.0]),
            terminal_y=np.asarray([9.0]),
            n_blocks=1,
            block_to_nets=(np.asarray([0], dtype=np.int32),),
        )
        dx, dy = optimal_translation(layout, netlist, outline_side=10)
        self.assertEqual((dx, dy), (8.0, 8.0))
        self.assertEqual(total_hpwl(layout, netlist, dx, dy), 0.0)
        score = evaluate_fixed_outline(layout, netlist, 10)
        self.assertTrue(score.feasible)
        self.assertEqual(score.hpwl, 0.0)

    def test_terminal_lower_bounds(self):
        raw = ROOT / "data" / "raw"
        expected = {"n100": 444, "n200": 438, "n300": 548}
        for dataset, side in expected.items():
            blocks = load_blocks(raw / f"{dataset}.blocks")
            netlist = load_netlist(
                raw / f"{dataset}.nets", raw / f"{dataset}.pl", blocks
            )
            self.assertEqual(integer_side_lower_bound(blocks, netlist, True), side)


if __name__ == "__main__":
    unittest.main()
