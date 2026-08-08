from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bstar_init import create_initial_tree
from src.bstar_pack import pack_bstar
from src.data import load_blocks
from src.validate import validate_layout, validate_tree


class TestPack(unittest.TestCase):
    def test_random_pack_is_legal(self):
        blocks = load_blocks(ROOT / "data" / "raw" / "n100.blocks")
        rng = np.random.default_rng(123)
        state = create_initial_tree(blocks, rng, method="random")
        validate_tree(state)
        layout = pack_bstar(blocks, state)
        validate_layout(blocks, layout, check_pairs=True)
        self.assertGreaterEqual(layout.area, blocks.total_area)


if __name__ == "__main__":
    unittest.main()
