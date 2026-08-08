from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bstar_init import create_initial_tree
from src.data import load_blocks
from src.operators import perturb
from src.validate import validate_tree


class TestOperators(unittest.TestCase):
    def test_many_perturbations_keep_tree_valid(self):
        blocks = load_blocks(ROOT / "data" / "raw" / "n100.blocks")
        rng = np.random.default_rng(7)
        state = create_initial_tree(blocks, rng, method="random")
        for _ in range(1000):
            state = perturb(state, rng, 0.3, 0.35, 0.35)
            validate_tree(state)


if __name__ == "__main__":
    unittest.main()
