from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bstar_pack import pack_bstar
from src.data import BlockData
from src.netlist import NetlistData
from src.spectral_init import anchored_spectral_embedding, create_spectral_shelf_tree
from src.validate import validate_layout, validate_tree


class TestSpectralInit(unittest.TestCase):
    def test_small_anchored_tree_is_legal(self):
        blocks = BlockData(
            names=["b0", "b1", "b2", "b3"],
            width=np.asarray([3, 2, 4, 2], dtype=np.int32),
            height=np.asarray([2, 4, 2, 3], dtype=np.int32),
        )
        netlist = NetlistData(
            nets=(
                np.asarray([0, 1, 4], dtype=np.int32),
                np.asarray([1, 2], dtype=np.int32),
                np.asarray([2, 3, 5], dtype=np.int32),
            ),
            terminal_names=("p0", "p1"),
            terminal_x=np.asarray([0.0, 10.0]),
            terminal_y=np.asarray([0.0, 10.0]),
            n_blocks=4,
            block_to_nets=(
                np.asarray([0], dtype=np.int32),
                np.asarray([0, 1], dtype=np.int32),
                np.asarray([1, 2], dtype=np.int32),
                np.asarray([2], dtype=np.int32),
            ),
        )
        embedding = anchored_spectral_embedding(blocks, netlist, outline_side=10)
        state = create_spectral_shelf_tree(
            blocks, embedding, 10, np.random.default_rng(3)
        )
        validate_tree(state)
        layout = pack_bstar(blocks, state)
        validate_layout(blocks, layout, check_pairs=True)
        self.assertTrue(np.all(np.isfinite(embedding.x)))
        self.assertTrue(np.all(np.isfinite(embedding.y)))


if __name__ == "__main__":
    unittest.main()
