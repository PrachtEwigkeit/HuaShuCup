from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_blocks


class TestParser(unittest.TestCase):
    def test_n100(self):
        blocks = load_blocks(ROOT / "data" / "raw" / "n100.blocks")
        self.assertEqual(blocks.n, 100)
        self.assertEqual(blocks.total_area, 179501)
        self.assertEqual((int(blocks.width[0]), int(blocks.height[0])), (43, 33))

    def test_n200_n300_area(self):
        b200 = load_blocks(ROOT / "data" / "raw" / "n200.blocks")
        b300 = load_blocks(ROOT / "data" / "raw" / "n300.blocks")
        self.assertEqual(b200.n, 200)
        self.assertEqual(b300.n, 300)
        self.assertEqual(b200.total_area, 175696)
        self.assertEqual(b300.total_area, 273170)


if __name__ == "__main__":
    unittest.main()
