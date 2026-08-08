from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_blocks
from src.netlist import load_netlist


class TestNetlistParser(unittest.TestCase):
    def test_n100_counts_and_ranges(self):
        raw = ROOT / "data" / "raw"
        blocks = load_blocks(raw / "n100.blocks")
        netlist = load_netlist(raw / "n100.nets", raw / "n100.pl", blocks)
        self.assertEqual(netlist.n_blocks, 100)
        self.assertEqual(netlist.n_terminals, 334)
        self.assertEqual(netlist.n_nets, 885)
        self.assertEqual(netlist.n_pins, 1873)
        self.assertEqual(float(netlist.terminal_x.min()), 0.0)
        self.assertEqual(float(netlist.terminal_x.max()), 444.0)
        self.assertTrue(all(len(net) >= 2 for net in netlist.nets))


if __name__ == "__main__":
    unittest.main()
