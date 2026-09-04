import random
import unittest

from ashare_pit import gtja191_screening as screening


class GTJAScreeningTests(unittest.TestCase):
    def test_screening_applies_robust_statistics_and_fdr(self):
        rng = random.Random(11)
        factor_ics = {
            "signal": [0.04 + rng.gauss(0, 0.02) for _ in range(60)],
            "noise": [0.04 if i % 2 == 0 else -0.04 for i in range(60)],
        }
        rows = screening.summarize_factor_ics(factor_ics)
        by_name = {row["factor"]: row for row in rows}
        self.assertIn("newey_west", by_name["signal"])
        self.assertIn("moving_block_bootstrap", by_name["signal"])
        self.assertTrue(by_name["signal"]["multiple_testing"]["significant"])
        self.assertFalse(by_name["noise"]["multiple_testing"]["significant"])


if __name__ == "__main__":
    unittest.main()
