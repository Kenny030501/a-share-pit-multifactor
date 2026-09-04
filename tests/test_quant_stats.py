import random
import unittest

from ashare_pit import quant_stats as stats


class QuantStatsTests(unittest.TestCase):
    def test_newey_west_detects_persistent_positive_mean(self):
        rng = random.Random(7)
        values = []
        previous = 0.0
        for _ in range(240):
            previous = 0.55 * previous + 0.02 + rng.gauss(0, 0.03)
            values.append(previous)
        result = stats.newey_west_mean_test(values)
        self.assertGreater(result["t_stat"], 2.0)
        self.assertLess(result["p_value"], 0.05)
        self.assertGreater(result["lags"], 0)

    def test_block_bootstrap_interval_preserves_clear_signal(self):
        values = [0.03 + (i % 5 - 2) * 0.001 for i in range(80)]
        result = stats.moving_block_bootstrap_mean_ci(values, samples=500, seed=3)
        self.assertGreater(result["ci"][0], 0)
        self.assertLess(result["ci"][0], result["ci"][1])

    def test_benjamini_hochberg_is_monotone_and_controls_fdr(self):
        result = stats.benjamini_hochberg({"a": 0.001, "b": 0.01, "c": 0.2})
        self.assertTrue(result["a"]["significant"])
        self.assertTrue(result["b"]["significant"])
        self.assertFalse(result["c"]["significant"])
        self.assertLessEqual(result["a"]["q_value"], result["b"]["q_value"])


if __name__ == "__main__":
    unittest.main()
