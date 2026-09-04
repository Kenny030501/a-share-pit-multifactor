import unittest

from ashare_pit import capacity


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.delta = {"A": 0.15, "B": -0.10, "C": -0.05}
        self.adv = {"A": 100_000_000, "B": 80_000_000, "C": 60_000_000}
        self.vol = {"A": 0.25, "B": 0.30, "C": 0.20}

    def test_impact_increases_with_aum(self):
        scenarios = capacity.capacity_scenarios(
            self.delta, self.adv, self.vol, [10_000_000, 100_000_000]
        )
        self.assertGreater(
            scenarios[1]["estimated_nav_impact_fraction"],
            scenarios[0]["estimated_nav_impact_fraction"],
        )
        self.assertGreater(
            scenarios[1]["participation_rate"]["max"],
            scenarios[0]["participation_rate"]["max"],
        )

    def test_missing_adv_is_disclosed(self):
        result = capacity.estimate_rebalance_impact(
            self.delta, {"A": 100_000_000}, self.vol, 10_000_000
        )
        self.assertLess(result["liquidity_coverage"], 1.0)
        self.assertEqual(result["missing_liquidity_names"], ["B", "C"])

    def test_invalid_model_inputs_fail_fast(self):
        with self.assertRaises(ValueError):
            capacity.estimate_rebalance_impact(self.delta, self.adv, self.vol, 0)
        with self.assertRaises(ValueError):
            capacity.estimate_rebalance_impact(
                self.delta, self.adv, self.vol, 10_000_000, impact_coefficient=-0.1
            )


if __name__ == "__main__":
    unittest.main()
