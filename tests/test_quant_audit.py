import tempfile
import unittest
from pathlib import Path

from ashare_pit import quant_audit


ROOT = Path(__file__).resolve().parents[1]


class QuantAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = quant_audit.build_audit(ROOT / "results")

    def test_walk_forward_and_cost_findings_match_saved_evidence(self):
        self.assertEqual(self.audit["walk_forward"]["beat_benchmark_folds"], "1/3")
        self.assertLess(self.audit["walk_forward"]["mean_generalization_gap_percent"], 0)
        self.assertGreater(self.audit["execution"]["realistic_active_improvement_percent"], 0)

    def test_factor_audit_has_multiple_testing_results(self):
        rows = self.audit["factor_significance"]
        self.assertGreaterEqual(len(rows), 9)
        self.assertTrue(all("q_value_bh" in row for row in rows))
        screen = self.audit["gtja191_multiple_testing_audit"]
        self.assertEqual(screen["factors_with_minimum_history"], 70)
        self.assertGreater(screen["significant_fdr_5pct_independence_approximation"], 0)

    def test_report_and_svg_generation(self):
        report = quant_audit.render_markdown(self.audit)
        self.assertIn("Walk-forward 泛化", report)
        with tempfile.TemporaryDirectory() as directory:
            paths = quant_audit.write_assets(self.audit, Path(directory))
            self.assertEqual(len(paths), 3)
            for path in paths:
                self.assertTrue(path.read_text(encoding="utf-8").startswith("<svg"))


if __name__ == "__main__":
    unittest.main()
