import unittest

import torch

from experiments.extract_v3_null_geometry import energy_correlation, row_summary


class V3NullGeometryExtractionTests(unittest.TestCase):
    def test_row_summary_is_finite_and_reports_quantiles(self) -> None:
        summary = row_summary(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
        self.assertEqual(5, len(summary["row_norm_quantiles"]))
        self.assertGreater(summary["frobenius_norm"], 0)

    def test_energy_correlation_accepts_different_native_row_counts(self) -> None:
        read = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        write = torch.arange(18, dtype=torch.float32).reshape(3, 6)
        value = energy_correlation(read, write)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
