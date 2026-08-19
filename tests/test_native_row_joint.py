import unittest

import torch

from nulls.native_row_joint import (
    final_native_projection_error,
    fit_native_row_joint_model,
    generate_native_row_joint_null,
)


class NativeRowJointTests(unittest.TestCase):
    def test_native_generation_is_deterministic_and_norm_matched(self):
        generator = torch.Generator().manual_seed(3)
        reads = [torch.randn(5 + index % 3, 12, generator=generator) for index in range(12)]
        writes = [torch.randn(7 + index % 2, 12, generator=generator) for index in range(12)]
        model = fit_native_row_joint_model(
            reads,
            writes,
            rows_per_sample=4,
            variance_target=0.7,
            maximum_rank=6,
            position_bins=3,
            shrinkage=0.1,
            seed=11,
        )
        read_norms = reads[0].norm(dim=1)
        write_norms = writes[0].norm(dim=1)
        first = generate_native_row_joint_null(model, read_norms, write_norms, 13)
        second = generate_native_row_joint_null(model, read_norms, write_norms, 13)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.allclose(first[0].norm(dim=1), read_norms, atol=1e-5))
        self.assertTrue(torch.allclose(first[1].norm(dim=1), write_norms, atol=1e-5))

    def test_projection_error_is_zero_for_spanned_rows(self):
        values = [torch.eye(4)[: 2 + index % 2] for index in range(8)]
        model = fit_native_row_joint_model(
            values,
            values,
            rows_per_sample=4,
            variance_target=1.0,
            maximum_rank=4,
            position_bins=2,
            shrinkage=0.1,
            seed=4,
        )
        self.assertLess(final_native_projection_error(values[0], model.read), 1e-5)


if __name__ == "__main__":
    unittest.main()
