import unittest

from tools.research_analysis.v3.audit_null_redesign_donors import distance_components
from nulls.joint_four_action import PairedGeometryMetadata, paired_geometry_distance


def metadata(sample_id: str, scale: float) -> PairedGeometryMetadata:
    return PairedGeometryMetadata(
        sample_id=sample_id,
        image_id=sample_id,
        dataset="gqa",
        layer=0,
        read_norm=scale,
        write_norm=2 * scale,
        read_rows=10,
        write_rows=20,
        image_tokens=20,
        prompt_tokens=30,
        read_scale_ratio=1,
        write_scale_ratio=1,
        read_row_cv=1,
        write_row_cv=1,
    )


class DonorGeometryTests(unittest.TestCase):
    def test_components_reproduce_frozen_max_distance(self):
        target = metadata("a", 1.0)
        donor = metadata("b", 1.25)
        components = distance_components(target, donor)
        self.assertAlmostEqual(max(components.values()), paired_geometry_distance(target, donor))
        self.assertEqual(max(components, key=components.get), "read_norm")


if __name__ == "__main__":
    unittest.main()
