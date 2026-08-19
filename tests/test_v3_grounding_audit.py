import unittest

from tools.research_analysis.v3.audit_grounding_controls import box_iou, matched_control


class V3GroundingAuditTests(unittest.TestCase):
    def test_iou_is_zero_for_disjoint_boxes(self) -> None:
        self.assertEqual(0.0, box_iou([0, 0, 10, 10], [20, 20, 5, 5]))

    def test_control_has_exact_target_area_and_is_nontarget(self) -> None:
        control = matched_control(
            [0, 0, 10, 20],
            [("other", [40, 40, 11, 19], "label")],
            width=100,
            height=100,
        )
        self.assertIsNotNone(control)
        self.assertEqual([40.5, 39.5, 10, 20], control["equal_area_control_box"])
        self.assertEqual(200, control["equal_area_control_box"][2] * control["equal_area_control_box"][3])


if __name__ == "__main__":
    unittest.main()
