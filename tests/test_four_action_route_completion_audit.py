from __future__ import annotations

import pandas as pd

from experiments.audit_route_conditioned_completion import audit_action_cells


def test_completion_audit_recomputes_factorial_effects_and_taxonomy():
    states = {
        "IGNORE": (1.0, True),
        "READ_ONLY": (0.5, False),
        "WRITE_ONLY": (0.75, True),
        "FULL": (0.0, False),
    }
    effects = {
        "read_w0": -0.5,
        "write_r0": -0.25,
        "read_w1": -0.75,
        "write_r1": -0.5,
        "interaction": -0.25,
    }
    rows = []
    for action, (margin, correct) in states.items():
        rows.append(
            {
                "uid": "u",
                "target_layer": 1,
                "action": action,
                "anchor_route_mask": [1, 0, 1],
                "anchor_off_count": 1,
                "fixed_correct_target_text": "right",
                "fixed_wrong_target_text": "wrong",
                "read_on": action in {"READ_ONLY", "FULL"},
                "write_on": action in {"WRITE_ONLY", "FULL"},
                "new_evaluation": action != "IGNORE",
                "correct": correct,
                "S_correct": margin - 2.0,
                "S_original_full_wrong": -2.0,
                "margin": margin,
                "taxonomy": "read_mediated",
                **effects,
            }
        )

    audit = audit_action_cells(pd.DataFrame(rows))

    assert audit["passed"]
    assert audit["cell_count"] == 1
    assert audit["maximum_factorial_formula_abs_error"] == 0.0
