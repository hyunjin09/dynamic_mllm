from tools.research_analysis.four_action.cohort import (
    CONTROL_NO_CORRECTION,
    CONTROL_VISION_REQUIRED,
    EXCLUDED_ALL_OFF_RESCUE,
    EXCLUDED_OTHER,
    PRIMARY,
    classify_summary,
    evenly_spaced_ids,
    shard_for,
)


def row(status, all_off, correction):
    return {
        "current_all_on_status": status,
        "all_off_correct": all_off,
        "correction_found": correction,
    }


def test_primary_and_control_taxonomy_is_exact():
    assert classify_summary(row("wrong", False, True)) == PRIMARY
    assert classify_summary(row("wrong", False, False)) == CONTROL_NO_CORRECTION
    assert classify_summary(row("correct", False, None)) == CONTROL_VISION_REQUIRED
    assert classify_summary(row("wrong", True, True)) == EXCLUDED_ALL_OFF_RESCUE
    assert classify_summary(row("correct", True, None)) == EXCLUDED_OTHER


def test_even_selection_spans_visual_token_distribution():
    rows = [{"uid": f"u{index}", "visual_token_count": index} for index in range(9)]
    assert evenly_spaced_ids(rows, 4) == ["u0", "u3", "u5", "u8"]


def test_sharding_is_stable_and_bounded():
    assert shard_for("gqa:one") == shard_for("gqa:one")
    assert 0 <= shard_for("textvqa:two") < 8
