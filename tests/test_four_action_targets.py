from tools.research_analysis.four_action.targets import (
    accepted_answer_targets,
    answer_targets_are_scorable,
    full_wrong_target,
)


def base_record(dataset="textvqa"):
    return {
        "uid": f"{dataset}:one",
        "dataset": dataset,
        "answer": "cat",
        "all_answer_norms": ["cat", "cat", "cat", "Cat.", "dog", "bird", "x", "y", "z", "q"],
        "metric_name": "textvqa_evalai_consensus" if dataset == "textvqa" else "exact_match_ignore_case_punctuation",
        "correctness_threshold": 0.5 if dataset == "textvqa" else 1.0,
        "full_prediction": "dog",
    }


def test_textvqa_targets_are_evaluator_valid_and_normalization_grouped():
    targets = accepted_answer_targets(base_record())
    assert len(targets) == 1
    assert targets[0].normalized_key == "cat"
    assert targets[0].text == "cat"
    assert targets[0].evaluator_score >= 0.5
    assert targets[0].source_count == 4


def test_article_only_textvqa_answer_is_not_dropped():
    record = base_record()
    record["answer"] = "a"
    record["all_answer_norms"] = ["a"] * 5 + ["caps lock", "s", "d", "x", "y"]
    assert accepted_answer_targets(record)[0].text == "a"


def test_textvqa_without_consensus_correct_target_is_unscorable():
    record = base_record()
    record["all_answer_norms"] = [f"unique answer {index}" for index in range(10)]

    assert not answer_targets_are_scorable(record)


def test_gqa_uses_canonical_answer_and_full_wrong_is_checked():
    record = base_record("gqa")
    record["all_answer_norms"] = None
    target = accepted_answer_targets(record)
    wrong = full_wrong_target(record)
    assert [item.text for item in target] == ["cat"]
    assert wrong.text == "dog"
    assert wrong.evaluator_score == 0.0


def test_chartqa_and_wemath_targets_use_their_canonical_generation_formats():
    chart = base_record("chartqa")
    chart["metric_name"] = "relaxed_accuracy"
    chart["answer"] = "12"
    assert accepted_answer_targets(chart)[0].text == "12"

    pro = base_record("wemath2pro")
    pro["metric_name"] = "wemath2pro_mathruler_accuracy"
    pro["answer"] = "16"
    assert accepted_answer_targets(pro)[0].text == "<answer>16</answer>"
