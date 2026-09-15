"""Deterministic scoring for external visual-reasoning heldouts."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Any

from dvr_qwen.eval_metrics import (
    multiple_choice_accuracy,
    reasoning_strict_accuracy,
    score_prediction,
)


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _direct_answer(text: str) -> str:
    value = str(text or "").strip()
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", value)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines:
        value = lines[-1]
    return re.sub(
        r"^(?:final\s+answer|short\s+answer|answer)\s*[:：]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().rstrip(".。"))


def mathvista_deterministic_accuracy(row: dict[str, Any], prediction: str) -> float:
    """Approximate MathVista's normalization without an external answer extractor.

    The heldout prompt requests a direct answer, so this scorer parses the final
    answer span locally. It deliberately does not claim equivalence to the
    official LLM-assisted extraction metric.
    """

    answer = str(row.get("answer") or "").strip()
    extracted = _direct_answer(prediction)
    question_type = str(row.get("question_type") or "")
    answer_type = str(row.get("answer_type") or "")
    choices = [str(value) for value in (row.get("choices") or [])]

    if question_type == "multi_choice":
        letter_match = re.search(r"\b([A-J])\b", extracted.upper())
        if letter_match:
            index = ord(letter_match.group(1)) - ord("A")
            extracted = choices[index] if 0 <= index < len(choices) else ""
        elif _normalized_text(extracted) not in {_normalized_text(choice) for choice in choices}:
            return 0.0
        return float(_normalized_text(extracted) == _normalized_text(answer))

    if answer_type in {"integer", "float"}:
        matches = _NUMBER.findall(extracted.replace(",", ""))
        if not matches:
            return 0.0
        try:
            prediction_number = Decimal(matches[-1])
            answer_number = Decimal(answer.replace(",", ""))
        except InvalidOperation:
            return 0.0
        if answer_type == "integer":
            return float(int(prediction_number) == int(answer_number))
        precision = int(float(row.get("precision") or 0))
        quantum = Decimal(1).scaleb(-precision)
        return float(prediction_number.quantize(quantum) == answer_number.quantize(quantum))

    return reasoning_strict_accuracy(extracted, answer)


def score_heldout_prediction(row: dict[str, Any], prediction: str) -> float:
    metric = str(row["metric_name"]).lower()
    if metric == "mathvista_deterministic_accuracy":
        return mathvista_deterministic_accuracy(row, prediction)
    if metric == "mathverse_deterministic_accuracy":
        if str(row.get("question_type") or "") == "multi-choice":
            return multiple_choice_accuracy(prediction, str(row.get("answer") or ""))
        return reasoning_strict_accuracy(prediction, str(row.get("answer") or ""))
    if metric == "wemath_choice_accuracy":
        return multiple_choice_accuracy(prediction, str(row.get("answer") or ""))
    return score_prediction(
        metric,
        prediction,
        row.get("answer"),
        row.get("all_answer_norms"),
    )
