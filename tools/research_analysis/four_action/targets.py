from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from reference.dvr_qwen.eval_metrics import _TEXTVQA_PROCESSOR, score_prediction


@dataclass(frozen=True)
class AnswerTarget:
    text: str
    evaluator_score: float
    normalized_key: str
    source_count: int


def accepted_answer_targets(record: dict[str, Any]) -> list[AnswerTarget]:
    """Freeze evaluator-valid correct strings before intervention outcomes.

    TextVQA references are grouped by official EvalAI normalization.  Each
    group receives one deterministic representative: most frequent exact raw
    form, then shortest, then lexical.  Only groups whose representative meets
    the record's frozen correctness threshold are retained.
    """
    if record["dataset"] != "textvqa":
        answer = str(record["answer"]).strip()
        if not answer:
            raise ValueError(f"{record['dataset']} answer target is empty")
        text = f"<answer>{answer}</answer>" if record["dataset"] == "wemath2pro" else answer
        evaluator_score = float(
            score_prediction(
                record["metric_name"],
                text,
                record["answer"],
                record.get("all_answer_norms"),
            )
        )
        if evaluator_score < float(record["correctness_threshold"]):
            raise ValueError(f"canonical answer target is not evaluator-correct for {record['uid']}")
        return [AnswerTarget(text, evaluator_score, text.lower(), 1)]
    references = [str(value).strip() for value in (record.get("all_answer_norms") or [record["answer"]])]
    references = [value for value in references if value]
    if not references:
        raise ValueError("TextVQA reference list is empty")
    groups: dict[str, list[str]] = defaultdict(list)
    for reference in references:
        groups[str(_TEXTVQA_PROCESSOR(reference))].append(reference)
    targets = []
    for normalized, values in sorted(groups.items()):
        counts = Counter(values)
        representative = min(counts, key=lambda value: (-counts[value], len(value), value))
        evaluator_score = score_prediction(
            record["metric_name"], representative, record["answer"], references
        )
        if evaluator_score >= float(record["correctness_threshold"]):
            targets.append(
                AnswerTarget(
                    text=representative,
                    evaluator_score=float(evaluator_score),
                    normalized_key=normalized,
                    source_count=len(values),
                )
            )
    if not targets:
        raise ValueError(f"no evaluator-valid TextVQA reference for {record['uid']}")
    return targets


def answer_targets_are_scorable(record: dict[str, Any]) -> bool:
    """Return whether the frozen evaluator admits at least one correct target."""
    try:
        accepted_answer_targets(record)
    except ValueError:
        return False
    return True


def full_wrong_target(record: dict[str, Any]) -> AnswerTarget:
    text = str(record["full_prediction"]).strip()
    if not text:
        raise ValueError(f"FULL generated wrong answer is empty for {record['uid']}")
    score = score_prediction(
        record["metric_name"], text, record["answer"], record.get("all_answer_norms")
    )
    if score >= float(record["correctness_threshold"]):
        raise ValueError(f"FULL wrong target is evaluator-correct for {record['uid']}")
    normalized = str(_TEXTVQA_PROCESSOR(text)) if record["dataset"] == "textvqa" else text.lower()
    return AnswerTarget(text=text, evaluator_score=float(score), normalized_key=normalized, source_count=1)
