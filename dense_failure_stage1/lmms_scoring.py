"""Thin adapter around the installed LMMS-Eval task implementations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Sequence

from lmms_eval.api.metrics import exact_match_hf_evaluate
from lmms_eval.tasks.chartqa.utils import chartqa_process_results
from lmms_eval.tasks.textvqa.utils import textvqa_process_results


_BINARY_THRESHOLDS = {
    "gqa": 1.0,
    "chartqa": 1.0,
    # Existing convention used to construct the repository's 8K buckets.
    "textvqa": 0.5,
}


@dataclass(frozen=True)
class LmmsSampleScore:
    metric_name: str
    raw_score: float
    normalized_prediction: str | None
    correctness_threshold: float
    correct: bool


def score_lmms_sample(
    *,
    dataset: str,
    prediction: str,
    answer: str,
    answers: Sequence[str] | None,
    uid: str,
) -> LmmsSampleScore:
    """Return the official LMMS-Eval per-sample task result and frozen label."""

    dataset = dataset.lower()
    if dataset == "gqa":
        payload = exact_match_hf_evaluate(
            predictions=[prediction],
            references=[answer],
            ignore_case=True,
            ignore_punctuation=True,
        )
        metric_name = "exact_match"
        raw_score = float(payload[metric_name])
        normalized_prediction = None
    elif dataset == "chartqa":
        payload = chartqa_process_results(
            {"answer": answer, "type": "human_test"}, [prediction]
        )
        metric_name = "relaxed_overall"
        raw_score = float(payload[metric_name])
        normalized_prediction = None
    elif dataset == "textvqa":
        if not answers:
            raise ValueError(f"{uid} has no TextVQA reference answers")
        payload = textvqa_process_results(
            {
                "answers": [str(value) for value in answers],
                "question_id": uid,
            },
            [prediction],
        )
        metric_name = "exact_match"
        raw_score = float(payload[metric_name])
        normalized_prediction = str(payload["submission"]["answer"])
    else:
        raise ValueError(f"unsupported LMMS-Eval Stage-1 dataset: {dataset!r}")

    threshold = _BINARY_THRESHOLDS[dataset]
    return LmmsSampleScore(
        metric_name=metric_name,
        raw_score=raw_score,
        normalized_prediction=normalized_prediction,
        correctness_threshold=threshold,
        correct=raw_score >= threshold,
    )


def lmms_eval_source_metadata() -> dict:
    """Identify the exact installed LMMS-Eval scoring sources."""

    modules = {
        "gqa_exact_match": Path(exact_match_hf_evaluate.__code__.co_filename),
        "chartqa_process_results": Path(chartqa_process_results.__code__.co_filename),
        "textvqa_process_results": Path(textvqa_process_results.__code__.co_filename),
    }
    sources = {}
    for name, path in modules.items():
        sources[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "distribution": "lmms-eval",
        "version": metadata.version("lmms-eval"),
        "sources": sources,
        "binary_thresholds": dict(_BINARY_THRESHOLDS),
    }

