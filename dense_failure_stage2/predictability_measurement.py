"""Pure construction and validation helpers for predictability Step A."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

from scoring.benchmark_metrics import normalize_exact, normalize_textvqa
from scoring.reference_likelihood import weighted_logsumexp


ACTIONS = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")


def accepted_answer_specs(sample: Mapping[str, Any]) -> list[dict[str, float | str]]:
    """Freeze evaluator-grounded reference strings and their aggregation weights.

    GQA uses its exact-match normalization. ChartQA retains the annotated answer
    literally because punctuation can be semantically numeric (for example,
    ``69.79``); relaxed numeric equivalence remains represented by the separate
    LMMS correctness outcome. TextVQA preserves empirical reference multiplicity
    after EvalAI normalization.
    """

    dataset = str(sample["dataset"]).lower()
    if dataset == "gqa":
        text = normalize_exact(str(sample["answer"]))
        if not text:
            raise ValueError("GQA answer is empty after evaluator normalization")
        return [{"text": text, "weight": 1.0}]
    if dataset == "chartqa":
        text = str(sample["answer"]).strip()
        if not text:
            raise ValueError("ChartQA annotated answer is empty")
        return [{"text": text, "weight": 1.0}]
    if dataset == "textvqa":
        raw = list(sample.get("all_answer_norms") or [sample["answer"]])
        normalized = [normalize_textvqa(str(answer)) for answer in raw]
        counts = Counter(answer for answer in normalized if answer)
        if not counts:
            raise ValueError("TextVQA reference set is empty after EvalAI normalization")
        total = sum(counts.values())
        return [
            {"text": text, "weight": count / total}
            for text, count in sorted(counts.items())
        ]
    raise ValueError(f"unsupported internal dataset: {dataset}")


def aggregate_reference_mean_logprobs(
    answers: Sequence[Mapping[str, Any]], mean_logprobs: Sequence[float]
) -> float:
    if len(answers) != len(mean_logprobs):
        raise ValueError("reference answers and scores must align")
    return weighted_logsumexp(
        [float(value) for value in mean_logprobs],
        [float(answer["weight"]) for answer in answers],
    )


def build_p90_trigger_rows(
    rows: Sequence[Mapping[str, Any]], *, threshold: float, layers: int = 28
) -> list[dict[str, Any]]:
    """Select one P90 score row per UID and recompute strict trigger identity."""

    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("threshold_name") not in (None, "P90"):
            continue
        uid = str(row["uid"])
        if uid in selected:
            raise ValueError(f"duplicate P90 UID: {uid}")
        selected[uid] = row
    output: list[dict[str, Any]] = []
    for uid, row in sorted(selected.items()):
        scores = [float(row[f"p_{layer}"]) for layer in range(layers)]
        hits = [layer for layer, score in enumerate(scores) if score > float(threshold)]
        first = hits[0] if hits else None
        output.append(
            {
                "uid": uid,
                "dataset": str(row["dataset"]),
                "source_regime": str(row["source_regime"]),
                "dense_correct": bool(row["dense_correct"]),
                "dense_wrong": not bool(row["dense_correct"]),
                "threshold_name": "P90",
                "threshold": float(threshold),
                "comparison": "strict_greater_than",
                "scores": scores,
                "max_score": max(scores),
                "triggered": first is not None,
                "first_trigger_layer": first,
                "score_at_trigger": None if first is None else scores[first],
                "post_trigger_state_count": 0 if first is None else layers - first,
            }
        )
    return output


def derive_utility_row(branches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(branches) != set(ACTIONS):
        raise ValueError(f"branches must be exactly {ACTIONS}")
    q = {action: float(branches[action]["mean_logprob"]) for action in ACTIONS}
    if not all(math.isfinite(value) for value in q.values()):
        raise ValueError("branch likelihoods must be finite")
    c = {action: bool(branches[action]["correct"]) for action in ACTIONS}
    u_read_w1 = q["FULL"] - q["WRITE_ONLY"]
    u_read_w0 = q["READ_ONLY"] - q["IGNORE"]
    u_write_r1 = q["FULL"] - q["READ_ONLY"]
    u_write_r0 = q["WRITE_ONLY"] - q["IGNORE"]
    u_read = 0.5 * (u_read_w1 + u_read_w0)
    u_write = 0.5 * (u_write_r1 + u_write_r0)
    interaction = q["FULL"] - q["READ_ONLY"] - q["WRITE_ONLY"] + q["IGNORE"]
    best = max(ACTIONS, key=lambda action: (q[action], -ACTIONS.index(action)))
    result = {
        **{f"q_{action.lower()}": q[action] for action in ACTIONS},
        **{f"c_{action.lower()}": c[action] for action in ACTIONS},
        "u_read_w1": u_read_w1,
        "u_read_w0": u_read_w0,
        "u_write_r1": u_write_r1,
        "u_write_r0": u_write_r0,
        "u_read": u_read,
        "u_write": u_write,
        "u_interaction": interaction,
        "exp_u_read_w1": math.exp(u_read_w1),
        "exp_u_read_w0": math.exp(u_read_w0),
        "exp_u_write_r1": math.exp(u_write_r1),
        "exp_u_write_r0": math.exp(u_write_r0),
        "best_action_by_q": best,
        "best_q": q[best],
        "full_gap": q[best] - q["FULL"],
        "local_rescue_exists": (not c["FULL"]) and any(c[a] for a in ACTIONS[1:]),
        "local_regression_exists": c["FULL"] and any(not c[a] for a in ACTIONS[1:]),
        "all_four_correct": all(c.values()),
        "all_four_wrong": not any(c.values()),
        "correct_action_count": sum(c.values()),
        "read_harmful_flip_w1": (not c["FULL"]) and c["WRITE_ONLY"],
        "read_beneficial_flip_w1": c["FULL"] and (not c["WRITE_ONLY"]),
        "read_harmful_flip_w0": (not c["READ_ONLY"]) and c["IGNORE"],
        "read_beneficial_flip_w0": c["READ_ONLY"] and (not c["IGNORE"]),
        "write_harmful_flip_r1": (not c["FULL"]) and c["READ_ONLY"],
        "write_beneficial_flip_r1": c["FULL"] and (not c["READ_ONLY"]),
        "write_harmful_flip_r0": (not c["WRITE_ONLY"]) and c["IGNORE"],
        "write_beneficial_flip_r0": c["WRITE_ONLY"] and (not c["IGNORE"]),
    }
    return result


def validate_complete_state_results(
    expected_state_ids: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    expected = [str(value) for value in expected_state_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected state IDs contain duplicates")
    observed = [str(row["state_id"]) for row in rows]
    duplicates = sorted(uid for uid, count in Counter(observed).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate state results: {duplicates[:3]}")
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(f"missing={missing[:3]} extra={extra[:3]}")
    for row in rows:
        if tuple(row["branches"]) != ACTIONS:
            raise ValueError(f"state {row['state_id']} does not contain exactly four branches")
