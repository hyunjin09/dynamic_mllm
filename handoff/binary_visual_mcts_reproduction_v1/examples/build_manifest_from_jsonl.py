#!/usr/bin/env python3
"""Example adapter for a dataset already exported as one JSON object per line.

This adapter is intentionally simple. Replace it when the benchmark needs
dataset-specific prompt construction, answer aggregation, or image lookup.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--correctness-threshold", required=True, type=float)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--uid-field", default="id")
    parser.add_argument("--image-field", default="image_path")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--answers-field", default="all_answer_norms")
    parser.add_argument("--image-group-field", default="image_id")
    parser.add_argument("--prompt-field")
    parser.add_argument("--prompt-suffix", default="\nAnswer the question using a single word or phrase.")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    image_root = Path(args.image_root).resolve()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_rows = []
    seen = set()
    for index, raw in enumerate(rows):
        sample_id = str(raw[args.uid_field])
        uid = f"{args.benchmark}:{sample_id}"
        if uid in seen:
            raise ValueError(f"duplicate uid: {uid}")
        seen.add(uid)
        image_path = (image_root / str(raw[args.image_field])).resolve()
        question = str(raw[args.question_field])
        prompt = str(raw[args.prompt_field]) if args.prompt_field else question + args.prompt_suffix
        image_group = raw.get(args.image_group_field, raw[args.image_field])
        output_rows.append(
            {
                "uid": uid,
                "sample_id": sample_id,
                "benchmark": args.benchmark,
                "question": question,
                "prompt": prompt,
                "answer": str(raw[args.answer_field]),
                "all_answer_norms": raw.get(args.answers_field),
                "metric_name": args.metric_name,
                "correctness_threshold": args.correctness_threshold,
                "max_new_tokens": args.max_new_tokens,
                "historical_all_on_status": None,
                "image_group_id": str(image_group),
                "local_image_path": str(image_path),
                "image_content_sha256": sha256(image_path.read_bytes()).hexdigest(),
                "source_file": str(source),
                "source_index": index,
                "source_row_sha256": sha256(
                    json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
                "max_image_tokens": None,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(output_rows), "sha256": sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
