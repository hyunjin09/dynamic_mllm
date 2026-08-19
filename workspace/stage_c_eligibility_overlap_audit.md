# Stage C TextVQA Eligibility and Overlap Audit

Status: **passed and frozen outcome-blind on 2026-08-05**. No Stage C
intervention likelihood or primary endpoint was computed or inspected.

## Candidate Pool and Frozen Selection

- Source: `lmms-lab/textvqa`, official `validation` split, dataset revision
  `9c0699cd19768ac5ab97568f6b3cbac4c0062884`.
- Source fingerprint: `475bf9de899d571b`; source candidates: 5,000.
- Frozen invalid rules: image unavailable/unreadable; missing question; missing
  accepted answers; failed official answer normalization; invalid frequency
  weights; prompt-construction failure; empty or misaligned answer-token span;
  prompt longer than the validated 4,861-token domain; invalid image-token
  indices; or pinned processor/tokenizer failure.
- Selection: sort eligible records by ascending
  `SHA256("20260805:" + record_id)`, then retain the first record for each new
  image ID until 800 records are selected.
- Result: 4,991 eligible records across 3,162 unique images. The frozen
  manifest contains exactly 800 records and 800 unique images.

## Technical Exclusions

Exactly nine records failed predeclared technical rules. No new rule was
introduced.

| Reason | Count | Question IDs |
|---|---:|---|
| Accepted-answer normalization produced no nonempty answer | 4 | 34820, 35165, 36153, 39028 |
| Prompt exceeded 4,861 tokens | 5 | 35389, 36283, 36747, 36759, 36760 |

Every retained record passed image readability, question/answer availability,
normalization and weight checks, prompt construction, answer-span alignment,
context length, contiguous image-token indexing, and pinned processor/tokenizer
processing. Accepted-answer weights are positive and sum to one within `1e-9`.
In the selected manifest, prompt lengths range from 483 to 1,414 tokens,
image-token counts from 444 to 1,369, unique normalized accepted-answer counts
from 1 to 10, and answer lengths from 1 to 38 tokens. All 800 per-record
checksums reproduce exactly.

## Stage B Overlap Audit

The 200 frozen Stage B TextVQA records were compared by record ID, question ID,
annotation ID, image ID, canonical RGB-pixel SHA-256, normalized image path,
normalized question text, and image-question pair.

| Check in the selected 800 | Count |
|---|---:|
| Stage B record/question/annotation overlap | 0 |
| Stage B image-ID overlap | 0 |
| Stage B canonical image-hash overlap | 0 |
| Stage B normalized image-path overlap | 0 |
| Stage B image-question pair overlap | 0 |
| Duplicate Stage C image | 0 |

Across the full validation pool, 87 records shared normalized question text
alone with Stage B. This weak, non-identifying match was reported but not used
as an exclusion: it establishes neither record nor effective-image overlap.
All reliable identifying overlap checks were zero.

## Frozen Artifacts

- Manifest: `outputs/stage_c/manifest/stage_c_manifest_v1.jsonl`
- Manifest SHA-256:
  `e3e9e08329fa626bc75706fba6623357f9ca05140bae1f138c98b9cd26e45357`
- Detailed machine-readable audit:
  `outputs/stage_c/manifest/stage_c_eligibility_overlap_audit_v1.json`
- Selected immutable image copies:
  `/data/dataset/dynamic_mllm/TextVQA/stage_c_validation_images_v1`

The first attempted image-save run used an overconservative question-text-only
overlap rule and was cancelled before producing a manifest. Its 190 partial
image copies are isolated at
`/data/dataset/dynamic_mllm/TextVQA/stage_c_validation_images_v1_cancelled_question_overlap_policy_20260805`
and are not inputs to any frozen artifact.
