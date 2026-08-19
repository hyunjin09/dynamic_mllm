# v3 Minimal Visual-Grounding Control

Status: frozen prospectively from official annotation geometry; no grounding
intervention or held-out terminal score has been run.

## Shared perturbation and comparison

For each eligible record, construct exactly two counterfactual images from the
same decoded RGB source:

- `TARGET_BLUR`: blur the frozen answer-relevant annotation box;
- `MATCHED_NONTARGET_BLUR`: blur the frozen target-sized control rectangle
  centered on the selected non-target annotation.

The two rectangles have exactly equal width and height. Clip neither box;
eligibility requires both to lie within the image. Convert coordinates to the
smallest enclosing integer pixel rectangle. Create a full-image separable
Gaussian blur in float32 RGB using reflect padding, with
`sigma = 0.15 * sqrt(box_width * box_height)` pixels and kernel width
`2 * ceil(3 * sigma) + 1`; copy blurred pixels into the selected rectangle with
a hard rectangular mask and convert back to the source image encoding. Use
Pillow `11.1.0`, torch `2.6.0+cu124`, and torchvision `0.21.0+cu124` for both
conditions. A future technical preflight must checksum both images before
any terminal score is opened.

Let `S_ORIGINAL`, `S_TARGET`, and `S_CONTROL` be the unchanged maximum-over-21
real suppression statistic on the corresponding image. The frozen primary
grounding contrast is

\[
M_i=|S_{TARGET,i}-S_{ORIGINAL,i}|-
    |S_{CONTROL,i}-S_{ORIGINAL,i}|.
\]

The grounding gate passes separately for each dataset only if the image-level
bootstrap 95% CI for the mean `M` lies entirely above zero. As a manipulation
check, also report
`Q_FULL(CONTROL) - Q_FULL(TARGET)` for accepted-reference utility and require
its mean CI to lie above zero. Use 10,000 bootstrap draws and seeds derived as
`SHA256("v3-grounding-v1:<dataset>:<replicate>")`. No subgroup or direction may
be changed after outcomes are visible.

## TextVQA eligibility

Use only records in the prospectively proposed 800-record identity preview for
which:

1. official TextVQA normalization yields at least one accepted answer with
   annotation frequency at least three of ten;
2. exactly one TextOCR v0.1 word annotation normalizes to any such answer;
3. that annotation has a finite, positive-area bounding box;
4. a different OCR annotation whose normalized text is not accepted has source
   area within a multiplicative ratio of `1.25` and target overlap IoU at most
   `0.05`;
5. the target-sized rectangle centered on that annotation is in bounds and has
   target overlap IoU at most `0.05`.

Choose the non-target deterministically by minimum
`log(area_ratio)^2 + log(aspect_ratio)^2`, then annotation ID. The audit finds
130 eligible records, exceeding the frozen minimum of 100.

## GQA eligibility

Use only records in the prospectively proposed 800-record identity preview for
which:

1. the official GQA adapter annotations and semantic program jointly reference
   exactly one numeric GQA scene-graph object ID;
2. the object exists with a finite, positive-area box in the official GQA v1.1
   validation scene graph;
3. a different scene-graph object has source area within a multiplicative
   ratio of `1.25` and target overlap IoU at most `0.05`;
4. the target-sized rectangle centered on that object is in bounds and has
   target overlap IoU at most `0.05`.

Use the same deterministic non-target ranking as TextVQA. The audit finds 123
eligible records, exceeding the frozen minimum of 100.

## Fail-closed rules

The eligible IDs, target boxes, and control boxes are fixed in
`outputs/v3_preflight/grounding_eligibility_audit_v1.json`. Do not substitute a
new box, OCR token, object, record, blur strength, or perturbation method after
terminal outcomes are opened. If image decoding, coordinate conversion,
perturbation serialization, or the minimum count fails in a future preflight,
stop before confirmation rather than use heuristic regions.
