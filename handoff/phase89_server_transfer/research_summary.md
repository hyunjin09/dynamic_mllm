# Phase89 research state

The purpose of this phase is to certify an existing3B route corpus before
studying representation geometry. Old labels may change across runtimes, and
an old Wrong-Dense→Correct-Route case can become a preservation case if Dense
changes. Geometry cohorts must use a verified, consistent runtime's Dense and
routed labels. Corpus certification itself makes no representation claim.

The corpus has10,000 samples(GQA4,000; ChartQA/DocVQA/TextVQA2,000 each),9,273
image-content groups and3,907,717 unique36-layer binary visual-on/off routes.
Every sample has exactly one all-FULL anchor. There are no duplicate or unknown
source labels. Routes/sample mean390.7717, median400, p90407, maximum409.
Saved route labels are1,880,345Correct and2,027,372Wrong; these are historical.
This3B binary domain is distinct from Phase88's7B READ/WRITE action study.

## Established on the stopped H100 server

- Initial Gate A2939 failed31/32. On `textvqa:textvqa_19417`, native HF produced
 22(wrong), custom all-FULL23(correct). That was an executor parity failure;
 full replay was correctly blocked.
- Diagnostic2941 measured the first divergence at layer0 prefill SDPA output,
 max absolute difference0.0078125. Input states, RoPE and canonical Q/K/V were
 exact. Native implicit causal-mask/GQA dispatch reproduced the native output
 when applied to identical custom Q/K/V. This supports a call-convention repair,
 not a claim that differing Torch versions caused the mismatch.
- The V2 repair uses native mask handling for unpadded FULL prefill while
 preserving padded and text-only behavior. Complete gate2942 passed32/32;
 no divergence remained over72 recorded failing-anchor boundaries. Both8worker
 full-stage preflights and the later24worker preflight passed.
- All10,000 Dense/FULL samples completed. Current Dense is5,702Correct and
 4,298Wrong. Old→current:5,607C→C;125C→W;95W→C;4,173W→W. There are220
 label changes(2.2%). These measurements apply to this source environment.
- The3-process/GPU trial passed all9,323 route outputs across24 completed
 ChartQA samples exactly. Pilot throughput34.52routes/s; expensive-first
 DocVQA production initially17.53routes/s. Historical same-sample timings
 suggested1.90x throughput, with CPU allocation also changing32→96. This is
 not a controlled comparison to the earlier2-process attempt mentioned by the
 user, and those rates do not predict another server reliably.

## Exact terminal evidence and remaining uncertainty

At packaging,734,481 durable routed-stream records independently verify their
record hashes, source bindings, unique identity and unchanged Dense reuse.
There are1,858complete routed samples,31partial,8,111without a routed file,
and zero saved error records. Separate Dense records are complete for all10,000.
The union contains742,592unique available route results;3,165,125fresh routed
inferences remain, subject to future destination compatibility decisions.

Coverage is biased: routed streams contain720,540ChartQA records and13,941DocVQA
records, with no GQA or TextVQA routed streams yet. Sparse parity probes and
Dense labels cover those benchmarks, but do not replace their full route replay.
Do not interpret this prefix as the full corpus's stability or cohort distribution.

The user cancelled2994 and its report2957. Cancellation is not an out-of-memory
or parity failure. The earlier2956 step's terminal failure/cancellation occurred
during the intentional switch to2994. Preserve these events as execution
history; no automatic restart is authorized. There is no final full-routed
transition matrix, RS-A/B/C category or READY/NOT READY decision.

The plan has not established that correct routes converge, wrong routes diverge,
C/W representations separate, or that the corpus is ready for geometry. No
finetuning, preference-pair rebuilding or hidden-state corpus capture ran.
Transient diagnostic tensors were summarized, not stored as a geometry corpus.

## Contract and environment identity

| Artifact | Canonical SHA256 |
|---|---|
| Original V2 diagnostic contract |e4d74e8a3c2d47e7d9fb46fc0654d4cdc7cf293fdacb1cd7c7e539178793e0e1|
| Parity repair contract |fd85605441447e5219e7bd6a8a8845b9eb38e1ec7429711da9b8866d03bace40|
| Dense/routed inference contract |0e096f78a05a5f139fc9976931c22328493beb9cc966f2b60334ed83f6ccac47|
| Three-process scheduling overlay |9a4f24328f8b86d22d6e97d0257a3fe42c3948e4ff23df7e12f90089c1495292|
| Certified current Dense manifest |0fe3091b7d2853b7983eda17ee4bcc51e258204c79d24a2a5b4686aa03b5e094|

Model snapshot66285546d2b821cf421d4f5eb2576359d3770cd3,36layers,hidden2048.
Local Python3.12.7,Torch2.6.0+cu124,Transformers5.3.0,BF16SDPA,greedy decoding,
repetition penalty1.05,max new tokens16/32 as recorded per sample,EOS151645.
Original corpus Torch2.9.1+cu128 differs; its Python/Transformers versions were
not recorded. Environment differences remain observations, not standalone
root-cause evidence. The handoff now includes full weight-file checksums.

Future last-question-token BF16/FP16 storage for all routes is576,216,317,952
bytes(~536.64GiB); FP32 doubles it. The original raw-visual projection is about
458TB decimal and is not a capture authorization. Extraction indexing and
practical geometry readiness still need separate validation within future scope.

Phase88's source random-control completion and final interpretation remain
unverified. Phase85 remains stopped. Those historical research lines are not
restarted by this server transfer.
