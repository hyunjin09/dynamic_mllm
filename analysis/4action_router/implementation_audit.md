# Online Four-Action Router Implementation Audit

## Scope

This is the separate `plans/four_action_train.md` architecture. It does not
replace, cancel, resume, or consume outputs from the pending upfront POLAR job
1662.

The online router is trained only from GQA, ChartQA, and TextVQA. WeMath
Standard and Pro are excluded by the user's explicit training-data decision.
The best internally selected checkpoint will be evaluated only on ChartQA,
MMMU-Pro Standard/Vision, and POPE.

## Frozen scientific contract

- Backbone: frozen Qwen2.5-VL-7B-Instruct revision `cc594898...` in BF16.
- Executor: unified four-action contract `d8f524b9...`.
- Action order: IGNORE, READ_ONLY, WRITE_ONLY, FULL.
- Router input at layer `l`: the last valid current text/control row and all
  current routed visual rows immediately before layer `l`.
- READ: the current text state plus `e_R[l]` queries current visual K/V.
- WRITE: `e_W[l]` alone queries current visual K/V; the text state conditions
  WRITE only after visual pooling.
- Joint head: READ unary + WRITE unary + a scaled four-way interaction.
- Training state: exact teacher-forced routed prefixes, never all-FULL states
  paired with non-FULL labels.
- Supervision: exact prefix tries and `-log(sum p(valid next action))`.
- Backbone gradients: prohibited; only the 7,621,638 router parameters train.

## Data audit

The checksum-bound manifest contains 6,811 eligible samples: 5,945 train and
866 validation, with 248,804 complete valid routes and 6,490 image groups.
The audit found 5,112,442 exact trie nodes. Valid outgoing-action
multiplicities are 4,886,909 single, 211,698 double, 11,210 triple, and 2,625
four-way nodes. Multi-valid supervision is therefore exercised directly.

Every included sample has the same executor contract and source evaluator
metadata. The source manifest supplies answer aliases, evaluator, threshold,
generation length, and image-content checksum; none of these answer fields are
router inputs.

## Training and validation

- Eight H100s, one frozen Qwen and router replica per GPU, NCCL DDP.
- Physical batch 1/GPU, accumulation 16, exact effective sample batch 128.
- Per epoch: 6,144 balanced draws (1,024 per dataset × route-type cell), 48
  optimizer steps; 10 epochs and 480 steps total.
- One deterministic valid route per sample per epoch; repeated balanced draws
  of the same sample use that same epoch route.
- AdamW, learning rate `5e-4`, weight decay `0.01`, ten-step warmup, cosine
  decay, BF16 router forward, gradient clipping at 1.0.
- A checkpoint is saved after every epoch.
- All 866 validation records undergo both exact teacher-route node evaluation
  and actual online routed execution every epoch.
- Selection is frozen before external evaluation using balanced W2C rescue and
  C2C preservation, then overall routed accuracy, fewer C2C regressions, fewer
  FULL actions, and earlier epoch.

## Fail-closed gates

The first queued job is an eight-sample, eight-GPU real-data smoke. It covers
all three datasets and both W2C/C2C types and checks actual routed-prefix state
changes, READ/WRITE execution flags, multi-valid loss, deterministic route
sampling, finite/decreasing loss, gradients in both function-specific query
tables, a fully frozen backbone, exact checkpoint roundtrip, and the exact
query token/index. Main training requires the checksum-bound passed smoke and
is submitted with an `afterok` dependency. Evaluation similarly depends on
successful training and has its own six-split semantic/determinism preflight.

## CPU verification

Focused router, trie, executor, checkpoint-binding, atomic epoch-resume,
external-evaluation, and merge tests pass. The full repository suite passes
476 tests.
