V2 diagnostic contract: exact original snapshot, inputs, scoring, BF16, SDPA,
greedy decoding, repetition1.05 and EOS151645. Full baseline uses row token limit.
Instrumented trace limits generation to2 tokens to locate the known first
decision mismatch; it must reproduce the baseline prefix. Source Torch2.9.1+cu128;
source Transformers/Python NOT RECORDED. Current versions are frozen below.
Original32 anchors and four sparse probes are unchanged. Diagnostic changes
no model weights, execution outputs, canonical package or gate criteria.

{
  "attention": "sdpa",
  "do_sample": false,
  "dtype": "bfloat16",
  "eos_token_ids": [
    151645
  ],
  "max_new_tokens": "source row16 or32",
  "num_beams": 1,
  "position_convention": "unchanged packaged binary executor; no Phase88 repair",
  "processor_use_fast": false,
  "repetition_penalty": 1.05
}
