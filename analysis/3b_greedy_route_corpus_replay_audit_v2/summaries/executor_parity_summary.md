Diagnostic job2941 COMPLETED successfully in20 seconds,23:22:58–23:23:18 KST.
It reproduced HF22/custom23; tracing did not change the generated prefixes.

First measured divergence: layer0 prefill SDPA output, max absolute difference
0.0078125. Input hidden states, RoPE and canonical Q/K/V are identical. Native
uses implicit causal masking plus GQA; custom uses an explicit causal mask plus
expanded KV. Native call settings applied to the same custom Q/K/V reproduce
the native attention output exactly. This supports repairing the FULL attention
call convention; it does not establish that a Torch version change caused it.

A local V2 adapter now uses native masking for unpadded FULL prefill and retains
original padded/text-only masks. The canonical package and diagnostic contract
are unchanged. Three CPU repair tests, import and shell checks pass.

Complete32-anchor repair validation is queued as job2942, oneH100,4CPUs,64GiB,
gpu-normal,30 minutes. Eight-GPU full replay has not been submitted; Gate A
must pass first. Current dense/routed labels and geometry readiness remain pending.
