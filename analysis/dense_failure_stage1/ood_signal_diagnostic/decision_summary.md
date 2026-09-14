# OOD First-Failure Signal Diagnostic Decision Summary

Frozen protocol: `dfef70e6f802bef02320b034aad995dd6384bf73a71d31e7a747731ce5c64fff`.

| Source datasets | OOD target | Source-selected layer | OOD AUROC | OOD AUPRC | Input AUROC | Hidden - input | Actual preservation @ source 99% | Wrong recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gqa + textvqa | chartqa | 26 | 0.4502 | 0.5026 | 0.5472 | -0.0970 | 0.3053 | 0.5840 |
| chartqa + textvqa | gqa | 20 | 0.6160 | 0.5953 | 0.5432 | +0.0729 | 0.1380 | 0.9385 |
| gqa + chartqa | textvqa | 22 | 0.7236 | 0.7345 | 0.5219 | +0.2017 | 0.9950 | 0.0920 |

## Q1. Does prediction transfer to an entirely unseen target benchmark?

Not uniformly under the frozen descriptive floor of 0.65; at least one source-selected target fell below it.

## Q2. How much does performance drop from in-domain to OOD?

The comparison below uses the exact same Phase-48 target-test UIDs; positive values mean in-domain is better.

| Target | Layer | In-domain AUROC | OOD AUROC | AUROC drop | In-domain recall @ own val-99% | OOD recall @ source-val-99% | Recall drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| chartqa | 26 | 0.9721 | 0.4343 | +0.5378 | 0.7000 | 0.6200 | +0.0800 |
| gqa | 20 | 0.7487 | 0.6109 | +0.1379 | 0.0850 | 0.9350 | -0.8500 |
| textvqa | 22 | 0.9827 | 0.7561 | +0.2266 | 0.5700 | 0.0600 | +0.5100 |

## Q3. Do decoder hidden states outperform the input control OOD?

The source-selected hidden state exceeded the native pre-language-decoder control by at least 0.02 AUROC on `2` of 3 targets. The control already includes learned visual encoding and multimodal insertion, so this comparison isolates additions after language-decoder computation rather than raw-input difficulty.

## Q4. Does OOD predictability improve with depth?

| Target | L0 AUROC | L14 AUROC | L21 AUROC | L27 AUROC | L21 - L0 |
|---|---:|---:|---:|---:|---:|
| chartqa | 0.6626 | 0.8351 | 0.8018 | 0.4524 | +0.1392 |
| gqa | 0.5527 | 0.5577 | 0.5975 | 0.5801 | +0.0448 |
| textvqa | 0.4995 | 0.6670 | 0.8046 | 0.7239 | +0.3051 |

## Q5. Is conservative wrong detection useful OOD?

At each source-selected layer, the table at the top reports the actual OOD preservation and wrong recall after transferring the source-validation 99% threshold unchanged. Full 99/98/95% curves are in each run's `selective_metrics.csv`.

## Q6. Is the signal benchmark-general rather than purely benchmark-specific?

The fixed heuristic does not support a strong benchmark-general computation-dependent claim. The evidence is mixed or benchmark-dependent, even if some individual transfers are useful.

## Q7. Should the shared Stage-1 predictor be the next experiment?

Not yet on the benchmark-general rationale. A shared predictor could only be justified as a deployment-mixture predictor or after a separately approved diagnostic/pivot. No further experiment is selected here.

## Validity limits

- Target data was evaluated only after source checkpoints, thresholds, and representative layers were frozen.
- The full OOD targets preserve the deliberately selected near-balanced candidate population; AUPRC/precision are conditional on that population.
- Linear accessibility is predictive evidence, not proof of causal self-awareness.
- The native pre-language-decoder control is learned and multimodal, not raw input.
