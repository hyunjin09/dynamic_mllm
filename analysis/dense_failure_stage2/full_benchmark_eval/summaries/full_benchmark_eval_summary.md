# Full-Benchmark End-to-End Evaluation Summary

All four established benchmark families were evaluated under the frozen contract `63379eeff80fd5b046cb980cdfccea7fab232e15eb5b14924a24b60393327e83`. Every one of the 19,960 UIDs has exactly one successful paired dense/routed record; no partial worker or duplicate UID was accepted.

The population, prompt builder, native dense generation contract, and task scorers are the frozen `shared_prefix_eval_20260812` reference method. Native dense generation with passive Stage-1 hooks versus hook-free native dense passed exact token/scorer parity on all seven preflight task variants. No-trigger and all-FULL policy paths are defined as the exact native dense result; the four-action executor is invoked only when the frozen policy selects at least one non-FULL action. Dataset-level dense agreement with the rounded original-server reference is recorded in `metrics/reference_dense_parity.csv`.

| Benchmark family | N | Dense Acc | Routed Acc | ΔAcc | W→C | C→W | Net | C→C preservation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chartqa | 2500 | 0.858800 | 0.858000 | -0.000800 | 1 | 3 | -2 | 0.998603 |
| textvqa | 5000 | 0.857600 | 0.855600 | -0.002000 | 2 | 12 | -10 | 0.997201 |
| mmmu_pro | 3460 | 0.354046 | 0.352890 | -0.001156 | 0 | 4 | -4 | 0.996735 |
| pope | 9000 | 0.880000 | 0.880000 | +0.000000 | 0 | 0 | +0 | 1.000000 |
| overall | 19960 | 0.780561 | 0.779760 | -0.000802 | 3 | 19 | -16 | 0.998780 |

- Pooled ΔAccuracy 95% paired-bootstrap CI: [-0.001253, -0.000351].
- Pooled W→C rate 95% CI: [0.000000, 0.001603].
- Pooled C→C preservation 95% CI: [0.998203, 0.999296].
- Stage-1 trigger rate: 0.045140; P(trigger|Dense-C)=0.025995, P(trigger|Dense-W)=0.113242, trigger precision=0.550499.
- Among triggered samples, any non-FULL: 0.234184; post-trigger FULL fraction: 0.909500; mean non-FULL actions: 0.8479.
- Dominant rescue family: **textvqa** (2 W→C). Dominant regression family: **textvqa** (12 C→W).
- The effect is concentrated rather than broad.
