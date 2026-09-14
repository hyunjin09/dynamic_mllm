# Robust Stage-1 operating-point summary

The frozen Stage-1 object is the Phase-63 five-checkpoint per-layer probability-mean system. Train-map rates below are descriptive in-sample admission rates, not held-out calibration claims.

| Point | Tau | Hist C preserve | Canon C preserve | Hist W recall | Canon W recall | Pooled W recall | Median layer | Fold range | Retained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P99 | 0.983709 | 0.9975 | 0.9901 | 0.0250 | 0.0597 | 0.0488 | 24.0 | 0.0047 | False |
| P98 | 0.971135 | 0.9875 | 0.9808 | 0.0650 | 0.1240 | 0.1054 | 23.0 | 0.0036 | True |
| P97 | 0.961830 | 0.9850 | 0.9703 | 0.0975 | 0.1596 | 0.1400 | 22.0 | 0.0053 | False |
| P95 | 0.944802 | 0.9775 | 0.9501 | 0.1525 | 0.2480 | 0.2179 | 21.5 | 0.0100 | True |
| P90 | 0.906133 | 0.9500 | 0.9003 | 0.2525 | 0.4053 | 0.3572 | 20.0 | 0.0146 | True |

Retained operating points are P98, P95, P90. They represent conservative, middle, and permissive Stage-1 candidates because they passed the prospective fold-stability, distinctness, and >=5-point incremental pooled-W-recall rules. None is a final deployment threshold: only future measured W→C and C→W can select that.
