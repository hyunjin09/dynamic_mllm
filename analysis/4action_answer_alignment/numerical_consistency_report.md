# Numerical Consistency Report

## Definitions

Native Qwen FULL is an external semantic/cohort diagnostic. Unified
materialized-mask FULL is M11. Every reported READ/WRITE factorial
effect is computed only within the unified executor. Native/unified
drift is never an effect threshold.

## Validation semantics

```json
{
  "unified_full_vs_native": {
    "comparisons": 72,
    "correctness_match_count": 72,
    "evaluator_score_match_count": 72,
    "generated_answer_match_count": 72,
    "generated_ids_match_count": 72
  },
  "unified_ignore_vs_old_binary_single_off": {
    "comparisons": 1816,
    "correctness_match_count": 1816,
    "evaluator_score_match_count": 1816,
    "generated_answer_match_count": 1816,
    "generated_ids_match_count": 1816
  },
  "validation_sample_count": 72
}
```

## Native FULL versus unified FULL drift

Signed drift is `unified - native`. Absolute drift is reported
separately. Values are length-normalized teacher-forced log-probability
or correct-vs-frozen-FULL-wrong margin units.

| Analysis set | Cohort | Dataset | Quantity | Distribution | Mean | Median | Std | P90 | P95 | P99 | Min | Max |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| production | control_full_correct_all_off_wrong | gqa | S_correct | signed | 0.000284 | 0.000000 | 0.015377 | 0.010302 | 0.024073 | 0.055375 | -0.077372 | 0.117421 |
| production | control_full_correct_all_off_wrong | gqa | S_correct | absolute | 0.006813 | 0.000696 | 0.013789 | 0.023884 | 0.037146 | 0.061966 | 0.000000 | 0.117421 |
| production | control_full_correct_all_off_wrong | gqa | margin | signed | 0.000284 | 0.000000 | 0.015377 | 0.010302 | 0.024073 | 0.055375 | -0.077372 | 0.117421 |
| production | control_full_correct_all_off_wrong | gqa | margin | absolute | 0.006813 | 0.000696 | 0.013789 | 0.023884 | 0.037146 | 0.061966 | 0.000000 | 0.117421 |
| production | control_full_correct_all_off_wrong | textvqa | S_correct | signed | -0.000260 | -0.000008 | 0.011148 | 0.004277 | 0.010482 | 0.032219 | -0.111902 | 0.118815 |
| production | control_full_correct_all_off_wrong | textvqa | S_correct | absolute | 0.004039 | 0.000352 | 0.010394 | 0.011722 | 0.019620 | 0.051376 | 0.000000 | 0.118815 |
| production | control_full_correct_all_off_wrong | textvqa | margin | signed | -0.000260 | -0.000008 | 0.011148 | 0.004277 | 0.010482 | 0.032219 | -0.111902 | 0.118815 |
| production | control_full_correct_all_off_wrong | textvqa | margin | absolute | 0.004039 | 0.000352 | 0.010394 | 0.011722 | 0.019620 | 0.051376 | 0.000000 | 0.118815 |
| production | control_full_correct_all_off_wrong | joint | S_correct | signed | 0.000037 | -0.000000 | 0.013622 | 0.007138 | 0.016596 | 0.052394 | -0.111902 | 0.118815 |
| production | control_full_correct_all_off_wrong | joint | S_correct | absolute | 0.005553 | 0.000476 | 0.012439 | 0.016659 | 0.030242 | 0.057248 | 0.000000 | 0.118815 |
| production | control_full_correct_all_off_wrong | joint | margin | signed | 0.000037 | -0.000000 | 0.013622 | 0.007138 | 0.016596 | 0.052394 | -0.111902 | 0.118815 |
| production | control_full_correct_all_off_wrong | joint | margin | absolute | 0.005553 | 0.000476 | 0.012439 | 0.016659 | 0.030242 | 0.057248 | 0.000000 | 0.118815 |
| production | control_no_correction_found | gqa | S_correct | signed | -0.000075 | 0.000049 | 0.057954 | 0.062649 | 0.101075 | 0.161990 | -0.265362 | 0.247412 |
| production | control_no_correction_found | gqa | S_correct | absolute | 0.040449 | 0.029576 | 0.041503 | 0.101466 | 0.123093 | 0.184368 | 0.000000 | 0.265362 |
| production | control_no_correction_found | gqa | S_full_wrong | signed | 0.000285 | 0.000061 | 0.022539 | 0.025694 | 0.043192 | 0.063297 | -0.088922 | 0.102278 |
| production | control_no_correction_found | gqa | S_full_wrong | absolute | 0.013812 | 0.005766 | 0.017813 | 0.041600 | 0.051042 | 0.073580 | 0.000000 | 0.102278 |
| production | control_no_correction_found | gqa | margin | signed | -0.000360 | -0.000000 | 0.066717 | 0.070373 | 0.124795 | 0.155718 | -0.312501 | 0.250000 |
| production | control_no_correction_found | gqa | margin | absolute | 0.047173 | 0.038466 | 0.047181 | 0.125000 | 0.125000 | 0.187500 | 0.000000 | 0.312501 |
| production | control_no_correction_found | textvqa | S_correct | signed | -0.004501 | -0.002209 | 0.031441 | 0.021189 | 0.034837 | 0.067597 | -0.167956 | 0.223415 |
| production | control_no_correction_found | textvqa | S_correct | absolute | 0.018479 | 0.010112 | 0.025832 | 0.039323 | 0.059896 | 0.116883 | 0.000003 | 0.223415 |
| production | control_no_correction_found | textvqa | S_full_wrong | signed | 0.000395 | -0.000035 | 0.018334 | 0.021747 | 0.028489 | 0.052889 | -0.087046 | 0.078656 |
| production | control_no_correction_found | textvqa | S_full_wrong | absolute | 0.012250 | 0.007499 | 0.013646 | 0.028479 | 0.039175 | 0.062988 | 0.000001 | 0.087046 |
| production | control_no_correction_found | textvqa | margin | signed | -0.004895 | -0.000604 | 0.036500 | 0.034928 | 0.045037 | 0.065784 | -0.184486 | 0.232112 |
| production | control_no_correction_found | textvqa | margin | absolute | 0.023708 | 0.017717 | 0.028180 | 0.052027 | 0.062500 | 0.121566 | 0.000000 | 0.232112 |
| production | control_no_correction_found | joint | S_correct | signed | -0.001330 | -0.000030 | 0.051871 | 0.059484 | 0.079803 | 0.135474 | -0.265362 | 0.247412 |
| production | control_no_correction_found | joint | S_correct | absolute | 0.034220 | 0.021275 | 0.039005 | 0.085770 | 0.121045 | 0.172412 | 0.000000 | 0.265362 |
| production | control_no_correction_found | joint | S_full_wrong | signed | 0.000316 | 0.000050 | 0.021431 | 0.023874 | 0.039401 | 0.063140 | -0.088922 | 0.102278 |
| production | control_no_correction_found | joint | S_full_wrong | absolute | 0.013369 | 0.006352 | 0.016752 | 0.038575 | 0.049270 | 0.071160 | 0.000000 | 0.102278 |
| production | control_no_correction_found | joint | margin | signed | -0.001646 | -0.000000 | 0.059758 | 0.063105 | 0.108418 | 0.127994 | -0.312501 | 0.250000 |
| production | control_no_correction_found | joint | margin | absolute | 0.040520 | 0.029425 | 0.043953 | 0.117889 | 0.125000 | 0.187500 | 0.000000 | 0.312501 |
| production | primary_a_plus | gqa | S_correct | signed | 0.001609 | 0.000008 | 0.067796 | 0.079538 | 0.107425 | 0.124720 | -0.335359 | 1.122044 |
| production | primary_a_plus | gqa | S_correct | absolute | 0.039053 | 0.018361 | 0.055442 | 0.107334 | 0.118725 | 0.181219 | 0.000000 | 1.122044 |
| production | primary_a_plus | gqa | S_full_wrong | signed | 0.000139 | 0.000004 | 0.025863 | 0.027037 | 0.044371 | 0.068366 | -0.126872 | 0.162144 |
| production | primary_a_plus | gqa | S_full_wrong | absolute | 0.015645 | 0.007198 | 0.020596 | 0.043753 | 0.059374 | 0.087367 | 0.000000 | 0.162144 |
| production | primary_a_plus | gqa | margin | signed | 0.001470 | 0.000000 | 0.081223 | 0.122629 | 0.125000 | 0.136151 | -0.355414 | 1.125000 |
| production | primary_a_plus | gqa | margin | absolute | 0.048191 | 0.016418 | 0.065398 | 0.125000 | 0.125000 | 0.250000 | 0.000000 | 1.125000 |
| production | primary_a_plus | textvqa | S_correct | signed | -0.001078 | -0.000015 | 0.034738 | 0.033296 | 0.051446 | 0.108854 | -0.237303 | 0.123110 |
| production | primary_a_plus | textvqa | S_correct | absolute | 0.021179 | 0.011570 | 0.027556 | 0.053800 | 0.080478 | 0.113869 | 0.000000 | 0.237303 |
| production | primary_a_plus | textvqa | S_full_wrong | signed | -0.000140 | -0.000037 | 0.022578 | 0.021874 | 0.034742 | 0.070287 | -0.149845 | 0.122174 |
| production | primary_a_plus | textvqa | S_full_wrong | absolute | 0.014206 | 0.008460 | 0.017550 | 0.034437 | 0.051072 | 0.077761 | 0.000000 | 0.149845 |
| production | primary_a_plus | textvqa | margin | signed | -0.000938 | 0.000000 | 0.046068 | 0.047325 | 0.067570 | 0.125000 | -0.250000 | 0.250000 |
| production | primary_a_plus | textvqa | margin | absolute | 0.028764 | 0.015593 | 0.035996 | 0.071893 | 0.112172 | 0.137989 | 0.000000 | 0.250000 |
| production | primary_a_plus | joint | S_correct | signed | 0.000669 | 0.000004 | 0.058409 | 0.063239 | 0.098212 | 0.123210 | -0.335359 | 1.122044 |
| production | primary_a_plus | joint | S_correct | absolute | 0.032797 | 0.014406 | 0.048337 | 0.097970 | 0.114418 | 0.155131 | 0.000000 | 1.122044 |
| production | primary_a_plus | joint | S_full_wrong | signed | 0.000042 | -0.000000 | 0.024764 | 0.025490 | 0.041442 | 0.068825 | -0.149845 | 0.162144 |
| production | primary_a_plus | joint | S_full_wrong | absolute | 0.015141 | 0.007710 | 0.019596 | 0.040956 | 0.055789 | 0.083346 | 0.000000 | 0.162144 |
| production | primary_a_plus | joint | margin | signed | 0.000627 | 0.000000 | 0.070938 | 0.090075 | 0.125000 | 0.128468 | -0.355414 | 1.125000 |
| production | primary_a_plus | joint | margin | absolute | 0.041392 | 0.015914 | 0.057614 | 0.125000 | 0.125000 | 0.250000 | 0.000000 | 1.125000 |
| production | all | gqa | S_correct | signed | 0.000755 | 0.000002 | 0.051709 | 0.055560 | 0.085985 | 0.123618 | -0.335359 | 1.122044 |
| production | all | gqa | S_correct | absolute | 0.027011 | 0.006069 | 0.044100 | 0.086021 | 0.112747 | 0.145756 | 0.000000 | 1.122044 |
| production | all | gqa | S_full_wrong | signed | 0.000188 | 0.000010 | 0.024801 | 0.026922 | 0.043753 | 0.068199 | -0.126872 | 0.162144 |
| production | all | gqa | S_full_wrong | absolute | 0.015032 | 0.006456 | 0.019728 | 0.042856 | 0.056663 | 0.079732 | 0.000000 | 0.162144 |
| production | all | gqa | margin | signed | 0.000639 | 0.000000 | 0.061007 | 0.062608 | 0.125000 | 0.125000 | -0.355414 | 1.125000 |
| production | all | gqa | margin | absolute | 0.032156 | 0.003870 | 0.051849 | 0.124301 | 0.125000 | 0.187500 | 0.000000 | 1.125000 |
| production | all | textvqa | S_correct | signed | -0.001109 | -0.000011 | 0.025004 | 0.016799 | 0.032779 | 0.082503 | -0.237303 | 0.223415 |
| production | all | textvqa | S_correct | absolute | 0.012041 | 0.002338 | 0.021942 | 0.035559 | 0.053847 | 0.109015 | 0.000000 | 0.237303 |
| production | all | textvqa | S_full_wrong | signed | 0.000004 | -0.000035 | 0.021517 | 0.021753 | 0.032866 | 0.068034 | -0.149845 | 0.122174 |
| production | all | textvqa | S_full_wrong | absolute | 0.013678 | 0.008116 | 0.016610 | 0.032494 | 0.048257 | 0.073600 | 0.000000 | 0.149845 |
| production | all | textvqa | margin | signed | -0.001111 | -0.000003 | 0.031572 | 0.024897 | 0.044477 | 0.104025 | -0.250000 | 0.250000 |
| production | all | textvqa | margin | absolute | 0.015429 | 0.002589 | 0.027567 | 0.047829 | 0.064604 | 0.125000 | 0.000000 | 0.250000 |
| production | all | joint | S_correct | signed | 0.000040 | -0.000000 | 0.043467 | 0.037143 | 0.065546 | 0.120886 | -0.335359 | 1.122044 |
| production | all | joint | S_correct | absolute | 0.021273 | 0.004229 | 0.037906 | 0.065376 | 0.100989 | 0.124702 | 0.000000 | 1.122044 |
| production | all | joint | S_full_wrong | signed | 0.000127 | 0.000004 | 0.023771 | 0.025107 | 0.040786 | 0.068198 | -0.149845 | 0.162144 |
| production | all | joint | S_full_wrong | absolute | 0.014586 | 0.007192 | 0.018770 | 0.040211 | 0.054079 | 0.079093 | 0.000000 | 0.162144 |
| production | all | joint | margin | signed | -0.000032 | 0.000000 | 0.051750 | 0.049848 | 0.089888 | 0.125000 | -0.355414 | 1.125000 |
| production | all | joint | margin | absolute | 0.025744 | 0.003316 | 0.044892 | 0.089888 | 0.125000 | 0.138831 | 0.000000 | 1.125000 |
| validation_pilot | primary_a_plus | gqa | S_correct | signed | -0.000809 | 0.000006 | 0.062206 | 0.072339 | 0.111647 | 0.119430 | -0.184519 | 0.119912 |
| validation_pilot | primary_a_plus | gqa | S_correct | absolute | 0.041099 | 0.029931 | 0.046702 | 0.105165 | 0.119288 | 0.167075 | 0.000010 | 0.184519 |
| validation_pilot | primary_a_plus | gqa | S_full_wrong | signed | 0.001320 | -0.000293 | 0.020053 | 0.029907 | 0.031629 | 0.052770 | -0.037517 | 0.060372 |
| validation_pilot | primary_a_plus | gqa | S_full_wrong | absolute | 0.013143 | 0.005809 | 0.015203 | 0.032368 | 0.035838 | 0.054201 | 0.000002 | 0.060372 |
| validation_pilot | primary_a_plus | gqa | margin | signed | -0.002129 | -0.000000 | 0.072851 | 0.081779 | 0.124100 | 0.125000 | -0.187500 | 0.125000 |
| validation_pilot | primary_a_plus | gqa | margin | absolute | 0.046673 | 0.008627 | 0.055976 | 0.125000 | 0.125000 | 0.170625 | 0.000000 | 0.187500 |
| validation_pilot | primary_a_plus | textvqa | S_correct | signed | -0.009776 | -0.000967 | 0.035156 | 0.027972 | 0.045656 | 0.049024 | -0.108896 | 0.049529 |
| validation_pilot | primary_a_plus | textvqa | S_correct | absolute | 0.023784 | 0.015745 | 0.027674 | 0.060224 | 0.076752 | 0.100336 | 0.000036 | 0.108896 |
| validation_pilot | primary_a_plus | textvqa | S_full_wrong | signed | 0.002204 | 0.000022 | 0.024038 | 0.027226 | 0.044616 | 0.066976 | -0.053491 | 0.073600 |
| validation_pilot | primary_a_plus | textvqa | S_full_wrong | absolute | 0.015548 | 0.008892 | 0.018464 | 0.040165 | 0.051943 | 0.068171 | 0.000018 | 0.073600 |
| validation_pilot | primary_a_plus | textvqa | margin | signed | -0.011980 | -0.000005 | 0.048891 | 0.036147 | 0.054838 | 0.068574 | -0.125000 | 0.073472 |
| validation_pilot | primary_a_plus | textvqa | margin | absolute | 0.032386 | 0.014850 | 0.038535 | 0.092371 | 0.116428 | 0.125000 | 0.000000 | 0.125000 |
| validation_pilot | primary_a_plus | joint | S_correct | signed | -0.005293 | -0.000562 | 0.050723 | 0.048595 | 0.070391 | 0.118931 | -0.184519 | 0.119912 |
| validation_pilot | primary_a_plus | joint | S_correct | absolute | 0.032441 | 0.019088 | 0.039350 | 0.088370 | 0.111204 | 0.148985 | 0.000010 | 0.184519 |
| validation_pilot | primary_a_plus | joint | S_full_wrong | signed | 0.001762 | -0.000010 | 0.022140 | 0.030087 | 0.039529 | 0.066325 | -0.053491 | 0.073600 |
| validation_pilot | primary_a_plus | joint | S_full_wrong | absolute | 0.014346 | 0.007328 | 0.016955 | 0.035458 | 0.050173 | 0.066325 | 0.000002 | 0.073600 |
| validation_pilot | primary_a_plus | joint | margin | signed | -0.007055 | -0.000000 | 0.062233 | 0.062478 | 0.085711 | 0.125000 | -0.187500 | 0.125000 |
| validation_pilot | primary_a_plus | joint | margin | absolute | 0.039530 | 0.012096 | 0.048582 | 0.125000 | 0.125000 | 0.153125 | 0.000000 | 0.187500 |
| validation_pilot | all | gqa | S_correct | signed | -0.000809 | 0.000006 | 0.062206 | 0.072339 | 0.111647 | 0.119430 | -0.184519 | 0.119912 |
| validation_pilot | all | gqa | S_correct | absolute | 0.041099 | 0.029931 | 0.046702 | 0.105165 | 0.119288 | 0.167075 | 0.000010 | 0.184519 |
| validation_pilot | all | gqa | S_full_wrong | signed | 0.001320 | -0.000293 | 0.020053 | 0.029907 | 0.031629 | 0.052770 | -0.037517 | 0.060372 |
| validation_pilot | all | gqa | S_full_wrong | absolute | 0.013143 | 0.005809 | 0.015203 | 0.032368 | 0.035838 | 0.054201 | 0.000002 | 0.060372 |
| validation_pilot | all | gqa | margin | signed | -0.002129 | -0.000000 | 0.072851 | 0.081779 | 0.124100 | 0.125000 | -0.187500 | 0.125000 |
| validation_pilot | all | gqa | margin | absolute | 0.046673 | 0.008627 | 0.055976 | 0.125000 | 0.125000 | 0.170625 | 0.000000 | 0.187500 |
| validation_pilot | all | textvqa | S_correct | signed | -0.009776 | -0.000967 | 0.035156 | 0.027972 | 0.045656 | 0.049024 | -0.108896 | 0.049529 |
| validation_pilot | all | textvqa | S_correct | absolute | 0.023784 | 0.015745 | 0.027674 | 0.060224 | 0.076752 | 0.100336 | 0.000036 | 0.108896 |
| validation_pilot | all | textvqa | S_full_wrong | signed | 0.002204 | 0.000022 | 0.024038 | 0.027226 | 0.044616 | 0.066976 | -0.053491 | 0.073600 |
| validation_pilot | all | textvqa | S_full_wrong | absolute | 0.015548 | 0.008892 | 0.018464 | 0.040165 | 0.051943 | 0.068171 | 0.000018 | 0.073600 |
| validation_pilot | all | textvqa | margin | signed | -0.011980 | -0.000005 | 0.048891 | 0.036147 | 0.054838 | 0.068574 | -0.125000 | 0.073472 |
| validation_pilot | all | textvqa | margin | absolute | 0.032386 | 0.014850 | 0.038535 | 0.092371 | 0.116428 | 0.125000 | 0.000000 | 0.125000 |
| validation_pilot | all | joint | S_correct | signed | -0.005293 | -0.000562 | 0.050723 | 0.048595 | 0.070391 | 0.118931 | -0.184519 | 0.119912 |
| validation_pilot | all | joint | S_correct | absolute | 0.032441 | 0.019088 | 0.039350 | 0.088370 | 0.111204 | 0.148985 | 0.000010 | 0.184519 |
| validation_pilot | all | joint | S_full_wrong | signed | 0.001762 | -0.000010 | 0.022140 | 0.030087 | 0.039529 | 0.066325 | -0.053491 | 0.073600 |
| validation_pilot | all | joint | S_full_wrong | absolute | 0.014346 | 0.007328 | 0.016955 | 0.035458 | 0.050173 | 0.066325 | 0.000002 | 0.073600 |
| validation_pilot | all | joint | margin | signed | -0.007055 | -0.000000 | 0.062233 | 0.062478 | 0.085711 | 0.125000 | -0.187500 | 0.125000 |
| validation_pilot | all | joint | margin | absolute | 0.039530 | 0.012096 | 0.048582 | 0.125000 | 0.125000 | 0.153125 | 0.000000 | 0.187500 |
| validation_preflight | primary_a_plus | gqa | S_correct | signed | -0.039991 | -0.046787 | 0.112393 | 0.082957 | 0.100543 | 0.114612 | -0.184519 | 0.118129 |
| validation_preflight | primary_a_plus | gqa | S_correct | absolute | 0.099500 | 0.106296 | 0.065812 | 0.164602 | 0.174561 | 0.182528 | 0.000888 | 0.184519 |
| validation_preflight | primary_a_plus | gqa | S_full_wrong | signed | 0.009966 | 0.008100 | 0.013839 | 0.025342 | 0.027939 | 0.030017 | -0.006871 | 0.030537 |
| validation_preflight | primary_a_plus | gqa | S_full_wrong | absolute | 0.013402 | 0.010045 | 0.010546 | 0.025342 | 0.027939 | 0.030017 | 0.002981 | 0.030537 |
| validation_preflight | primary_a_plus | gqa | margin | signed | -0.049958 | -0.068665 | 0.118928 | 0.083801 | 0.104400 | 0.120880 | -0.187500 | 0.125000 |
| validation_preflight | primary_a_plus | gqa | margin | absolute | 0.112458 | 0.125000 | 0.063189 | 0.168750 | 0.178125 | 0.185625 | 0.012331 | 0.187500 |
| validation_preflight | primary_a_plus | textvqa | S_correct | signed | 0.012686 | 0.002780 | 0.020626 | 0.035731 | 0.041696 | 0.046468 | -0.002477 | 0.047661 |
| validation_preflight | primary_a_plus | textvqa | S_correct | absolute | 0.015091 | 0.005185 | 0.018938 | 0.035731 | 0.041696 | 0.046468 | 0.002332 | 0.047661 |
| validation_preflight | primary_a_plus | textvqa | S_full_wrong | signed | -0.000303 | 0.000820 | 0.003638 | 0.002859 | 0.003133 | 0.003352 | -0.006261 | 0.003407 |
| validation_preflight | primary_a_plus | textvqa | S_full_wrong | absolute | 0.002827 | 0.002493 | 0.002310 | 0.005405 | 0.005833 | 0.006176 | 0.000061 | 0.006261 |
| validation_preflight | primary_a_plus | textvqa | margin | signed | 0.012989 | 0.001887 | 0.024041 | 0.039640 | 0.046781 | 0.052494 | -0.005739 | 0.053922 |
| validation_preflight | primary_a_plus | textvqa | margin | absolute | 0.017128 | 0.006026 | 0.021292 | 0.039640 | 0.046781 | 0.052494 | 0.002538 | 0.053922 |
| validation_preflight | primary_a_plus | joint | S_correct | signed | -0.013653 | -0.000722 | 0.084985 | 0.068801 | 0.093465 | 0.113196 | -0.184519 | 0.118129 |
| validation_preflight | primary_a_plus | joint | S_correct | absolute | 0.057295 | 0.027777 | 0.064235 | 0.138046 | 0.161283 | 0.179872 | 0.000888 | 0.184519 |
| validation_preflight | primary_a_plus | joint | S_full_wrong | signed | 0.004832 | 0.002280 | 0.011346 | 0.018415 | 0.024476 | 0.029325 | -0.006871 | 0.030537 |
| validation_preflight | primary_a_plus | joint | S_full_wrong | absolute | 0.008115 | 0.004834 | 0.009286 | 0.018415 | 0.024476 | 0.029325 | 0.000061 | 0.030537 |
| validation_preflight | primary_a_plus | joint | margin | signed | -0.018484 | -0.004139 | 0.091386 | 0.075246 | 0.100123 | 0.120025 | -0.187500 | 0.125000 |
| validation_preflight | primary_a_plus | joint | margin | absolute | 0.064793 | 0.033127 | 0.067045 | 0.143750 | 0.165625 | 0.183125 | 0.002538 | 0.187500 |
| validation_preflight | all | gqa | S_correct | signed | -0.039991 | -0.046787 | 0.112393 | 0.082957 | 0.100543 | 0.114612 | -0.184519 | 0.118129 |
| validation_preflight | all | gqa | S_correct | absolute | 0.099500 | 0.106296 | 0.065812 | 0.164602 | 0.174561 | 0.182528 | 0.000888 | 0.184519 |
| validation_preflight | all | gqa | S_full_wrong | signed | 0.009966 | 0.008100 | 0.013839 | 0.025342 | 0.027939 | 0.030017 | -0.006871 | 0.030537 |
| validation_preflight | all | gqa | S_full_wrong | absolute | 0.013402 | 0.010045 | 0.010546 | 0.025342 | 0.027939 | 0.030017 | 0.002981 | 0.030537 |
| validation_preflight | all | gqa | margin | signed | -0.049958 | -0.068665 | 0.118928 | 0.083801 | 0.104400 | 0.120880 | -0.187500 | 0.125000 |
| validation_preflight | all | gqa | margin | absolute | 0.112458 | 0.125000 | 0.063189 | 0.168750 | 0.178125 | 0.185625 | 0.012331 | 0.187500 |
| validation_preflight | all | textvqa | S_correct | signed | 0.012686 | 0.002780 | 0.020626 | 0.035731 | 0.041696 | 0.046468 | -0.002477 | 0.047661 |
| validation_preflight | all | textvqa | S_correct | absolute | 0.015091 | 0.005185 | 0.018938 | 0.035731 | 0.041696 | 0.046468 | 0.002332 | 0.047661 |
| validation_preflight | all | textvqa | S_full_wrong | signed | -0.000303 | 0.000820 | 0.003638 | 0.002859 | 0.003133 | 0.003352 | -0.006261 | 0.003407 |
| validation_preflight | all | textvqa | S_full_wrong | absolute | 0.002827 | 0.002493 | 0.002310 | 0.005405 | 0.005833 | 0.006176 | 0.000061 | 0.006261 |
| validation_preflight | all | textvqa | margin | signed | 0.012989 | 0.001887 | 0.024041 | 0.039640 | 0.046781 | 0.052494 | -0.005739 | 0.053922 |
| validation_preflight | all | textvqa | margin | absolute | 0.017128 | 0.006026 | 0.021292 | 0.039640 | 0.046781 | 0.052494 | 0.002538 | 0.053922 |
| validation_preflight | all | joint | S_correct | signed | -0.013653 | -0.000722 | 0.084985 | 0.068801 | 0.093465 | 0.113196 | -0.184519 | 0.118129 |
| validation_preflight | all | joint | S_correct | absolute | 0.057295 | 0.027777 | 0.064235 | 0.138046 | 0.161283 | 0.179872 | 0.000888 | 0.184519 |
| validation_preflight | all | joint | S_full_wrong | signed | 0.004832 | 0.002280 | 0.011346 | 0.018415 | 0.024476 | 0.029325 | -0.006871 | 0.030537 |
| validation_preflight | all | joint | S_full_wrong | absolute | 0.008115 | 0.004834 | 0.009286 | 0.018415 | 0.024476 | 0.029325 | 0.000061 | 0.030537 |
| validation_preflight | all | joint | margin | signed | -0.018484 | -0.004139 | 0.091386 | 0.075246 | 0.100123 | 0.120025 | -0.187500 | 0.125000 |
| validation_preflight | all | joint | margin | absolute | 0.064793 | 0.033127 | 0.067045 | 0.143750 | 0.165625 | 0.183125 | 0.002538 | 0.187500 |
| validation_smoke | primary_a_plus | gqa | S_correct | signed | -0.039991 | -0.046787 | 0.112393 | 0.082957 | 0.100543 | 0.114612 | -0.184519 | 0.118129 |
| validation_smoke | primary_a_plus | gqa | S_correct | absolute | 0.099500 | 0.106296 | 0.065812 | 0.164602 | 0.174561 | 0.182528 | 0.000888 | 0.184519 |
| validation_smoke | primary_a_plus | gqa | S_full_wrong | signed | 0.009966 | 0.008100 | 0.013839 | 0.025342 | 0.027939 | 0.030017 | -0.006871 | 0.030537 |
| validation_smoke | primary_a_plus | gqa | S_full_wrong | absolute | 0.013402 | 0.010045 | 0.010546 | 0.025342 | 0.027939 | 0.030017 | 0.002981 | 0.030537 |
| validation_smoke | primary_a_plus | gqa | margin | signed | -0.049958 | -0.068665 | 0.118928 | 0.083801 | 0.104400 | 0.120880 | -0.187500 | 0.125000 |
| validation_smoke | primary_a_plus | gqa | margin | absolute | 0.112458 | 0.125000 | 0.063189 | 0.168750 | 0.178125 | 0.185625 | 0.012331 | 0.187500 |
| validation_smoke | primary_a_plus | textvqa | S_correct | signed | 0.012686 | 0.002780 | 0.020626 | 0.035731 | 0.041696 | 0.046468 | -0.002477 | 0.047661 |
| validation_smoke | primary_a_plus | textvqa | S_correct | absolute | 0.015091 | 0.005185 | 0.018938 | 0.035731 | 0.041696 | 0.046468 | 0.002332 | 0.047661 |
| validation_smoke | primary_a_plus | textvqa | S_full_wrong | signed | -0.000303 | 0.000820 | 0.003638 | 0.002859 | 0.003133 | 0.003352 | -0.006261 | 0.003407 |
| validation_smoke | primary_a_plus | textvqa | S_full_wrong | absolute | 0.002827 | 0.002493 | 0.002310 | 0.005405 | 0.005833 | 0.006176 | 0.000061 | 0.006261 |
| validation_smoke | primary_a_plus | textvqa | margin | signed | 0.012989 | 0.001887 | 0.024041 | 0.039640 | 0.046781 | 0.052494 | -0.005739 | 0.053922 |
| validation_smoke | primary_a_plus | textvqa | margin | absolute | 0.017128 | 0.006026 | 0.021292 | 0.039640 | 0.046781 | 0.052494 | 0.002538 | 0.053922 |
| validation_smoke | primary_a_plus | joint | S_correct | signed | -0.013653 | -0.000722 | 0.084985 | 0.068801 | 0.093465 | 0.113196 | -0.184519 | 0.118129 |
| validation_smoke | primary_a_plus | joint | S_correct | absolute | 0.057295 | 0.027777 | 0.064235 | 0.138046 | 0.161283 | 0.179872 | 0.000888 | 0.184519 |
| validation_smoke | primary_a_plus | joint | S_full_wrong | signed | 0.004832 | 0.002280 | 0.011346 | 0.018415 | 0.024476 | 0.029325 | -0.006871 | 0.030537 |
| validation_smoke | primary_a_plus | joint | S_full_wrong | absolute | 0.008115 | 0.004834 | 0.009286 | 0.018415 | 0.024476 | 0.029325 | 0.000061 | 0.030537 |
| validation_smoke | primary_a_plus | joint | margin | signed | -0.018484 | -0.004139 | 0.091386 | 0.075246 | 0.100123 | 0.120025 | -0.187500 | 0.125000 |
| validation_smoke | primary_a_plus | joint | margin | absolute | 0.064793 | 0.033127 | 0.067045 | 0.143750 | 0.165625 | 0.183125 | 0.002538 | 0.187500 |
| validation_smoke | all | gqa | S_correct | signed | -0.039991 | -0.046787 | 0.112393 | 0.082957 | 0.100543 | 0.114612 | -0.184519 | 0.118129 |
| validation_smoke | all | gqa | S_correct | absolute | 0.099500 | 0.106296 | 0.065812 | 0.164602 | 0.174561 | 0.182528 | 0.000888 | 0.184519 |
| validation_smoke | all | gqa | S_full_wrong | signed | 0.009966 | 0.008100 | 0.013839 | 0.025342 | 0.027939 | 0.030017 | -0.006871 | 0.030537 |
| validation_smoke | all | gqa | S_full_wrong | absolute | 0.013402 | 0.010045 | 0.010546 | 0.025342 | 0.027939 | 0.030017 | 0.002981 | 0.030537 |
| validation_smoke | all | gqa | margin | signed | -0.049958 | -0.068665 | 0.118928 | 0.083801 | 0.104400 | 0.120880 | -0.187500 | 0.125000 |
| validation_smoke | all | gqa | margin | absolute | 0.112458 | 0.125000 | 0.063189 | 0.168750 | 0.178125 | 0.185625 | 0.012331 | 0.187500 |
| validation_smoke | all | textvqa | S_correct | signed | 0.012686 | 0.002780 | 0.020626 | 0.035731 | 0.041696 | 0.046468 | -0.002477 | 0.047661 |
| validation_smoke | all | textvqa | S_correct | absolute | 0.015091 | 0.005185 | 0.018938 | 0.035731 | 0.041696 | 0.046468 | 0.002332 | 0.047661 |
| validation_smoke | all | textvqa | S_full_wrong | signed | -0.000303 | 0.000820 | 0.003638 | 0.002859 | 0.003133 | 0.003352 | -0.006261 | 0.003407 |
| validation_smoke | all | textvqa | S_full_wrong | absolute | 0.002827 | 0.002493 | 0.002310 | 0.005405 | 0.005833 | 0.006176 | 0.000061 | 0.006261 |
| validation_smoke | all | textvqa | margin | signed | 0.012989 | 0.001887 | 0.024041 | 0.039640 | 0.046781 | 0.052494 | -0.005739 | 0.053922 |
| validation_smoke | all | textvqa | margin | absolute | 0.017128 | 0.006026 | 0.021292 | 0.039640 | 0.046781 | 0.052494 | 0.002538 | 0.053922 |
| validation_smoke | all | joint | S_correct | signed | -0.013653 | -0.000722 | 0.084985 | 0.068801 | 0.093465 | 0.113196 | -0.184519 | 0.118129 |
| validation_smoke | all | joint | S_correct | absolute | 0.057295 | 0.027777 | 0.064235 | 0.138046 | 0.161283 | 0.179872 | 0.000888 | 0.184519 |
| validation_smoke | all | joint | S_full_wrong | signed | 0.004832 | 0.002280 | 0.011346 | 0.018415 | 0.024476 | 0.029325 | -0.006871 | 0.030537 |
| validation_smoke | all | joint | S_full_wrong | absolute | 0.008115 | 0.004834 | 0.009286 | 0.018415 | 0.024476 | 0.029325 | 0.000061 | 0.030537 |
| validation_smoke | all | joint | margin | signed | -0.018484 | -0.004139 | 0.091386 | 0.075246 | 0.100123 | 0.120025 | -0.187500 | 0.125000 |
| validation_smoke | all | joint | margin | absolute | 0.064793 | 0.033127 | 0.067045 | 0.143750 | 0.165625 | 0.183125 | 0.002538 | 0.187500 |

## Separation from causal effects

M00, M10, M01, and M11 share the unified-FULL prefix and suffix.
No value in this report is subtracted from, used to calibrate, or
used to threshold a within-unified causal effect.
