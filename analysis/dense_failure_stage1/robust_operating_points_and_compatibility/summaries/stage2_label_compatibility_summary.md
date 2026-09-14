# Stage-2 label compatibility summary

Exact current-runtime replay statuses over route×operating-point rows: {'REPLAY_COMPATIBLE_CORRECT': 2618}.

| Point | Triggered W | Same | Earlier | Later | Compatible single routes | Compatible MCTS routes | W with reusable route | New search | New triggered C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P98 | 344 | 5 | 10 | 308 | 351 | 12 | 67 | 277 | 7 |
| P95 | 727 | 26 | 68 | 515 | 703 | 35 | 146 | 581 | 30 |
| P90 | 1307 | 42 | 223 | 699 | 1442 | 75 | 259 | 1048 | 106 |

Structural compatibility was only a pre-filter; every counted reusable route passed current Qwen/LMMS execution and exact stored-token parity. A missing compatible label is not evidence of unfixability.
