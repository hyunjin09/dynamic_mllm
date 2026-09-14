# Schedule freeze

Benchmark, pooled global and20seeded matched-bit schedules frozen after complete READ/WRITE calibration and before any held-out intervention. No combined calibration gain was used to retune isolated choices. All NONE decisions and duplicate random draws retained.

| benchmark   | bit   |   selected_layer |   n |      gain |   net |   w_to_c |   c_to_w |        mean_q |
|:------------|:------|-----------------:|----:|----------:|------:|---------:|---------:|--------------:|
| chartqa     | R     |               12 | 256 | 0.0117188 |     3 |        4 |        1 |   0.00219631  |
| chartqa     | W     |               25 | 256 | 0.0117188 |     3 |        3 |        0 |   0.000642235 |
| textvqa     | R     |               11 | 255 | 0.0117647 |     3 |        3 |        0 | nan           |
| textvqa     | W     |               24 | 255 | 0.0117647 |     3 |        4 |        1 | nan           |
| mmmupro     | R     |               17 | 256 | 0.015625  |     4 |        7 |        3 |   0.0163971   |
| mmmupro     | W     |               14 | 256 | 0.0195312 |     5 |       12 |        7 |   0.0769678   |
| pope        | R     |                7 | 252 | 0.0119048 |     3 |        3 |        0 |   0.0294587   |
| pope        | W     |               14 | 252 | 0.0119048 |     3 |        3 |        0 |   0.0118234   |

Global: {"read_layer": 17, "write_layer": 19, "actions": ["FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "WRITE_ONLY", "FULL", "READ_ONLY", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL", "FULL"], "pool_uids": 1019}
