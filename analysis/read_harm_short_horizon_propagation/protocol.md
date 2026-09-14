# READ short-horizon propagation protocol

- Frozen contract: `3a0d2251e23b43082227b9b05befd9a37d9b9ecc6bb40a570861ebd7affebb6a`
- Target: `H_R = q_WRITE_ONLY - q_FULL`; positive is harmful READ.
- Population: 15,185 exact dense states / 1,413 UIDs.
- Horizons: H=1/2/4/8; branch ON is FULL at layer l, branch OFF is WRITE_ONLY at l, and every later executed layer is FULL.
- Primary emergence: two-layer MLP on `delta = ON - OFF`, evaluated on common H=8 support against H=1.
- Fixed controls: ON, OFF, PAIR, PAIR+DELTA, text/visual delta, identical token comparator, random pairs, and UID-permuted training targets.
- OOF: inherited Step-B five-fold image-group registry, fold-local normalization, UID-balanced Huber loss, three fixed seeds.
- Bulk extraction requires exact Phase-82 H=1 parity, ON canonical parity at every reached horizon, fresh-cache repeatability, and swapped branch-order parity.
- No WRITE analysis, search, deployment router, generation, external evaluation, or target redesign.
