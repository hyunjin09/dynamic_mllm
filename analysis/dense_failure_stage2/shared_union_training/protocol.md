# Shared-union Stage-2 frozen protocol

- Contract: `b17a81d4749b842fd843a344d3799bffdfc0f41e2c516a835242b656324440f8`
- Phase-65 source contract: `767284f1e8ed76d5d1c797562a3aaea4ea59183703d62b1521545cacc7c57ab4`
- Union A/B/C: 106 preservation UIDs, 270 single W UIDs / 1,688 routes, 216 MCTS W UIDs / 725 routes.
- Router: unchanged shared READ/WRITE V1 architecture; no layer, Stage-1, source, or dataset input.
- A: 698 single W + 350 C draws/epoch. B: 349 single + 349 MCTS W + 350 C draws/epoch.
- Both: 12 epochs, 3,144 updates, AdamW 3e-4, constant schedule, plain four-way CE, final-update checkpoint only.
- Rollout: same checkpoint at strict P98/P95/P90 over the zero-overlap frozen Historical validation 800.
- Canonical validation: unavailable; Canonical OOF 4,000 contains Stage-2 union-training UIDs and is not used as held-out evidence.
- Experiment B runs only if A has a passing implementation/overfit/training record and at least one threshold with W-to-C > 0 and net correction > 0.
- Stop: after the matched A/B (conditional) validation comparison; no test or Stage-1 change.
