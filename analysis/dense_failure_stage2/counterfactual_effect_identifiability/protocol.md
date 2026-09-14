# Counterfactual effect identifiability protocol

- Frozen contract: `a70e921a7f36f7fc2cee4b492c36774a2436aeddcb9c684fb3fa4e10cb602705`
- Dense primary: 15185 exact post-trigger states; routed secondary: 35565 exact prefix states.
- Inputs stop immediately after the current layer. No suffix state, final norm, LM head, generated answer, or correctness is an input.
- READ target: `q_FULL-q_WRITE_ONLY`; WRITE target: `q_FULL-q_READ_ONLY` from frozen Step A.
- Primary fixed-capacity comparison: Linear and 128-wide two-layer MLP over PRE/FULL_POST/OFF_POST/PAIR/DELTA/PAIR_PLUS_DELTA.
- Token comparator: shared 64-wide text/visual projection plus one-head attention per branch, then `[FULL;OFF;FULL-OFF]` readout.
- OOF: inherited Step-B five-fold image-group split, fold-local normalization, UID-balanced Huber regression, three frozen seeds.
- External deployment, router training, MCTS, and target redesign are out of scope.
