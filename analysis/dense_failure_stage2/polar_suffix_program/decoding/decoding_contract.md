# Decoding contract

At the first robust Stage-1 P90 trigger, the predictor consumes the exact all-FULL incoming trigger state once. It decodes the entire fixed-length suffix with beam width 8 and sum of action log probabilities before any suffix action executes. Greedy is logged for diagnostics only. No oracle, MCTS, reranking, or intervention penalty is used.
