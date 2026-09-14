# Routed-state cache schema

Each unique `(UID, layer, exact post-trigger action prefix)` state stores the final valid text/query hidden vector, the exact routed visual-state tensor and mask, action-prefix hash, and tensor hashes. The state is captured immediately before the current layer action. The visual tensor is not compacted because shape changes can alter attention logits by ulps.
