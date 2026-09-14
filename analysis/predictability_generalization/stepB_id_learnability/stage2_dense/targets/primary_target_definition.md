# Primary Stage-2 target definition

- READ: `u_read_w1 = q_FULL - q_WRITE_ONLY`.
- WRITE: `u_write_r1 = q_FULL - q_READ_ONLY`.
- Continuous Huber regression is primary; sign and correctness-flip metrics are evaluation-only.
- Exact zero is neutral and excluded from harmful-sign AUROC/AUPRC.
