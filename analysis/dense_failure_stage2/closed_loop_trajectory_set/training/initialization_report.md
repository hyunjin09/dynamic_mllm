# Initialization report

- Router architecture: existing SharedReadWriteRouter.
- Initialization: exact strict load of the frozen Sequential-A checkpoint.
- Trainable: READ branch, WRITE branch, shared action head.
- Frozen/not loaded during cache training: Qwen backbone and Stage-1.
