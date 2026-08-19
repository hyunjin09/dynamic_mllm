# GPU Policy

## Purpose
This document defines how the agent should handle GPU-required tasks.

## Rule
For any task requiring GPU execution, the agent must use the project GPU scheduler.

## GPU-required tasks
- model inference
- generation
- training
- validation with torch/transformers CUDA
- latency or memory benchmark

## CPU-only tasks
- code editing
- static analysis
- documentation
- syntax checks
- small unit tests

## Execution policy
- Do not manually set CUDA_VISIBLE_DEVICES.
- Do not run long GPU jobs in the foreground.
- Use the project GPU scheduler.
- If scheduler or conda environment is missing, stop and report the blocker.

## Node06 NCCL workaround

Node06 has shown multi-process CUDA hangs when NCCL routes peer traffic through
`P2P/CUMEM`. If node06 must be used for multi-GPU training or inference, export
these settings before launching `torchrun`:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

For single-node node06 runs, prefer:

```bash
torchrun --standalone --nproc_per_node=<num_gpus> ...
```

Node06 should still be treated as lower-confidence than node02, node03, and
node07. For real training or inference on node06, monitor startup and confirm
the job reaches actual train iterations, validation, or inference samples before
leaving it unattended.

## Scheduler memory caps

The scheduler must not request more Slurm memory than the approved
single-GPU cap for the selected node. For multi-GPU jobs on one node, the
allowed total memory is this per-GPU cap multiplied by the requested GPU count.

| Node | GPU type | Max memory per requested GPU |
|---|---|---|
| node01 | a100 | 60G |
| node05 | a4000 | 25G |
| node04 | a5000 | 30G |
| node02 | a6000 | 50G |
| node03 | a6000 | 50G |
| node06 | a6000 | 30G |
| node07 | a6000 | 60G |

If a requested `--mem` exceeds the selected node cap, the scheduler must reject
the job before launch rather than silently over-allocating memory.

## Scheduler capacity source

`infra/gpu_scheduler.py status` estimates cluster capacity from Slurm node state plus running job allocations:

- `scontrol show node -o` provides node GPU capacity and drain/down state.
- `squeue -t R` provides running GPU allocations when `AllocTRES` does not include GPU counts.
- Drained, draining, down, failed, and maintenance nodes are treated as unavailable for new jobs.
- `RTX6000` GRES entries are normalized to the project `a6000` GPU type.

## Logging
For every GPU job, record:
- job id
- command
- conda environment
- log path
- result path
- status
