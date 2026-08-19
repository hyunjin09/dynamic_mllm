# Infra

GPU jobs and CPU-heavy jobs must use:

```text
infra/gpu_scheduler.py
```

## CPU-only computational job

Use `--gpus 0`.

```bash
python infra/gpu_scheduler.py add --project . --id EXP_ID --exp-id EXP_ID --gpu-type auto --gpus 0 --command "COMMAND" --log-path runs/EXP_ID/slurm.log --result-path outputs/EXP_ID/result.json
python infra/gpu_scheduler.py plan --project .
python infra/gpu_scheduler.py launch --project . --execute
```

CPU-only jobs prefer:

```text
a4000
a5000
a6000
a100
```

## GPU job

Use `--gpus 1` or more.

```bash
python infra/gpu_scheduler.py add --project . --id EXP_ID --exp-id EXP_ID --gpu-type auto --gpus 1 --command "COMMAND" --log-path runs/EXP_ID/slurm.log --result-path outputs/EXP_ID/result.json
python infra/gpu_scheduler.py plan --project .
python infra/gpu_scheduler.py launch --project . --execute
```

Memory requests are capped by selected node. For one requested GPU, the maximum
`--mem` values are: node01 60G, node05 25G, node04 30G, node02 50G, node03 50G,
node06 30G, node07 60G. The scheduler rejects requests above the selected node
cap.

## Dataset cache check

Check these two roots:

```bash
python infra/check_dataset_cache.py --dataset DATASET_NAME --roots /data/dataset /home/hyunjin/.cache/huggingface/datasets --report outputs/datasets/DATASET_NAME/cache_check.json
```

## HF dataset download

One dataset at a time. Use `num_proc <= 3`.

```bash
python infra/download_hf_dataset.py --dataset DATASET_ID --cache-dir /data/dataset/huggingface/datasets --num-proc 3 --report outputs/datasets/DATASET_ID/download_report.json
```

Submit the download command through `infra/gpu_scheduler.py` with `--gpus 0`.
