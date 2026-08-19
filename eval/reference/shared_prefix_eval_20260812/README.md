# Shared-Prefix Visual Router Evaluation Reproduction Bundle

이 디렉터리는 **Qwen2.5-VL-7B + SW31 router**의 core VQA 평가와 **all-on fallback admission gate**의 외부 평가를 다른 서버에서 재현하기 위한 독립 번들이다. 모델, 이미지, image-query manifest, 고정 체크포인트, all-on baseline 결과, generation/routing/scoring 코드가 모두 포함되어 있다.

세 suite를 분리한다.

1. **Core VQA:** ChartQA test + TextVQA val + DocVQA val, 총 12,849개에서 all-on과 SW31 route를 비교한다.
2. **External admission:** canonical population에서 선택한 gate를 고정한 뒤 MMStar/MMMU 계열 UID-disjoint population 5,807개에서 실제 generation 성능을 잰다.
3. **POPE:** adversarial/popular/random 총 9,000개에서 all-on과 SW31 route의 yes/no accuracy를 비교한다.

## Quick Start

~~~bash
cd shared_prefix_eval_20260812
python3 scripts/verify_bundle.py

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124
python -m pip install -r environment/requirements_runtime.txt

PYTHON=$PWD/.venv/bin/python NUM_GPUS=4 scripts/run_eval_4gpu.sh
~~~

Core VQA 평가는 별도 명령으로 실행한다.

~~~bash
PYTHON=$PWD/.venv/bin/python NUM_GPUS=4 scripts/run_core_vqa_eval_4gpu.sh
PYTHON=$PWD/.venv/bin/python scripts/compare_results.py --suite core-vqa

PYTHON=$PWD/.venv/bin/python NUM_GPUS=4 scripts/run_pope_eval_4gpu.sh
PYTHON=$PWD/.venv/bin/python scripts/compare_results.py --suite pope
~~~

48 GB GPU 기준 memory cap은 40 GiB이다. 24-32 GB GPU에서는 다음처럼 낮춘다.

~~~bash
PYTHON=$PWD/.venv/bin/python \
NUM_GPUS=4 FIRST_GPU_MAX_MEMORY_GB=28 MIN_FREE_GB=16 \
scripts/run_eval_4gpu.sh
~~~

## Frozen Contract

| item | contract |
|---|---|
| Base model | Qwen/Qwen2.5-VL-7B-Instruct |
| HF revision | cc594898137f460bfe9f0759e9844b3ce807cfb5 |
| Router | sw31_bt_leg_s41, epoch 1, threshold 0 |
| Router SHA256 | 6ecf2f...af255 |
| Shared prefix | first K=8 LLM layers are forced visual-on |
| Admission SHA256 | 3a1385...7ef4 |
| Core population | ChartQA test + TextVQA val + DocVQA val |
| Core total/scoring | 12,849 samples; relaxed accuracy / EvalAI consensus / ANLS |
| External population | MMStar val + MMMU val MC + MMMU-Pro standard/vision test |
| External total/scoring | 5,807 samples; first standalone A-J exact correctness |
| POPE population | adversarial + popular + random |
| POPE total/scoring | 9,000 samples; yes/no accuracy |

전체 상수와 checksum은 [execution_contract.json](environment/execution_contract.json)에 있다.

## Layout

~~~text
shared_prefix_eval_20260812/
├── README.md
├── EVAL_PROTOCOL.md
├── environment/
├── model/Qwen2.5-VL-7B-Instruct_cc594898.../
├── data/heldout_mmstar_mmmu_final_v2/
│   ├── samples.jsonl
│   └── images/
├── data/heldout_lmms_recommended_v1/
│   ├── samples.jsonl
│   └── images/
├── data/heldout_pope_v1/
│   ├── samples.jsonl
│   └── images/
├── checkpoints/
│   ├── sw31/router_epoch_001.pt
│   └── prefix_admission/
├── baseline/
│   ├── all_on_generation_rows.jsonl
│   ├── all_on_summary.json
│   ├── core_vqa_all_on_generation_rows.jsonl
│   └── pope_all_on_generation_rows.jsonl
├── code/
├── scripts/
└── results/
~~~

**samples.jsonl**이 sample/image/query/GT의 authoritative source다. 원래 서버 절대 image path는 provenance로 남아 있지만, 실행 시 bundle 내부 images 경로로 자동 remap된다.

## Core VQA Outputs

| path | content |
|---|---|
| results/core_vqa_regeneration/`RUN_ID`/merged_final/heldout_generation_rows.jsonl | UID별 all-on/router 생성, 점수, mask |
| results/core_vqa_regeneration/`RUN_ID`/merged_final/summary.json | benchmark별 accuracy/score/layer 통계와 CI |
| results/reference_core_vqa/heldout_generation_rows.jsonl | 기존 SW31 reference 12,849개 |
| results/reference_core_vqa/report.md | reference 결과 요약 |

기본값은 포함된 all-on cache를 재사용한다. all-on까지 새 서버에서 다시 생성하려면 다음을 실행한다.

~~~bash
PYTHON=$PWD/.venv/bin/python NUM_GPUS=4 USE_CACHED_BASELINE=0 \
  RUN_ID=reproduced_core_vqa_full scripts/run_core_vqa_eval_4gpu.sh
~~~

## Main Outputs

| path | content |
|---|---|
| results/reproduced_prefix_hybrid/prefix_08/*.pt | 실제 sparse generation, mask, K=8 prefix features |
| results/reproduced_prefix_admission_eval/external_predictions.jsonl | UID별 correctness와 gate decision |
| results/reproduced_prefix_admission_eval/summary.json | accuracy, harm/rescue, routing, bootstrap CI |
| results/reproduced_prefix_admission_eval/report.md | 최종 표 |
| results/reproduced_prefix_admission_eval/result.png | 요약 그림 |
| results/reproduced_prefix_admission_eval/runtime_equivalence.json | offline 조합과 one-pass runtime 동치 감사 |

POPE reference와 재생성 결과는 각각 `results/reference_pope/`와 `results/pope_regeneration/`에 저장된다.

이미 hybrid cache가 있다면 generation 없이 재채점만 할 수 있다.

~~~bash
PYTHON=$PWD/.venv/bin/python scripts/run_scoring_only.sh
~~~

## Transfer

전체 apparent size는 약 27 GB이다.

~~~bash
rsync -a --info=progress2 \
  /mnt/hyemin/10k_dataset_mask/reproduction_bundles/shared_prefix_eval_20260812/ \
  USER@NEW_SERVER:/path/shared_prefix_eval_20260812/
~~~

전송 후 반드시 다음을 실행한다.

~~~bash
python3 scripts/verify_bundle.py
python3 scripts/verify_bundle.py --full-images --full-model
~~~

두 번째 명령은 이미지와 16 GB model shard의 SHA256까지 계산하므로 더 오래 걸린다.

## Scope And Caveats

- 이 번들은 **evaluation 재현용**이다. admission gate 재학습용 canonical feature cache는 포함하지 않았다. 선택 checkpoint, split UID, canonical predictions, selection summary는 포함했다.
- Core VQA와 external multiple-choice suite는 prompt와 scorer가 다르므로 하나의 overall accuracy로 합치지 않는다.
- POPE도 별도 yes/no hallucination suite이며 Core VQA나 external MC accuracy와 합산하지 않는다.
- external population은 canonical train/calibration과 UID-disjoint지만, 프로젝트 선행 분석에서 이미 확인한 benchmark다. publication용 untouched test라는 의미는 아니다.
- mean visual-on layers는 route-sensitive compute proxy이지 실제 FLOPs/latency가 아니다.
- exact current protocol에서 cached all-on EOS는 [151645], K=8 hybrid EOS는 [151643,151645]다. 둘 다 repetition penalty 1.05를 float32 logits에 적용한다. 후속 strict decoder-policy ablation에서는 EOS 집합 통일 여부를 확인해야 한다.

세부 image-query, routing, gate 수식, scoring은 [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)에 기록했다.
