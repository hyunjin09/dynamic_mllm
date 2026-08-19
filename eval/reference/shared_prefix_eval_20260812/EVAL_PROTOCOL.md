# Evaluation Protocol

## 1. Evaluation Questions

이 번들은 서로 다른 세 평가를 포함한다. population과 scorer가 다르므로 결과를 하나의 accuracy로 합치지 않는다.

### A. Core VQA router evaluation

ChartQA, TextVQA, DocVQA에서 Qwen2.5-VL-7B all-on과 frozen SW31 online router를 실제 생성으로 비교한다. 이 평가는 기존 22,349-sample held-out run 중 POPE/SEED를 제외한 12,849개 자유형 VQA population을 정확히 재현한다.

### B. External shared-prefix admission evaluation

MMStar/MMMU 계열에서 입력 image-query를 모든 layer visual-on으로 실행하는 대신, 첫 8개 layer의 공통 dense visual prefix 상태에서 다음 두 continuation 중 하나를 선택했을 때 정확도와 visual-on layer 수가 어떻게 변하는지 평가한다.

### C. POPE hallucination evaluation

POPE adversarial/popular/random에서 all-on과 frozen SW31 online router의 yes/no generation을 비교한다. 이 suite는 object hallucination에 대한 보존과 rescue/harm을 측정하며 shared-prefix admission gate는 적용하지 않는다.

1. **All-on:** 28개 LLM layer 모두 visual contextualization을 허용한다.
2. **Selected sparse:** layer 0-7은 visual-on으로 고정하고, layer 8-27은 frozen SW31 router가 현재 hidden state로 visual-on/off를 결정한다.

Admission gate는 두 생성 답의 정오를 보지 않는다. 공통 prefix hidden-state feature만으로 sparse continuation을 허용할지 결정한다.

## 2. Populations

### 2.1 Core VQA

| internal name | source | split/config | n | metric | correct threshold |
|---|---|---|---:|---|---:|
| chartqa | lmms-lab/ChartQA | test | 2,500 | relaxed accuracy | 1.0 |
| textvqa | lmms-lab/textvqa | validation | 5,000 | EvalAI consensus | 0.5 |
| docvqa | lmms-lab/DocVQA | DocVQA validation | 5,349 | ANLS | 0.5 |
| **Total** | | | **12,849** | | |

Source revisions:

- ChartQA: 9e63b7df1592a1c2158e735cc1725454aef0d6d9
- TextVQA: 9c0699cd19768ac5ab97568f6b3cbac4c0062884
- DocVQA: 539088ef8a8ada01ac8e2e6d4e372586748a265e

Authoritative manifest는 `data/heldout_lmms_recommended_v1/samples.jsonl`이다. 각 UID는 image 한 장을 참조하며 총 12,849개 image가 bundle에 포함된다.

### 2.2 External MMStar/MMMU

| internal name | source | split/config | n |
|---|---|---|---:|
| mmstar_val | Lin-Chen/MMStar | validation | 1,500 |
| mmmu_val | lmms-lab/MMMU | validation, multiple-choice only | 847 |
| mmmu_pro_standard_test | MMMU/MMMU_Pro | standard (10 options) test | 1,730 |
| mmmu_pro_vision_test | MMMU/MMMU_Pro | vision test | 1,730 |
| **Total** | | | **5,807** |

Source cache revisions:

- MMStar: bc98d668301da7b14f648724866e57302778ab27
- MMMU: 364f2e2eb107b36e07ff4c5a15f5947a759cef47
- MMMU-Pro: 1ba55708b8588a8f9b180b8fec9e6435c88ce363

Manifest에는 5,807 unique UID와 6,173 image references가 있다. Multi-image sample은 image_paths 순서대로 interleave한다.

### 2.3 POPE

| internal name | source | split/config | n | metric | correct threshold |
|---|---|---|---:|---|---:|
| pope_adversarial | lmms-lab/POPE | Full/adversarial | 3,000 | yes/no accuracy | 1.0 |
| pope_popular | lmms-lab/POPE | Full/popular | 3,000 | yes/no accuracy | 1.0 |
| pope_random | lmms-lab/POPE | Full/random | 3,000 | yes/no accuracy | 1.0 |
| **Total** | | | **9,000** | | |

Source revision은 `4db1276663dfa5eb8ad16a52d24c31a09e470896`이다. Manifest에는 9,000 UID/image references가 있고, 반복 사용되는 실제 고유 image content hash는 500개다.

## 3. Image-Query Construction

### 3.1 Core VQA

실제 router 평가에는 각 row의 `prompt`를 사용한다. 공통 형식은 다음과 같으며 system message를 추가하지 않는다.

~~~text
{question}
Answer the question using a single word or phrase.
~~~

ChartQA example:

~~~text
How many food item is shown in the bar graph?
Answer the question using a single word or phrase.
~~~

TextVQA example:

~~~text
what is the brand of this camera?
Answer the question using a single word or phrase.
~~~

DocVQA example:

~~~text
What is the ‘actual’ value per 1000, during the year 1975?
Answer the question using a single word or phrase.
~~~

`lmms_eval_default_prompt`와 `lmms_eval_qwen_vl_prompt`도 manifest에 보존하지만, reference SW31 run은 `prompt` 필드를 사용했다. prompt policy를 바꾸면 reference 결과와 직접 비교할 수 없다.

Core image policy:

- ChartQA/TextVQA: 원본 크기, `max_pixels` 제한 없음.
- DocVQA: `max_pixels=802816`, 최대 1,024 image tokens.
- ChartQA/TextVQA `max_new_tokens=16`; DocVQA `max_new_tokens=32`.
- `processor use_fast=False`, `return_mm_token_type_ids=True`.

### 3.2 External MMStar/MMMU

실제 입력 문자열은 data/heldout_mmstar_mmmu_final_v2/samples.jsonl의 prompt 필드다.

### MMStar

~~~text
Which option describe the object relationship in the image correctly?
Options: A: The suitcase is on the book., B: The suitcase is beneath the cat., C: The suitcase is beneath the bed., D: The suitcase is beneath the book.
Answer with the option letter only.
~~~

### MMMU Validation

~~~text
<image 1> Baxter Company has a relevant range ... what are the per unit manufacturing overhead costs incurred?
A. $6
B. $7
C. $8
D. $9
Answer with the option letter only.
~~~

### MMMU-Pro Standard

원본 question과 image placeholder를 유지하고 A-J option을 붙인다.

~~~text
Which of the following best explains the overall trend shown in the <image 1>?
A. Political instability leading to population decline
...
J. Rise of religious conflicts along the Silk Road
Answer with the option letter only.
~~~

### MMMU-Pro Vision

vision config는 question text가 image에 포함되어 generic query와 option을 사용한다.

~~~text
Question: analyze the image and answer the associated question.
Options:
A. ...
...
J. ...
Answer with the option letter only.
~~~

Qwen chat message는 role=user 하나다. image와 text를 placeholder 순서로 배치하고 AutoProcessor.apply_chat_template(add_generation_prompt=True)를 적용한다. system message는 추가하지 않는다.

Image policy:

- MMStar: 원본 image, max_pixels 제한 없음.
- MMMU/MMMU-Pro: max_pixels=802816, 최대 1,024 image tokens.
- processor use_fast=False, return_mm_token_type_ids=True.

### 3.3 POPE

실제 prompt는 다음 형식이다.

~~~text
Is there a snowboard in the image?
Answer the question using a single word or phrase.
~~~

각 row는 image 한 장을 사용한다. `max_pixels` 제한은 없고 `max_new_tokens=128`이다. 긴 generation budget은 기존 reference run과 lmms-eval-style 정책을 보존하기 위한 상한이며, EOS가 나오면 즉시 종료한다.

## 4. Model And Decoding

- Model revision: Qwen2.5-VL-7B-Instruct cc594898137f...
- BF16, SDPA, local-files-only.
- Greedy argmax; row별 `max_new_tokens`는 위 population contract를 따른다. POPE는 128이다.
- repetition_penalty=1.05, float32 logits processing.
- Core VQA all-on/SW31 EOS: [151645].
- POPE all-on/SW31 EOS: [151645].
- External cached all-on EOS: [151645].
- External K=8 selected sparse EOS: [151643,151645].

All-on cache는 paired comparison을 유지하고 반복 비용을 줄이기 위해 재사용한다. 완전한 재생성은 scripts/regenerate_all_on_and_sw31_4gpu.sh로 수행한다.

## 5. Sparse Route

SW31 router contract:

- 28개 binary action: visual-off 또는 visual-on.
- d_model 3,584, hidden MLP 256.
- previous gate feature 사용.
- visual summaries는 mean과 mean-absolute.
- layer embedding 32.
- threshold 0.

Core VQA와 POPE reference는 layer 0부터 SW31을 적용한다. External shared-prefix evaluation에서만 first 8 layers를 visual-on으로 강제한다.

visual-off는 LLM block 전체 skip이 아니다. 해당 layer의 visual contextualization을 차단하는 DVR binary route이며 text stream computation은 유지된다.

## 6. Prefix Admission Gate

이 절은 external MMStar/MMMU 평가에만 적용된다. Core VQA와 POPE reference는 gate 없이 SW31 route 자체를 all-on과 비교한다.

K=8 prefix 종료 시 다음 5개 vector를 각각 L2 normalize하고 concatenate한다.

1. instruction-token mean
2. instruction-token window mean
3. instruction-token last
4. visual-token mean
5. visual-token mean-absolute

Input width는 5 x 3,584 = 17,920이다. benchmark 이름은 입력하지 않는다.

Harm MLP ensemble과 rescue linear ensemble은 seed 17, 41, 73의 확률을 출력한다.

~~~text
harm_ucb   = mean(p_harm) + 2 * std(p_harm)
rescue_lcb = max(0, mean(p_rescue) - 2 * std(p_rescue))
risk       = harm_ucb - 4 * rescue_lcb

use selected sparse iff risk <= -1.5164537767623125
otherwise use all-on
~~~

K, architecture, beta, weight, threshold는 canonical calibration에서 고정되며 external population에서 다시 조정하지 않는다.

## 7. Generation And Selection

### 7.1 Core VQA

`evaluate_heldout_online_visual_router_generation.py`는 각 sample에 대해 all-on과 SW31 route를 생성하고 다음을 저장한다.

- UID, benchmark, GT references
- all-on/router decoded prediction
- benchmark-native score와 thresholded correctness
- 28-bit visual-on mask, on-layer count, transition count
- route logits와 generated token IDs

포함된 all-on cache는 같은 model revision, prompt, processor, decoder, scorer로 생성한 reference다. `USE_CACHED_BASELINE=0`으로 설정하면 all-on도 새 서버에서 다시 생성한다.

### 7.2 External MMStar/MMMU

generate_prefix_hybrid_outcomes.py는 sample마다 selected sparse continuation을 실제 생성하고 다음을 저장한다.

- generated IDs와 decoded answer
- task score와 correctness
- 28-bit visual-on mask
- selected layer count와 transition count
- K=8 prefix features
- all-on 대비 preserve/harm/rescue/unsolved outcome

evaluate_prefix_admission_external.py는 frozen gate를 적용한다. 구현상 두 outcome을 cache한 뒤 gate decision으로 조합하지만 gate는 정오 label을 입력받지 않는다. validate_prefix_runtime_equivalence.py가 동일 callback을 실제 one-pass generation에 적용해 offline 조합과 동치인지 benchmark별 16개 sample로 검사한다.

| all-on | sparse | label |
|---:|---:|---|
| correct | correct | preserve |
| correct | wrong | harm |
| wrong | correct | rescue |
| wrong | wrong | unsolved |

### 7.3 POPE

Core VQA와 동일한 evaluator를 사용하되 `data/heldout_pope_v1/samples.jsonl`과 `pope_yes_no_accuracy` scorer를 사용한다. 결과는 `results/reference_pope/`에 있으며 all-on 87.98%, SW31 86.18%, 평균 visual-on layer 17.47개다.

## 8. Scoring

### 8.1 Core VQA

구현은 `code/dvr_qwen/eval_metrics.py`이며 `score`와 `correct`를 구분한다.

ChartQA relaxed accuracy:

~~~text
numeric: score = 1[abs(pred - GT) / abs(GT) <= 0.05]
text:    score = 1[lower(strip(pred)) == lower(GT)]
correct = 1[score >= 1.0]
~~~

TextVQA EvalAI consensus는 prediction과 10개 human reference를 EvalAI 방식으로 normalize한다. 각 reference를 하나씩 hold out한 뒤 나머지 9개 중 prediction과 일치하는 수를 `min(1, matches/3)`로 계산하고 10개 값을 평균한다.

~~~text
score = mean_i min(1, matches(pred, refs excluding i) / 3)
correct = 1[score >= 0.5]
~~~

DocVQA ANLS는 여러 GT 중 가장 작은 normalized Levenshtein distance를 사용한다.

~~~text
d = min_gt edit_distance(lower(pred), lower(gt)) / max(len(pred), len(gt))
score = 1 - d, if 1 - d >= 0.5; otherwise 0
correct = 1[score >= 0.5]
~~~

따라서 TextVQA/DocVQA의 benchmark mean score와 thresholded correct rate는 다른 통계다. `results/reference_core_vqa/report.md`는 둘을 모두 보고한다. 세 benchmark의 combined row는 12,849개 sample micro-average이며 benchmark macro-average가 아니다.

### 8.2 External MMStar/MMMU

구현은 code/dvr_qwen/eval_metrics.py의 multiple_choice_accuracy 함수다.

Generation을 uppercase로 바꾸고 regex \b([A-J])\b에 처음 매칭되는 standalone letter를 예측으로 사용한다.

~~~text
pred_letter(y) = first standalone A-J in uppercase(y)
score_i = 1[pred_letter(y_i) == GT_i]
correct_i = 1[score_i >= 1.0]
accuracy = sum(correct_i) / N
~~~

모든 benchmark가 correctness threshold 1.0을 쓴다. Prompt가 option-letter-only 출력을 요구하는 이유는 설명 문장 속 standalone letter가 잘못 추출되는 경우를 줄이기 위해서다.

보고 metric:

- all-on, ungated sparse, learned admission accuracy
- oracle upper bound와 matched-random admission
- paired accuracy delta
- harm, rescue, preservation
- sparse route fraction
- mean visual-on layers
- 5,000 UID bootstrap CI
- benchmark별 동일 통계

### 8.3 POPE

~~~text
normalized_pred = "yes" if lower(strip(pred)).startswith("yes")
                  "no"  if lower(strip(pred)).startswith("no")
                  lower(strip(pred)) otherwise
score = 1[normalized_pred == lower(GT) and GT in {"yes", "no"}]
correct = 1[score >= 1.0]
~~~

세 split은 각각 보고하고, combined POPE는 9,000개 micro-average로만 사용한다.

## 9. Commands

~~~bash
python3 scripts/verify_bundle.py

# ChartQA + TextVQA + DocVQA; cached all-on을 사용
PYTHON=$PWD/.venv/bin/python NUM_GPUS=4 scripts/run_core_vqa_eval_4gpu.sh
PYTHON=$PWD/.venv/bin/python scripts/compare_results.py --suite core-vqa

# all-on까지 실제로 재생성
PYTHON=$PWD/.venv/bin/python NUM_GPUS=4 USE_CACHED_BASELINE=0 \
  RUN_ID=reproduced_core_vqa_full scripts/run_core_vqa_eval_4gpu.sh

# External MMStar/MMMU shared-prefix admission
PYTHON=$PWD/.venv/bin/python NUM_GPUS=4 scripts/run_eval_4gpu.sh

# POPE adversarial + popular + random
PYTHON=$PWD/.venv/bin/python NUM_GPUS=4 scripts/run_pope_eval_4gpu.sh
PYTHON=$PWD/.venv/bin/python scripts/compare_results.py --suite pope
PYTHON=$PWD/.venv/bin/python scripts/run_scoring_only.sh
PYTHON=$PWD/.venv/bin/python NUM_GPUS=4 scripts/regenerate_all_on_and_sw31_4gpu.sh
~~~

## 10. Required Validity Checks

1. Bundle checksum, UID, image audit 통과.
2. 5,807 UID가 정확히 한 번씩 생성됨.
3. 모든 selected mask의 first 8 bit가 1.
4. baseline cache와 manifest의 benchmark, metric, threshold 일치.
5. baseline prediction 재채점이 stored score/correctness와 일치.
6. runtime audit에서 decision, route, prediction, score, correctness 일치.
7. mean visual-on layers를 FLOPs 또는 latency로 직접 해석하지 않음.
8. Core VQA manifest가 12,849 unique UID와 12,849 image references를 가짐.
9. Core benchmark/metric/threshold가 각각 ChartQA/relaxed/1.0, TextVQA/consensus/0.5, DocVQA/ANLS/0.5와 일치함.
10. Core all-on cache를 현재 scorer로 전수 재채점했을 때 stored score/correctness와 일치함.
11. Core 재생성 결과는 `results/reference_core_vqa/heldout_generation_rows.jsonl`과 UID별로 비교함.
12. `source_manifest_alignment.json`이 reference run의 원본 plus manifest와 portable core manifest의 UID, prompt, GT, image hash, metric, decoding field 14개가 전부 일치함을 확인함.
13. POPE manifest가 split별 3,000개, 총 9,000 unique UID와 image references를 가짐.
14. POPE 9,000개가 `pope_yes_no_accuracy`, threshold 1.0, max_new_tokens 128을 사용함.
15. POPE all-on/router prediction을 현재 scorer로 전수 재채점했을 때 stored score/correctness와 일치함.
