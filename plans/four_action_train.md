# Four-Action Online Router: Architecture and Training Plan

## 1. Objective

Train a lightweight router on the newly generated four-action route labels.

The router should decide, **at each decoder layer and from the current routed hidden state**, whether visual computation should be:

- `FULL`      : READ=1, WRITE=1
- `READ_ONLY` : READ=1, WRITE=0
- `WRITE_ONLY`: READ=0, WRITE=1
- `IGNORE`    : READ=0, WRITE=0

The main hypothesis is:

> The usefulness of visual READ and WRITE is trajectory-dependent. Therefore, the routing decision at layer `l` should depend on the actual visual/text hidden states produced by the previously executed routing prefix, rather than only on the original image/query embeddings.

This is intentionally different from a POLAR-style upfront program predictor.

The first goal is not to maximize router complexity. The first goal is to test whether **online routed hidden states + function-specific READ/WRITE features + layer-specific learned queries** are sufficient to predict useful four-action trajectories.

---

## 2. Scientific Motivation

Previous analysis established three important properties.

### 2.1 READ and WRITE are functionally distinct

A binary visual `ON/OFF` action conflates:

- visual evidence access by the text stream (`READ`)
- visual-state refinement/update (`WRITE`)

Four-action routing exposes these separately.

### 2.2 READ and WRITE have different depth structure

Route-conditioned analysis showed that WRITE-mediated and READ-mediated corrections occur at substantially different depths.

This motivates explicit layer identity / layer-specific parameters rather than a completely layer-agnostic router.

### 2.3 Action utility is strongly trajectory-dependent

Many suppressions that are necessary inside a successful correcting trajectory do not produce a standalone correction when intervened on under the all-FULL context.

Therefore the router should observe:

```text
current routed state
```

rather than predict all 28 actions once from the original input.

---

## 3. Routing Formulation

Let the routed hidden state immediately before decoder layer `l` be:

```text
T_l : current text/control hidden states
V_l : current visual hidden states
```

The router chooses:

```text
a_l ∈ {FULL, READ_ONLY, WRITE_ONLY, IGNORE}
```

Then layer `l` executes under `a_l`, producing:

```text
T_{l+1}, V_{l+1}
```

The next router decision is made from this newly routed state.

Conceptually:

```text
(V_0, T_0)
    ↓
Router at layer 0
    ↓
action_0
    ↓
execute layer 0
    ↓
(V_1, T_1)
    ↓
Router at layer 1
    ↓
action_1
    ↓
...
```

The policy is therefore:

```text
a_l = πθ(V_l, T_l, l)
```

not:

```text
(a_0, ..., a_27) = πθ(initial input only)
```

---

## 4. Router Input State

### 4.1 Text state

Use the **current hidden state of the final query/text-control token** at layer `l`.

Denote:

```text
q_l = h_l[last_query_text_position]
```

Important:

- this is NOT the static input embedding;
- it is the contextualized hidden state after the routed prefix;
- identify the exact token position from the current Qwen2.5-VL input/executor contract;
- do not assume a token index without auditing the current prompt/token layout.

No text pooling is required in V1.

The intended interpretation is:

> `q_l` is the current reasoning/task state carried by the text stream.

---

## 5. Layer-Specific Parameters

Do not build 28 fully independent large routers.

Use shared routing modules with **layer-specific learned queries**.

For each decoder layer `l`, learn:

```text
e_R[l] : READ-specific layer query/embedding
e_W[l] : WRITE-specific layer query/embedding
```

These are separate because READ and WRITE play different roles across depth.

The shared router learns general routing principles, while `e_R[l]` and `e_W[l]` encode layer-specific functional roles.

---

## 6. READ Branch

### 6.1 Question

The READ branch asks:

> Given the current reasoning state, should the text stream directly consume the currently available visual evidence at this layer?

Text-to-visual querying is therefore appropriate.

### 6.2 Architecture

Project the current text state:

```text
qR_l = Wq_R(q_l) + e_R[l]
```

Project visual tokens to a small router dimension:

```text
K_R = Wk_R(V_l)
U_R = Wv_R(V_l)
```

Use a small single-query/multi-head attention:

```text
vR_l = Attn(qR_l, K_R, U_R)
```

Then form:

```text
zR_l = MLP_R([Proj_T(q_l), vR_l, e_R[l]])
```

Interpretation:

```text
q_l
  ↓
layer-conditioned READ query
  ↓
attend to current visual tokens
  ↓
representation of what visual evidence the current reasoning state would consume
```

Recommended initial router dimension:

```text
d_router = 256
```

Recommended attention:

```text
1 query
4 heads or fewer
```

Do not use a large Transformer router in V1.

---

## 7. WRITE Branch

### 7.1 Question

The WRITE branch asks:

> Given the current visual representation, should this decoder layer be allowed to further update/refine the visual state for the current task?

The native WRITE operation should not be modeled as another text-query-to-vision READ operation.

Therefore do not use `q_l` as the visual pooling query in the WRITE branch.

### 7.2 Architecture

Use the layer-specific WRITE query directly:

```text
qW_l = e_W[l]
```

Project the visual states:

```text
K_W = Wk_W(V_l)
U_W = Wv_W(V_l)
```

Pool the visual state using the layer-specific WRITE query:

```text
vW_l = Attn(qW_l, K_W, U_W)
```

Then condition the WRITE decision on the current task/reasoning state:

```text
zW_l = MLP_W([vW_l, Proj_TW(q_l), e_W[l]])
```

Interpretation:

```text
V_l
 ↓
layer-specific visual-state query e_W[l]
 ↓
summary of the current visual representation
 +
current task/reasoning state q_l
 ↓
estimate whether another visual update is appropriate
```

Important distinction:

- `q_l` is a QUERY into visual tokens for READ.
- `q_l` is only a CONDITIONER after visual pooling for WRITE.

This distinction should be preserved in the implementation.

---

## 8. Structured Four-Action Decision Head

Do not make READ and WRITE completely independent binary decisions.

Use:

```text
zR_l
zW_l
```

to produce:

1. READ unary score
2. WRITE unary score
3. a small READ×WRITE interaction residual

One practical implementation:

```text
s_R = Head_R(zR_l)          # scalar utility/logit for READ
s_W = Head_W(zW_l)          # scalar utility/logit for WRITE
r_RW = Head_I([zR_l,zW_l])  # 4 interaction residuals
```

Construct four joint logits:

```text
logit(FULL)       = +s_R +s_W + r_RW[1,1]
logit(READ_ONLY)  = +s_R -s_W + r_RW[1,0]
logit(WRITE_ONLY) = -s_R +s_W + r_RW[0,1]
logit(IGNORE)     = -s_R -s_W + r_RW[0,0]
```

Then:

```text
p(a_l | state_l) = softmax(joint_logits)
```

The exact parameterization can be implemented equivalently, but preserve the conceptual decomposition:

```text
READ utility
+
WRITE utility
+
small joint interaction
```

rather than using four unrelated classifiers.

---

## 9. Parameter Sharing

V1 should use shared modules across all 28 layers.

Shared:

```text
Wq_R
Wk_R
Wv_R
Wk_W
Wv_W
Proj_T
Proj_TW
MLP_R
MLP_W
Head_R
Head_W
Head_I
```

Layer-specific:

```text
e_R[0...27]
e_W[0...27]
```

Do not introduce 28 separate full router networks in V1.

Layer-specific output heads can be tested later as an ablation if necessary.

---

## 10. Backbone

Use the same frozen Qwen2.5-VL backbone and exact unified executor used for four-action label generation.

The base MLLM must remain frozen.

Train only:

```text
router projections
READ branch
WRITE branch
interaction head
layer-specific READ/WRITE queries
```

During supervised teacher-forced route replay:

- Qwen computations can run under `torch.no_grad()` / detached hidden states;
- gradients should flow only through router modules;
- verify that routing states exactly match the four-action executor semantics used to generate labels.

---

## 11. Training Label Sources

Use the newly generated four-action labels from the final authoritative output root, expected to be under:

```text
datasets/mcts_labels_4action/
```

Audit the exact manifests before training.

Expected datasets:

```text
GQA
TextVQA
ChartQA
WeMath2.0 Standard
WeMath2.0 Pro
```

Preserve route metadata:

```text
W2C
C2C
all_off_seed
source binary route
route provenance
```

Do not silently mix incompatible source versions.

---

## 12. W2C and C2C Semantics

### 12.1 W2C

```text
FULL = wrong
four-action route = correct
```

Corrective supervision.

### 12.2 C2C

```text
FULL = correct
four-action route = correct
```

Correctness-preserving / efficiency supervision.

Do not describe C2C suppression as answer-unaligned.

Keep W2C and C2C identifiable in training and evaluation.

---

## 13. Multi-Route Supervision

A sample may have multiple valid four-action routes.

Do NOT collapse them into a single arbitrary ground-truth route.

Build a per-sample **prefix trie** over the valid four-action routes.

Each trie node corresponds to:

```text
one exact routing prefix
```

and therefore one deterministic routed hidden state.

The outgoing edges define:

```text
valid next actions for that state
```

This is the preferred supervision representation.

---

## 14. Training-Time Route Replay

Do not train the online router from all-FULL hidden states.

For each training sample:

1. select one valid four-action route from the sample's training route set;
2. replay that route from layer 0 to 27 using the frozen unified executor;
3. before each layer obtain the actual routed:
   - `q_l`
   - `V_l`
4. evaluate the router on this state;
5. obtain the valid outgoing action set for the current routing prefix from the trie;
6. compute the set-valued action loss;
7. execute the sampled teacher-forced action;
8. continue to the next layer.

Thus:

```text
training state at layer l
=
state actually produced by the route prefix used during training
```

This is critical.

Do not use:

```text
all-FULL state + route label
```

as the primary online-router training input.

---

## 15. Route Sampling per Epoch

Do not replay every valid route for every sample at every epoch.

Instead:

```text
sample first
→ sample one valid route for that sample
→ teacher-force/replay that route
```

Across epochs, different valid routes can be sampled.

Use deterministic epoch-dependent seeds.

Prefer sample-uniform sampling before route sampling so samples with many routes do not receive disproportionate training weight.

---

## 16. Set-Valued Action Loss

At a trie node/state `s_l`, let:

```text
A_valid(s_l)
```

be all valid next actions among labels sharing that exact prefix.

Use:

```text
L_action(s_l)
=
-log Σ_{a ∈ A_valid(s_l)} p(a | s_l)
```

Examples:

If only `READ_ONLY` is valid:

```text
L = -log p(READ_ONLY)
```

If both `READ_ONLY` and `WRITE_ONLY` are valid:

```text
L = -log [p(READ_ONLY) + p(WRITE_ONLY)]
```

Do not penalize one valid action merely because another valid action was selected for teacher forcing.

---

## 17. Sample / Route-Type Balancing

Avoid training directly over raw route occurrences.

Use sample-level sampling.

Recommended V1:

```text
50% W2C samples
50% C2C samples
```

when both pools are available.

Within each pool, use a documented dataset-balanced sampler so one benchmark does not dominate only because it has more samples.

Do not oversample samples merely because they have many valid routes.

If exact 50/50 balancing is impractical because of pool size, use the closest stable balanced sampler and document it.

---

## 18. Class Imbalance

FULL actions will likely be more common than non-FULL actions.

Do not immediately add aggressive class weighting.

For V1:

1. train with the set-valued action loss on all 28 layer decisions;
2. monitor predicted action distribution;
3. monitor minority-action recall;
4. monitor W2C rescue and C2C preservation.

Only add action-frequency weighting if the router clearly collapses to FULL.

If weighting is needed, use a bounded train-split-derived weighting scheme and keep an unweighted baseline.

---

## 19. Initial Training Hyperparameters

Stay close to the previous binary POLAR-style router training protocol.

First audit the exact current binary-router training configuration in the repository and use it as the authoritative reference.

Target V1:

```text
epochs: 10
optimizer: AdamW
learning rate: 5e-4
scheduler: cosine
warmup: same exact setting as previous binary router
weight decay: same exact setting as previous binary router unless a clear reason exists
precision: bf16 where applicable
backbone: frozen
router: trainable
```

### Effective batch size

Target:

```text
effective sample batch size ≈ 128
```

if practical, using gradient accumulation.

Because the online router replays frozen Qwen trajectories, the physical per-GPU batch may need to be much smaller than the previous embedding-only predictor.

Benchmark a safe physical batch size first, then use gradient accumulation.

Do not force physical batch 128.

---

## 20. Distributed Training

Use all 8 H100 GPUs for the main run.

Preferred:

```text
8-way DDP
1 training process / GPU
```

Each rank:

- owns one frozen Qwen replica;
- owns the router replica;
- processes its local mini-batch;
- runs Qwen route replay under no-grad;
- backpropagates only router gradients;
- synchronizes router gradients through DDP.

Do not use 2 model replicas/GPU for the first training run unless a separate throughput benchmark demonstrates an actual gain.

---

## 21. Training Smoke Test

Before the 10-epoch run, perform a small training smoke test using approximately:

```text
8 samples
```

covering multiple datasets and W2C/C2C.

Verify:

1. routed hidden states change according to teacher-forced previous actions;
2. READ branch uses `q_l` as visual query;
3. WRITE branch uses `e_W[l]` as visual pooling query;
4. `e_R[l]` and `e_W[l]` receive gradients;
5. backbone receives no gradients;
6. router receives gradients;
7. set-valued loss is finite;
8. a state with two valid actions does not penalize either valid branch;
9. loss decreases on repeated tiny-batch optimization;
10. checkpoint save/load works;
11. deterministic route sampling works.

If this fails, fix before full training.

---

## 22. Checkpointing

Save:

```text
epoch_1
...
epoch_10
best
last
```

Preserve:

```text
router weights
optimizer state
scheduler state
epoch
global step
training sampler seed/state
architecture config
label manifest hash
split manifest hash
git commit
```

---

## 23. Validation Metrics

Do not select checkpoints only by node-level action accuracy.

The real objective is successful routed execution.

### 23.1 Node-level metrics

Report:

```text
Valid-Action@1
negative log valid-action probability
action distribution
per-action precision/recall/F1
READ-bit accuracy
WRITE-bit accuracy
```

For multi-valid states, prediction is correct if:

```text
predicted action ∈ A_valid(state)
```

### 23.2 Route-level / execution metrics

Run the router online through frozen Qwen.

For W2C:

```text
rescue rate:
FULL wrong -> routed correct
```

For C2C:

```text
preservation rate:
FULL correct -> routed correct
```

Combined:

```text
routed accuracy
number of regressions
number of rescues
net accuracy change
```

Also report:

```text
mean FULL count
mean READ_ONLY count
mean WRITE_ONLY count
mean IGNORE count
READ suppression count
WRITE suppression count
estimated compute
actual latency
router overhead
```

Actual routed execution is primary.

A predicted route absent from the cached label set may still be valid.

Do not use cached-route membership as the final correctness criterion.

---

## 24. Checkpoint Selection

Use validation execution behavior.

Primary balanced checkpoint score:

```text
0.5 * W2C_rescue_rate
+
0.5 * C2C_preservation_rate
```

Tie-break:

1. higher overall routed validation accuracy
2. fewer FULL-correct regressions
3. lower router-adjusted compute / latency

Record all epoch metrics.

---

## 25. Main Baselines

After the main online router is stable, compare against at least:

### A. POLAR-style upfront router

Input:

```text
initial image/query representations
+ layer-specific learnable embeddings/queries
```

Predict the whole four-action program without seeing routed intermediate states.

Keep parameter count reasonably matched.

Purpose:

> Test whether current routed hidden states contain useful routing information beyond the original input.

### B. Generic online router

Use current routed hidden states but do not separate READ- and WRITE-specific feature extraction.

Purpose:

> Test whether the function-specific architecture matters beyond online routing itself.

Do not implement a large baseline suite before the main model works.

---

## 26. Key Architecture Ablations

After the main pipeline is stable, run a bounded set.

### Layer specificity

1. no layer embedding/query
2. one shared layer embedding `e_l`
3. separate `e_R[l]`, `e_W[l]` <- main
4. optional layer-specific output heads

### READ representation

1. text state only
2. READ query over visual state <- main

### WRITE representation

1. visual state only
2. visual state + text conditioning <- main
3. text-query-to-visual WRITE variant as comparison

### Decision head

1. generic 4-way MLP
2. READ unary + WRITE unary + interaction <- main

---

## 27. Expected Main Claim if the Method Works

The intended claim is:

> Visual routing should be conditioned on the current multimodal state produced by previous routing decisions. READ and WRITE require distinct routing signals: READ is a text-conditioned evidence-access decision, while WRITE is a layer-specific visual-state refinement decision conditioned on the current task.

A strong result would be:

```text
online routed-state router
>
POLAR-style upfront router
```

especially on W2C rescue while preserving C2C correctness.

That would support:

> Intermediate routed states contain information necessary for deciding future visual computation.

---

## 28. Interpretation Boundary

Do not claim the router identifies the unique root cause of every error.

The four-action labels are route-conditioned corrective programs.

A later READ may be harmful because of an earlier WRITE decision.

The router is learning:

```text
what action is useful from the current routed state
```

not necessarily:

```text
where the original causal corruption first originated
```

---

## 29. Implementation Phases

### Phase 0 — Audit

Inspect:

- final four-action label manifests
- current train/validation splits
- previous binary POLAR-style router/training code
- current unified four-action executor
- exact Qwen query/text token position
- optimizer/scheduler hyperparameters

Write:

```text
analysis/4action_router/implementation_audit.md
```

### Phase 1 — Router implementation

Implement:

- routed-state extraction
- READ branch
- WRITE branch
- separate layer queries
- structured joint head
- four-action online execution
- prefix-trie supervision
- set-valued action loss

### Phase 2 — 8-sample train smoke

Verify semantics, gradients, loss, and checkpointing.

### Phase 3 — Label / trie audit

Report:

- samples per dataset
- W2C/C2C counts
- unique routes/sample
- trie nodes/sample
- valid-action multiplicity
- action frequency by layer
- READ/WRITE suppression frequency

### Phase 4 — 10-epoch main training

Use 8 H100s.

### Phase 5 — Epoch-by-epoch validation

Run online execution validation for every saved epoch or at a fixed documented interval.

### Phase 6 — Final evaluation

Evaluate:

- overall
- per dataset
- W2C
- C2C
- ALL-OFF-seed W2C separately if present

### Phase 7 — Baselines / ablations

Only after the main pipeline is stable.

---

## 30. Required Output Files

Suggested output root:

```text
analysis/4action_router/
```

Create:

```text
implementation_audit.md
label_and_trie_audit.md
training_config.json
training_log.jsonl
epoch_metrics.csv
checkpoints/
main_router_results.md
execution_eval/
figures/
```

The final report should include:

1. exact architecture
2. parameter count
3. training hyperparameters
4. label counts
5. W2C/C2C sampling
6. action distributions
7. training/validation curves
8. best epoch
9. W2C rescue
10. C2C preservation
11. overall accuracy
12. regression/rescue counts
13. compute/action statistics
14. router latency overhead
15. dataset-wise breakdown
16. failure analysis
17. comparison with binary router if available
18. comparison with POLAR-style upfront router when implemented
19. ablation results when available

---

## 31. Stop / Failure Conditions

Do not continue blindly if:

- executor semantics differ from label-generation semantics;
- label routes fail replay at unexpectedly high rates;
- backbone receives gradients;
- route sampling is not reproducible;
- multi-valid states are reduced to arbitrary single labels;
- router collapses to one action with near-zero minority recall;
- online execution validation is substantially inconsistent with offline node metrics.

Diagnose first.

---

## 32. Initial Training Decision

The first scientific run should be:

```text
Model:
    online state-conditioned four-action router

Architecture:
    last current text hidden state q_l
    READ branch: text-conditioned visual attention
    WRITE branch: layer-specific visual pooling + text conditioning
    separate layer-specific queries e_R[l], e_W[l]
    structured READ/WRITE + interaction four-action head

Backbone:
    frozen Qwen2.5-VL

Training:
    teacher-forced route replay
    prefix-trie multi-valid supervision
    set-valued action loss
    10 epochs
    AdamW
    LR 5e-4
    cosine schedule
    previous binary-router warmup/weight-decay settings after audit
    effective batch ≈128 if practical
    8-H100 DDP

Primary validation:
    W2C rescue
    C2C preservation
    overall routed accuracy
```

Do not start with a larger router or reinforcement learning.

First establish whether the current routed hidden state is sufficient to predict the four-action labels and whether online routing improves actual execution behavior.
