# Dense answer-logit emergence analysis

- Frozen contract: `73daef6c851378784aa498493b152a49ec5fa864c66b53b033f29749e606c99b`
- Complete population: 7,999/7,999 unique UIDs; all logits finite.
- Outcomes: 3,999 current-dense correct and 4,000 current-dense wrong.
- Readout: `model.model.language_model.norm then model.lm_head` applied to `text_final` at layers 0–27.
- Emergence: strict raw-logit delta 1 for 3 consecutive layers; early cutoff layer 4.

## Validity finding

The requested fixed-head calculation completed exactly, but it does **not**
measure answer-start competition at the saved position. `text_final` is the
last literal user-prompt token. The next token there is a chat-template
delimiter, not the first assistant answer token. Consistent with that contract
mismatch, correct samples have a very large negative GT-vs-top-vocabulary
margin and 0/3,999 persistent GT-emergence events.

The single prespecified cheap validity diagnostic retained the same readout and
checked all 54 correct records in one frozen source shard. At layer 27,
`<|im_end|>` (token 151645) was top-1 for 54/54. This supports the positional
explanation directly; it is not a full-population top-token estimate. Raw
evidence is in `validity_diagnostic.json`.

## Population curves

- Layer 0: correct mean/median margin -14.787/-14.608; wrong mean/median GT-minus-answer margin -0.218/0.000.
- Layer 4: correct mean/median margin -17.041/-17.098; wrong mean/median GT-minus-answer margin 0.049/0.000.
- Layer 9: correct mean/median margin -15.109/-15.271; wrong mean/median GT-minus-answer margin 0.210/0.000.
- Layer 13: correct mean/median margin -17.301/-17.418; wrong mean/median GT-minus-answer margin 0.094/0.000.
- Layer 18: correct mean/median margin -11.769/-11.883; wrong mean/median GT-minus-answer margin 0.100/0.000.
- Layer 23: correct mean/median margin -9.305/-9.510; wrong mean/median GT-minus-answer margin -0.019/0.000.
- Layer 27: correct mean/median margin -29.683/-30.156; wrong mean/median GT-minus-answer margin -0.152/0.000.

## Emergence coverage

- Correct delta emergence is defined for 0/3,999 (0.0%).
- Wrong delta emergence is defined for 2,215/4,000 (55.4%).
- Correct 50%/75% delta-coverage layers: not reached / not reached.
- Wrong 50%/75% delta-coverage layers: 21 / not reached.
- Correct 50%/75% zero-crossing layers: not reached / not reached.
- Wrong 50%/75% zero-crossing layers: 4 / 24.

## Wrong-sample taxonomy

- `ambiguous`: 1,785 (44.6%)
- `answer_erosion`: 587 (14.7%)
- `early_wrong`: 1,203 (30.1%)
- `progressive_wrong`: 425 (10.6%)

Per dataset:

- GQA: ambiguous=616 (30.8%), answer_erosion=328 (16.4%), early_wrong=818 (40.9%), progressive_wrong=238 (11.9%)
- CHARTQA: ambiguous=628 (62.8%), answer_erosion=102 (10.2%), early_wrong=179 (17.9%), progressive_wrong=91 (9.1%)
- TEXTVQA: ambiguous=541 (54.1%), answer_erosion=157 (15.7%), early_wrong=206 (20.6%), progressive_wrong=96 (9.6%)

## First-token reference audit

- GT and generated first tokens coincide for 4,404/7,999 samples: 3,810 correct and 594 wrong.
- TextVQA uses the frozen canonical `gt_answer` first token for this lens; its other LMMS references determine correctness but are not outcome-selected for the logit target.
- Among wrong samples, 594/4,000 (14.9%) have identical canonical-GT and
  generated first tokens and are deterministically assigned `ambiguous`.
  Among the 3,406 non-collision wrong samples, the taxonomy is early-wrong
  1,203 (35.3%), progressive-wrong 425 (12.5%), answer erosion 587 (17.2%),
  and ambiguous 1,191 (35.0%).

## Answers to the plan questions

1. **No conclusion that early hidden states are uninformative is supported.**
   Early fixed-head answer margins are weak, but the readout is evaluated at a
   structural next-token position and is not a learned or calibrated probe of
   hidden-state failure information.
2. **No shared reliable separation depth was found.** Correct GT-vs-vocabulary
   emergence never occurs. For wrong samples, 50% reach persistent margin
   `< -1` by layer 21, while 75% never do; zero-crossing reaches 50% at layer 4
   and 75% at layer 24. These wrong-only crossings do not repair the invalid
   correct comparator.
3. **Wrong trajectories are not mostly progressive.** Under the frozen rule,
   the full wrong set is 30.1% early-wrong, 10.6% progressive-wrong, 14.7%
   answer erosion, and 44.6% ambiguous. The result remains a query-position
   relative-answer diagnostic and should be interpreted cautiously.
4. **No defensible supervision start layer follows from this run.** There is no
   layer at which both correct and wrong populations have substantial valid
   persistent answer-preference coverage.
5. **Do not select post-emergence or informative-range random-k supervision
   from these curves.** All-layer supervision remains an unselected baseline,
   not a recommendation. A valid comparison first requires answer-start hidden
   states (or another prospectively justified answer-position readout), which
   are not present in the saved Phase-45 feature definition.

## Interpretation boundary

The frozen text_final state is the final literal user-prompt token, not the actual assistant-start token. Results are a query-position logit lens and are not calibrated next-token probabilities.
Raw logits are not probabilities. Correct-sample semantic equivalence and TextVQA multi-reference scoring can also differ from the canonical GT first token used by this diagnostic. The optional teacher-forced sequence analysis was not triggered because the primary first-token pattern failed its positional interpretation.

## Stop status

The planned first-token analysis is complete. Teacher-forced sequence analysis, Stage-1 training, W→C repair, routing, and external evaluation were not run.
