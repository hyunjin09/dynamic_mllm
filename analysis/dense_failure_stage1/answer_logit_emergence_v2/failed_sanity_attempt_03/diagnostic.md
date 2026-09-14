# Failed sanity attempt 03

- Contract: `2a53c337a3116908dc019350c5168233b9e0271c4d33d6c3cd12947cf18abf48`
- Base answer-start validation: 72/72 exact.
- Raw cached layer-27 readout versus raw model logits: 13/13 exact top-1.
- Stored processed continuation replay: 12/13 exact overall and at least 4
  exact cases in every dataset.
- The one unreplayable comparison is `textvqa:textvqa_train_6823`, generated
  originally on rank 3 and replayed on rank 0. Its 16-token common prefix
  replays, but the next processed token changes.
- Contract correction: follow the requested per-sample collision skip rule.
  Require four exact replayable cases per dataset for the sanity gate; log and
  exclude any unreplayable collision from the full wrong-token comparison
  rather than failing the answer-position analysis or assigning ambiguity.

The full population run was not launched under this failed gate.
