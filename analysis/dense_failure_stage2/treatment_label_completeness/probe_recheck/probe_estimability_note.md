# Audited-probe estimability note

The frozen secondary probe requires five UID/image-group-disjoint binary folds after excluding `AUDITED_MIXED`.

The audited clean population contains 4 `AUDITED_KEEP` states from four distinct UID/image groups and 229 `AUDITED_INTERVENE` states. Five test folds cannot all contain both classes. The deterministic fold assignment contains one fold with 0 KEEP and 47 INTERVENE states, so fold-level AUROC and AUPRC are undefined.

The probe was therefore stopped fail-closed. No switch to four folds, resampling, class duplication, threshold change, or post-hoc metric substitution was made. The old-label probe was retained as descriptive context; all audited probe metrics are explicitly marked `not_estimable`.
