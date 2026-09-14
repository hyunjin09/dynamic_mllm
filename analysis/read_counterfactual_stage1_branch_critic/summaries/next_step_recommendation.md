# Next-step recommendation — BC-C

Paired branch-risk calibration/ranking experiment, as a separately authorized phase with held-out image-group evaluation.

Reason: The closest supported category is BC-C, qualified: both branches retain moderate above-chance absolute failure signal, but relative branch choice is unsupported. Balanced flip accuracy is 47.77%, discordant AUROC is 0.4684, and delta-p Spearman is -0.0187. Q3 rescue enrichment is uncertain. The offline policy has Net +24 (30 rescues, 6 regressions), but this does not establish useful action ranking or controller readiness. The apparent ON-to-OFF AUROC drop is not evidence of an OFF-score collapse: on fixed ON labels, ON/OFF scores give 0.6728/0.6710 AUROC; on fixed OFF labels they give 0.5948/0.5949. Changing the labels accounts for the observed gap even with the score held fixed. This supports a relative-choice failure description, not a claim that simple recalibration will solve it. The cause of poor paired preference remains unknown.

This recommendation is unexecuted. The authorized offline diagnostic stops here.
