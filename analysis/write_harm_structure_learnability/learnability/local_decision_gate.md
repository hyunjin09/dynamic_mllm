# W-LOCAL-WEAK

| population   | model     | baseline   |   spearman_gain |   spearman_ci_low |   spearman_ci_high |   harmful_auc_gain |   harmful_auc_ci_low |   harmful_auc_ci_high |   draws | bootstrap_unit   |
|:-------------|:----------|:-----------|----------------:|------------------:|-------------------:|-------------------:|---------------------:|----------------------:|--------:|:-----------------|
| full         | F_ALL/mlp | B1_GENERIC |      0.0116268  |        -0.0127075 |          0.0374092 |        -0.0165149  |           -0.0303653 |           -0.0017065  |    5000 | image_group_id   |
| full         | F_ALL/mlp | B2_DELTA   |      0.00949688 |        -0.0136829 |          0.0332319 |        -0.00784214 |           -0.0209743 |            0.00526666 |    5000 | image_group_id   |
| dense_w      | F_ALL/mlp | PRE        |      0.0040109  |        -0.023248  |          0.0322878 |        -0.0166551  |           -0.031689  |           -0.00175944 |    5000 | image_group_id   |
| dense_w      | F_ALL/mlp | DELTA      |      0.00714124 |        -0.0166797 |          0.0304224 |        -0.0116104  |           -0.0249931 |            0.00149935 |    5000 | image_group_id   |

[
  {
    "population": "full",
    "material_over_both": false,
    "useful_high_precision": false,
    "passes": false,
    "metrics": {
      "spearman": 0.04752441808700094,
      "pearson": 0.025571368936734853,
      "mae": 0.11311526456891746,
      "rmse": 0.2017106353712827,
      "harmful_auroc": 0.4818798029229957,
      "harmful_auprc": 0.42115664161938526,
      "harmful_prevalence": 0.4281198551201844,
      "states": 15185,
      "precision_at_0.05": 0.44342105263157894,
      "precision_at_0.1": 0.445687952600395,
      "precision_at_0.2": 0.4181758314125782,
      "recall_at_precision_0.9": 0.00046146746654360867,
      "recall_at_precision_0.95": 0.00046146746654360867
    }
  },
  {
    "population": "dense_w",
    "material_over_both": false,
    "useful_high_precision": false,
    "passes": false,
    "metrics": {
      "spearman": 0.042478549296348166,
      "pearson": 0.02651744171908437,
      "mae": 0.11795956853069939,
      "rmse": 0.2064184871048821,
      "harmful_auroc": 0.4851156545255654,
      "harmful_auprc": 0.4333884313832355,
      "harmful_prevalence": 0.43231655889794773,
      "states": 14228,
      "precision_at_0.05": 0.4957865168539326,
      "precision_at_0.1": 0.4799718903724526,
      "precision_at_0.2": 0.4483485593815882,
      "recall_at_precision_0.9": 0.00016257519102584944,
      "recall_at_precision_0.95": 0.00016257519102584944
    }
  }
]

Proceed only to the user-authorized W3 propagation stage. No search or controller.
