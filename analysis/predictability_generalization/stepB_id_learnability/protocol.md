# Predictability Step-B frozen protocol

- Run ID: `predictability_stepB_id_learnability_v1`
- Parent Step-A contract: `6103b9b91826455e9ebed97c2ee19c5e115a3ca0835006fdf2e5bf049934613d`
- Evaluation: five image-group-disjoint outer folds with one shared base-group registry.
- Inner calibration: deterministic 10% group-disjoint subset of each outer-training population.
- Weighting: every UID contributes equal total training/calibration mass.
- Stage-1 target: current dense all-FULL eventual wrongness.
- Stage-2 primary READ target: `q_F - q_WO` (`u_read_w1`).
- Stage-2 primary WRITE target: `q_F - q_RO` (`u_write_r1`).
- M0: nuisance-only control. M1: frozen summary linear. M2: fixed two-layer summary MLP.
- M3 Stage 1: current `SharedFailurePredictor` architecture, including its structural layer embedding.
- M3 Stage 2: current READ/WRITE cross-modal topology trained from scratch per target/representation/fold/seed.
- M1/M2/M3 receive no dataset/source/trigger-depth/final-outcome/utility/future/search inputs except the structural Stage-1 M3 layer embedding noted above.
- M2/M3 seeds: `[2026090801, 2026090802, 2026090803]`.
- Regression target scaling: outer-fit median and `1.4826*MAD`, floor `0.0001`; no clipping.
- Primary dense and secondary selected routed domains remain separate.
- This phase is in-domain only. It performs no nearest-question, clustering, LODO, external, or deployment evaluation.
