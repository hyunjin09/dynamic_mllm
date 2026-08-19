# Workflow Report Index

Project: `/data/projects/hyunjin/MLLM/dynamic_mllm`

## GPU Jobs

| id | status | exp_id | gpu_type | slurm_job_id | log |
|---|---|---|---|---|---|
| stage_a_env_20260804 | cancelled | stage_a_env_20260804 | a4000 | 97987 | runs/stage_a_env_20260804/slurm.log |
| stage_a_env_localcache_20260804 | succeeded | stage_a_env_localcache_20260804 | a4000 | 97995 | runs/stage_a_env_localcache_20260804/slurm.log |
| stage_a_probe_20260804 | failed | stage_a_probe_20260804 | a6000 | 98008 | runs/stage_a_probe_20260804/slurm.log |
| stage_a_probe_cublas_20260804 | succeeded | stage_a_probe_cublas_20260804 | a6000 | 98010 | runs/stage_a_probe_cublas_20260804/slurm.log |
| stage_a_probe_readfix_20260804 | succeeded | stage_a_probe_readfix_20260804 | a6000 | 98011 | runs/stage_a_probe_readfix_20260804/slurm.log |
| stage_a_validity_20260804 | failed | stage_a_validity_20260804 | a6000 | 98012 | runs/stage_a_validity_20260804/slurm.log |
| stage_a_validity_vision_sdpa_20260804 | failed | stage_a_validity_vision_sdpa_20260804 | a6000 | 98014 | runs/stage_a_validity_vision_sdpa_20260804/slurm.log |
| stage_a_validity_final_20260804 | failed | stage_a_validity_final_20260804 | a6000 | 98015 | runs/stage_a_validity_final_20260804/slurm.log |
| stage_a_validity_memory_bounded_20260804 | failed | stage_a_validity_memory_bounded_20260804 | a6000 | 98016 | runs/stage_a_validity_memory_bounded_20260804/slurm.log |
| stage_a_sdpa_reference_probe_20260804 | failed | stage_a_sdpa_reference_probe_20260804 | a6000 | 98020 | runs/stage_a_sdpa_reference_probe_20260804/slurm.log |
| stage_a_sdpa_reference_probe_chunked_20260804 | succeeded | stage_a_sdpa_reference_probe_chunked_20260804 | a6000 | 98022 | runs/stage_a_sdpa_reference_probe_chunked_20260804/slurm.log |
| stage_a_sdpa_reference_probe_causal_20260804 | succeeded | stage_a_sdpa_reference_probe_causal_20260804 | a6000 | 98023 | runs/stage_a_sdpa_reference_probe_causal_20260804/slurm.log |
| stage_a_chunked_eager_probe_20260804 | cancelled | stage_a_chunked_eager_probe_20260804 | a6000 | 98027 | runs/stage_a_chunked_eager_probe_20260804/slurm.log |
| stage_a_chunked_eager_probe_256_20260804 | cancelled | stage_a_chunked_eager_probe_256_20260804 | a6000 | 98028 | runs/stage_a_chunked_eager_probe_256_20260804/slurm.log |
| stage_a_chunked_eager_probe_1024_20260804 | succeeded | stage_a_chunked_eager_probe_1024_20260804 | a6000 | 98035 | runs/stage_a_chunked_eager_probe_1024_20260804/slurm.log |
| stage_a_chunked_stock_equivalence_20260804 | succeeded | stage_a_chunked_stock_equivalence_20260804 | a6000 | 98040 | runs/stage_a_chunked_stock_equivalence_20260804/slurm.log |
| stage_a_chunked_stock_equivalence_boundary_20260804 | failed | stage_a_chunked_stock_equivalence_boundary_20260804 | a6000 | 98041 | runs/stage_a_chunked_stock_equivalence_boundary_20260804/slurm.log |
| stage_a_stock_eager_validity_23_20260804 | succeeded | stage_a_stock_eager_validity_23_20260804 | a6000 | 98044 | runs/stage_a_stock_eager_validity_23_20260804/slurm.log |
| stage_b_reference_validity_20260804 | succeeded | stage_b_reference_validity_20260804 | a6000 | 98081 | runs/stage_b_reference_validity_20260804/slurm.log |
| stage_b_reference_validity_v2_20260804 | succeeded | stage_b_reference_validity_v2_20260804 | a6000 | 98084 | runs/stage_b_reference_validity_v2_20260804/slurm.log |
| stage_b_reference_full_20260804 | cancelled | stage_b_reference_full_20260804 | a6000 | 98085 | runs/stage_b_reference_full_20260804/slurm.log |
| stage_b_reference_validity_v3_20260804 | failed | stage_b_reference_validity_v3_20260804 | a6000 | 98087 | runs/stage_b_reference_validity_v3_20260804/slurm.log |
| stage_b_reference_validity_v4_20260804 | succeeded | stage_b_reference_validity_v4_20260804 | a6000 | 98088 | runs/stage_b_reference_validity_v4_20260804/slurm.log |
| stage_b_reference_full_v2_20260804 | succeeded | stage_b_reference_full_v2_20260804 | a6000 | 98089 | runs/stage_b_reference_full_v2_20260804/slurm.log |
| stage_b_reference_analysis_20260804 | succeeded | stage_b_reference_analysis_20260804 | a4000 | 98111 | runs/stage_b_reference_analysis_20260804/slurm.log |
| stage_c_datasets_env_20260805 | succeeded | stage_c_datasets_env_20260805 | a4000 | 98372 | runs/stage_c_datasets_env_20260805/slurm.log |
| stage_c_textvqa_validation_download_20260805 | succeeded | stage_c_textvqa_validation_download_20260805 | a4000 | 98373 | runs/stage_c_textvqa_validation_download_20260805/slurm.log |
| stage_c_manifest_freeze_20260805 | failed | stage_c_manifest_freeze_20260805 | a6000 | 98374 | runs/stage_c_manifest_freeze_20260805/slurm.log |
| stage_c_manifest_freeze_v2_20260805 | cancelled | stage_c_manifest_freeze_v2_20260805 | a6000 | 98375 | runs/stage_c_manifest_freeze_v2_20260805/slurm.log |
| stage_c_manifest_freeze_v3_20260805 | succeeded | stage_c_manifest_freeze_v3_20260805 | a6000 | 98376 | runs/stage_c_manifest_freeze_v3_20260805/slurm.log |
| stage_c_entry_gate_20260805 | failed | stage_c_entry_gate_20260805 | a6000 | 98377 | runs/stage_c_entry_gate_20260805/slurm.log |
| stage_c_entry_gate_v2_20260805 | succeeded | stage_c_entry_gate_v2_20260805 | a6000 | 98378 | runs/stage_c_entry_gate_v2_20260805/slurm.log |
| stage_c_entry_gate_v3_20260805 | succeeded | stage_c_entry_gate_v3_20260805 | a6000 | 98379 | runs/stage_c_entry_gate_v3_20260805/slurm.log |
| stage_c_full_preflight_20260805 | failed | stage_c_full_preflight_20260805 | a4000 | 98384 | runs/stage_c_full_preflight_20260805/slurm-%j.out |
| stage_c_prefix_static_preflight_v2_20260805 | succeeded | stage_c_prefix_static_preflight_v2_20260805 | a4000 | 98385 | runs/stage_c_prefix_static_preflight_v2_20260805/slurm-%j.out |
| stage_c_prefix_score_preflight_s0_20260805 | succeeded | stage_c_prefix_score_preflight_v1_20260805 | a6000 | 98386 | runs/stage_c_prefix_score_preflight_v1_20260805/shard_0/slurm-%j.out |
| stage_c_prefix_score_preflight_s1_20260805 | succeeded | stage_c_prefix_score_preflight_v1_20260805 | a6000 | 98387 | runs/stage_c_prefix_score_preflight_v1_20260805/shard_1/slurm-%j.out |
| stage_c_prefix_score_preflight_s2_20260805 | succeeded | stage_c_prefix_score_preflight_v1_20260805 | a6000 | 98388 | runs/stage_c_prefix_score_preflight_v1_20260805/shard_2/slurm-%j.out |
| stage_c_prefix_score_preflight_s3_20260805 | succeeded | stage_c_prefix_score_preflight_v1_20260805 | a6000 | 98389 | runs/stage_c_prefix_score_preflight_v1_20260805/shard_3/slurm-%j.out |
| stage_c_prefix_score_preflight_s4_20260805 | succeeded | stage_c_prefix_score_preflight_v1_20260805 | a6000 | 98390 | runs/stage_c_prefix_score_preflight_v1_20260805/shard_4/slurm-%j.out |
| stage_c_prefix_score_preflight_s5_20260805 | succeeded | stage_c_prefix_score_preflight_v1_20260805 | a6000 | 98391 | runs/stage_c_prefix_score_preflight_v1_20260805/shard_5/slurm-%j.out |
| stage_c_prefix_score_preflight_s6_20260805 | succeeded | stage_c_prefix_score_preflight_v1_20260805 | a6000 | 98392 | runs/stage_c_prefix_score_preflight_v1_20260805/shard_6/slurm-%j.out |
| stage_c_prefix_score_preflight_s7_20260805 | succeeded | stage_c_prefix_score_preflight_v1_20260805 | a6000 | 98393 | runs/stage_c_prefix_score_preflight_v1_20260805/shard_7/slurm-%j.out |
| stage_c_prefix_preflight_merge_20260805 | succeeded | stage_c_prefix_preflight_v1_20260805 | a4000 | 98397 | runs/stage_c_prefix_preflight_merge_20260805/slurm-%j.out |
| stage_c_full_s0_20260805 | failed | stage_c_full_v1_20260805 | a6000 | 98398 | runs/stage_c_full_v1_20260805/shard_0/slurm-%j.out |
| stage_c_full_s1_20260805 | cancelled | stage_c_full_v1_20260805 | a6000 | 98399 | runs/stage_c_full_v1_20260805/shard_1/slurm-%j.out |
| stage_c_full_s2_20260805 | cancelled | stage_c_full_v1_20260805 | a6000 | 98400 | runs/stage_c_full_v1_20260805/shard_2/slurm-%j.out |
| stage_c_full_s3_20260805 | cancelled | stage_c_full_v1_20260805 | a6000 | 98401 | runs/stage_c_full_v1_20260805/shard_3/slurm-%j.out |
| stage_c_full_s4_20260805 | cancelled | stage_c_full_v1_20260805 | a6000 | 98402 | runs/stage_c_full_v1_20260805/shard_4/slurm-%j.out |
| stage_c_full_s5_20260805 | cancelled | stage_c_full_v1_20260805 | a6000 | 98403 | runs/stage_c_full_v1_20260805/shard_5/slurm-%j.out |
| stage_c_full_s6_20260805 | cancelled | stage_c_full_v1_20260805 | a6000 | 98404 | runs/stage_c_full_v1_20260805/shard_6/slurm-%j.out |
| stage_c_full_s7_20260805 | cancelled | stage_c_full_v1_20260805 | a6000 | 98405 | runs/stage_c_full_v1_20260805/shard_7/slurm-%j.out |
| stage_c_donor_audit_s00_20260805 | succeeded | stage_c_donor_coverage_audit_v1_20260805 | a6000 | 98421 | runs/stage_c_donor_coverage_audit_v1_20260805/shard_00/slurm.log |
| stage_c_donor_audit_s01_20260805 | succeeded | stage_c_donor_coverage_audit_v1_20260805 | a6000 | 98422 | runs/stage_c_donor_coverage_audit_v1_20260805/shard_01/slurm.log |
| stage_c_donor_audit_s02_20260805 | succeeded | stage_c_donor_coverage_audit_v1_20260805 | a6000 | 98423 | runs/stage_c_donor_coverage_audit_v1_20260805/shard_02/slurm.log |
| stage_c_donor_audit_s03_20260805 | succeeded | stage_c_donor_coverage_audit_v1_20260805 | a6000 | 98424 | runs/stage_c_donor_coverage_audit_v1_20260805/shard_03/slurm.log |
| stage_c_donor_audit_s04_20260805 | succeeded | stage_c_donor_coverage_audit_v1_20260805 | a6000 | 98425 | runs/stage_c_donor_coverage_audit_v1_20260805/shard_04/slurm.log |
| stage_c_donor_audit_s05_20260805 | succeeded | stage_c_donor_coverage_audit_v1_20260805 | a6000 | 98426 | runs/stage_c_donor_coverage_audit_v1_20260805/shard_05/slurm.log |
| stage_c_donor_audit_s06_20260805 | succeeded | stage_c_donor_coverage_audit_v1_20260805 | a6000 | 98427 | runs/stage_c_donor_coverage_audit_v1_20260805/shard_06/slurm.log |
| stage_c_donor_audit_s07_20260805 | succeeded | stage_c_donor_coverage_audit_v1_20260805 | a6000 | 98428 | runs/stage_c_donor_coverage_audit_v1_20260805/shard_07/slurm.log |
| stage_c_full_v2_s00_20260805 | succeeded | stage_c_full_v2_20260805 | a6000 | 98453 | runs/stage_c_full_v2_20260805/shard_00/slurm.log |
| stage_c_full_v2_s01_20260805 | succeeded | stage_c_full_v2_20260805 | a6000 | 98454 | runs/stage_c_full_v2_20260805/shard_01/slurm.log |
| stage_c_full_v2_s02_20260805 | succeeded | stage_c_full_v2_20260805 | a6000 | 98455 | runs/stage_c_full_v2_20260805/shard_02/slurm.log |
| stage_c_full_v2_s03_20260805 | succeeded | stage_c_full_v2_20260805 | a6000 | 98456 | runs/stage_c_full_v2_20260805/shard_03/slurm.log |
| stage_c_full_v2_s04_20260805 | succeeded | stage_c_full_v2_20260805 | a6000 | 98457 | runs/stage_c_full_v2_20260805/shard_04/slurm.log |
| stage_c_full_v2_s05_20260805 | succeeded | stage_c_full_v2_20260805 | a6000 | 98458 | runs/stage_c_full_v2_20260805/shard_05/slurm.log |
| stage_c_full_v2_s06_20260805 | succeeded | stage_c_full_v2_20260805 | a6000 | 98459 | runs/stage_c_full_v2_20260805/shard_06/slurm.log |
| stage_c_full_v2_s07_20260805 | succeeded | stage_c_full_v2_20260805 | a6000 | 98460 | runs/stage_c_full_v2_20260805/shard_07/slurm.log |
| stage_c_merge_v2_20260805 | cancelled | stage_c_full_v2_20260805 | auto |  | runs/stage_c_full_v2_20260805/merge/slurm.log |
| stage_c_merge_v2_cpu_20260805 | succeeded | stage_c_full_v2_20260805 | a4000 | 98470 | runs/stage_c_full_v2_20260805/merge_cpu/slurm.log |
| stage_c_analysis_v1_20260805 | failed | stage_c_full_v2_20260805 | a4000 | 98471 | runs/stage_c_full_v2_20260805/analysis/slurm.log |
| stage_c_analysis_v1_retry_20260805 | succeeded | stage_c_full_v2_20260805 | a4000 | 98472 | runs/stage_c_full_v2_20260805/analysis_retry/slurm.log |
| stage_b_c_archive_outcome_b_20260805 | failed | stage_b_c_archive_outcome_b_20260805 | a4000 | 98521 | runs/stage_b_c_archive_outcome_b_20260805/slurm.log |
| stage_b_c_archive_outcome_b_retry_20260805 | succeeded | stage_b_c_archive_outcome_b_20260805 | a4000 | 98522 | runs/stage_b_c_archive_outcome_b_retry_20260805/slurm.log |
| v3_stage_b_reanalysis_20260806 | running | v3_stage_b_reanalysis_v1 | a4000 | 98746 | runs/v3_stage_b_reanalysis_20260806/slurm.log |
| v3_stage_b_reanalysis_20260806_r2 | running | v3_stage_b_reanalysis_v1 | a4000 | 98747 | runs/v3_stage_b_reanalysis_20260806_r2/slurm.log |
| v3_stage_b_reanalysis_20260806_r3 | running | v3_stage_b_reanalysis_v1 | a4000 | 98748 | runs/v3_stage_b_reanalysis_20260806_r3/slurm.log |
| v3_stage_b_reanalysis_20260806_r4 | running | v3_stage_b_reanalysis_v1 | a4000 | 98749 | runs/v3_stage_b_reanalysis_20260806_r4/slurm.log |
| v3_preflight_pool_audit_20260806 | running | v3_preflight_pool_audit_v1 | a4000 | 98760 | runs/v3_preflight_pool_audit_20260806/slurm.log |
| v3_preflight_pool_audit_20260806_r2 | running | v3_preflight_pool_audit_v1 | a4000 | 98761 | runs/v3_preflight_pool_audit_20260806_r2/slurm.log |
| v3_confirmation_preflight_smoke_20260806 | running | v3_confirmation_preflight_v1 | a6000 | 98765 | runs/v3_confirmation_preflight_smoke_20260806/slurm.log |
| v3_query_invariance_diag_20260806 | running | v3_confirmation_preflight_v1 | a6000 | 98767 | runs/v3_query_invariance_diag_20260806/slurm.log |
| v3_gqa_scenegraphs_download_20260806 | succeeded | v3_gqa_scenegraphs_download_20260806 | a4000 | 98771 | runs/v3_gqa_scenegraphs_download_20260806/slurm.log |
| v3_null_geometry_smoke_20260806 | failed | v3_null_geometry_smoke_20260806 | a6000 | 98774 | runs/v3_null_geometry_smoke_20260806/slurm.log |
| v3_null_geometry_smoke_r2_20260806 | failed | v3_null_geometry_smoke_r2_20260806 | a6000 | 98775 | runs/v3_null_geometry_smoke_r2_20260806/slurm.log |
| v3_textocr_annotations_download_20260806 | succeeded | v3_textocr_annotations_download_20260806 | a4000 | 98776 | runs/v3_textocr_annotations_download_20260806/slurm.log |
| v3_null_geometry_smoke_r3_20260806 | failed | v3_null_geometry_smoke_r3_20260806 | a6000 | 98777 | runs/v3_null_geometry_smoke_r3_20260806/slurm.log |
| v3_null_geometry_smoke_r4_20260806 | succeeded | v3_null_geometry_smoke_r4_20260806 | a6000 | 98780 | runs/v3_null_geometry_smoke_r4_20260806/slurm.log |
| v3_grounding_audit_20260806 | failed | v3_grounding_audit_20260806 | a4000 | 98781 | runs/v3_grounding_audit_20260806/slurm.log |
| v3_grounding_audit_r2_20260806 | succeeded | v3_grounding_audit_r2_20260806 | a4000 | 98783 | runs/v3_grounding_audit_r2_20260806/slurm.log |
| v3_null_geometry_s00_20260806 | succeeded | v3_null_geometry_s00_20260806 | a6000 | 98785 | runs/v3_null_geometry_s00_20260806/slurm.log |
| v3_null_geometry_s01_20260806 | succeeded | v3_null_geometry_s01_20260806 | a6000 | 98786 | runs/v3_null_geometry_s01_20260806/slurm.log |
| v3_null_geometry_s02_20260806 | succeeded | v3_null_geometry_s02_20260806 | a5000 | 98787 | runs/v3_null_geometry_s02_20260806/slurm.log |
| v3_null_geometry_s03_20260806 | failed | v3_null_geometry_s03_20260806 | a4000 | 98788 | runs/v3_null_geometry_s03_20260806/slurm.log |
| v3_null_geometry_s04_20260806 | failed | v3_null_geometry_s04_20260806 | a4000 | 98789 | runs/v3_null_geometry_s04_20260806/slurm.log |
| v3_null_geometry_s05_20260806 | failed | v3_null_geometry_s05_20260806 | a4000 | 98790 | runs/v3_null_geometry_s05_20260806/slurm.log |
| v3_null_geometry_s06_20260806 | failed | v3_null_geometry_s06_20260806 | a4000 | 98791 | runs/v3_null_geometry_s06_20260806/slurm.log |
| v3_null_geometry_s07_20260806 | failed | v3_null_geometry_s07_20260806 | a4000 | 98792 | runs/v3_null_geometry_s07_20260806/slurm.log |
| v3_null_geometry_s03_r2_20260806 | cancelled | v3_null_geometry_s03_r2_20260806 | a6000 |  | runs/v3_null_geometry_s03_r2_20260806/slurm.log |
| v3_null_geometry_s04_r2_20260806 | cancelled | v3_null_geometry_s04_r2_20260806 | a6000 |  | runs/v3_null_geometry_s04_r2_20260806/slurm.log |
| v3_null_geometry_s05_r2_20260806 | cancelled | v3_null_geometry_s05_r2_20260806 | a6000 |  | runs/v3_null_geometry_s05_r2_20260806/slurm.log |
| v3_null_geometry_s06_r2_20260806 | cancelled | v3_null_geometry_s06_r2_20260806 | a6000 |  | runs/v3_null_geometry_s06_r2_20260806/slurm.log |
| v3_null_geometry_s07_r2_20260806 | cancelled | v3_null_geometry_s07_r2_20260806 | a6000 |  | runs/v3_null_geometry_s07_r2_20260806/slurm.log |
| v3_null_geometry_s03_r3_20260806 | succeeded | v3_null_geometry_s03_r3_20260806 | a6000 | 98793 | runs/v3_null_geometry_s03_r3_20260806/slurm.log |
| v3_null_geometry_s04_r3_20260806 | succeeded | v3_null_geometry_s04_r3_20260806 | a6000 | 98794 | runs/v3_null_geometry_s04_r3_20260806/slurm.log |
| v3_null_geometry_s05_r3_20260806 | succeeded | v3_null_geometry_s05_r3_20260806 | a6000 | 98795 | runs/v3_null_geometry_s05_r3_20260806/slurm.log |
| v3_null_geometry_s06_r3_20260806 | succeeded | v3_null_geometry_s06_r3_20260806 | a6000 | 98796 | runs/v3_null_geometry_s06_r3_20260806/slurm.log |
| v3_null_geometry_s07_r3_20260806 | succeeded | v3_null_geometry_s07_r3_20260806 | a6000 | 98797 | runs/v3_null_geometry_s07_r3_20260806/slurm.log |
| v3_null_geometry_merge_20260806 | succeeded | v3_null_geometry_merge_20260806 | a4000 | 98798 | runs/v3_null_geometry_merge_20260806/slurm.log |
| v3_null_model_fit_20260806 | failed | v3_null_model_fit_20260806 | a6000 | 98799 | runs/v3_null_model_fit_20260806/slurm.log |
| v3_null_geometry_merge_r2_20260806 | succeeded | v3_null_geometry_merge_r2_20260806 | a4000 | 98800 | runs/v3_null_geometry_merge_r2_20260806/slurm.log |
| v3_null_model_fit_r2_20260806 | failed | v3_null_model_fit_r2_20260806 | a6000 | 98801 | runs/v3_null_model_fit_r2_20260806/slurm.log |
| v3_null_redesign_pool_20260807 | succeeded | v3_null_redesign_pool_v1 | a4000 | 98898 | runs/v3_null_redesign_pool_20260807/slurm.log |
| v3_null_redesign_geom_s00_20260807 | failed | v3_null_redesign_geometry_s00_v1 | a6000 | 98904 | runs/v3_null_redesign_geometry_20260807/shard_00.log |
| v3_null_redesign_geom_s01_20260807 | failed | v3_null_redesign_geometry_s01_v1 | a6000 | 98905 | runs/v3_null_redesign_geometry_20260807/shard_01.log |
| v3_null_redesign_geom_s02_20260807 | failed | v3_null_redesign_geometry_s02_v1 | a6000 | 98906 | runs/v3_null_redesign_geometry_20260807/shard_02.log |
| v3_null_redesign_geom_s03_20260807 | failed | v3_null_redesign_geometry_s03_v1 | a6000 | 98907 | runs/v3_null_redesign_geometry_20260807/shard_03.log |
| v3_null_redesign_geom_s04_20260807 | failed | v3_null_redesign_geometry_s04_v1 | a6000 | 98908 | runs/v3_null_redesign_geometry_20260807/shard_04.log |
| v3_null_redesign_geom_s05_20260807 | failed | v3_null_redesign_geometry_s05_v1 | a6000 | 98909 | runs/v3_null_redesign_geometry_20260807/shard_05.log |
| v3_null_redesign_geom_s06_20260807 | failed | v3_null_redesign_geometry_s06_v1 | a6000 | 98910 | runs/v3_null_redesign_geometry_20260807/shard_06.log |
| v3_null_redesign_geom_s07_20260807 | failed | v3_null_redesign_geometry_s07_v1 | a6000 | 98911 | runs/v3_null_redesign_geometry_20260807/shard_07.log |
| v3_null_redesign_geom_s08_20260807 | cancelled | v3_null_redesign_geometry_s08_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_08.log |
| v3_null_redesign_geom_s09_20260807 | cancelled | v3_null_redesign_geometry_s09_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_09.log |
| v3_null_redesign_geom_s10_20260807 | cancelled | v3_null_redesign_geometry_s10_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_10.log |
| v3_null_redesign_geom_s11_20260807 | cancelled | v3_null_redesign_geometry_s11_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_11.log |
| v3_null_redesign_geom_s12_20260807 | cancelled | v3_null_redesign_geometry_s12_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_12.log |
| v3_null_redesign_geom_s13_20260807 | cancelled | v3_null_redesign_geometry_s13_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_13.log |
| v3_null_redesign_geom_s14_20260807 | cancelled | v3_null_redesign_geometry_s14_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_14.log |
| v3_null_redesign_geom_s15_20260807 | cancelled | v3_null_redesign_geometry_s15_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_15.log |
| v3_null_redesign_geom_s16_20260807 | cancelled | v3_null_redesign_geometry_s16_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_16.log |
| v3_null_redesign_geom_s17_20260807 | cancelled | v3_null_redesign_geometry_s17_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_17.log |
| v3_null_redesign_geom_s18_20260807 | cancelled | v3_null_redesign_geometry_s18_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_18.log |
| v3_null_redesign_geom_s19_20260807 | cancelled | v3_null_redesign_geometry_s19_v1 | a6000 |  | runs/v3_null_redesign_geometry_20260807/shard_19.log |
| v3_null_redesign_geom_s00_r2_20260807 | failed | v3_null_redesign_geometry_s00_v1_r2 | a6000 | 98912 | runs/v3_null_redesign_geometry_r2_20260807/shard_00.log |
| v3_null_redesign_geom_s01_r2_20260807 | failed | v3_null_redesign_geometry_s01_v1_r2 | a6000 | 98913 | runs/v3_null_redesign_geometry_r2_20260807/shard_01.log |
| v3_null_redesign_geom_s02_r2_20260807 | failed | v3_null_redesign_geometry_s02_v1_r2 | a6000 | 98914 | runs/v3_null_redesign_geometry_r2_20260807/shard_02.log |
| v3_null_redesign_geom_s03_r2_20260807 | failed | v3_null_redesign_geometry_s03_v1_r2 | a6000 | 98915 | runs/v3_null_redesign_geometry_r2_20260807/shard_03.log |
| v3_null_redesign_geom_s04_r2_20260807 | failed | v3_null_redesign_geometry_s04_v1_r2 | a6000 | 98916 | runs/v3_null_redesign_geometry_r2_20260807/shard_04.log |
| v3_null_redesign_geom_s05_r2_20260807 | failed | v3_null_redesign_geometry_s05_v1_r2 | a6000 | 98917 | runs/v3_null_redesign_geometry_r2_20260807/shard_05.log |
| v3_null_redesign_geom_s06_r2_20260807 | failed | v3_null_redesign_geometry_s06_v1_r2 | a6000 | 98918 | runs/v3_null_redesign_geometry_r2_20260807/shard_06.log |
| v3_null_redesign_geom_s07_r2_20260807 | failed | v3_null_redesign_geometry_s07_v1_r2 | a6000 | 98919 | runs/v3_null_redesign_geometry_r2_20260807/shard_07.log |
| v3_null_redesign_geom_s08_r2_20260807 | cancelled | v3_null_redesign_geometry_s08_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_08.log |
| v3_null_redesign_geom_s09_r2_20260807 | cancelled | v3_null_redesign_geometry_s09_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_09.log |
| v3_null_redesign_geom_s10_r2_20260807 | cancelled | v3_null_redesign_geometry_s10_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_10.log |
| v3_null_redesign_geom_s11_r2_20260807 | cancelled | v3_null_redesign_geometry_s11_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_11.log |
| v3_null_redesign_geom_s12_r2_20260807 | cancelled | v3_null_redesign_geometry_s12_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_12.log |
| v3_null_redesign_geom_s13_r2_20260807 | cancelled | v3_null_redesign_geometry_s13_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_13.log |
| v3_null_redesign_geom_s14_r2_20260807 | cancelled | v3_null_redesign_geometry_s14_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_14.log |
| v3_null_redesign_geom_s15_r2_20260807 | cancelled | v3_null_redesign_geometry_s15_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_15.log |
| v3_null_redesign_geom_s16_r2_20260807 | cancelled | v3_null_redesign_geometry_s16_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_16.log |
| v3_null_redesign_geom_s17_r2_20260807 | cancelled | v3_null_redesign_geometry_s17_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_17.log |
| v3_null_redesign_geom_s18_r2_20260807 | cancelled | v3_null_redesign_geometry_s18_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_18.log |
| v3_null_redesign_geom_s19_r2_20260807 | cancelled | v3_null_redesign_geometry_s19_v1_r2 | a6000 |  | runs/v3_null_redesign_geometry_r2_20260807/shard_19.log |
| v3_null_redesign_geom_s00_r3_20260807 | succeeded | v3_null_redesign_geometry_s00_v1_r3 | a6000 | 98920 | runs/v3_null_redesign_geometry_r3_20260807/shard_00.log |
| v3_null_redesign_geom_s01_r3_20260807 | succeeded | v3_null_redesign_geometry_s01_v1_r3 | a6000 | 98921 | runs/v3_null_redesign_geometry_r3_20260807/shard_01.log |
| v3_null_redesign_geom_s02_r3_20260807 | succeeded | v3_null_redesign_geometry_s02_v1_r3 | a6000 | 98922 | runs/v3_null_redesign_geometry_r3_20260807/shard_02.log |
| v3_null_redesign_geom_s03_r3_20260807 | succeeded | v3_null_redesign_geometry_s03_v1_r3 | a6000 | 98923 | runs/v3_null_redesign_geometry_r3_20260807/shard_03.log |
| v3_null_redesign_geom_s04_r3_20260807 | succeeded | v3_null_redesign_geometry_s04_v1_r3 | a6000 | 98924 | runs/v3_null_redesign_geometry_r3_20260807/shard_04.log |
| v3_null_redesign_geom_s05_r3_20260807 | succeeded | v3_null_redesign_geometry_s05_v1_r3 | a6000 | 98925 | runs/v3_null_redesign_geometry_r3_20260807/shard_05.log |
| v3_null_redesign_geom_s06_r3_20260807 | succeeded | v3_null_redesign_geometry_s06_v1_r3 | a6000 | 98926 | runs/v3_null_redesign_geometry_r3_20260807/shard_06.log |
| v3_null_redesign_geom_s07_r3_20260807 | succeeded | v3_null_redesign_geometry_s07_v1_r3 | a6000 | 98927 | runs/v3_null_redesign_geometry_r3_20260807/shard_07.log |
| v3_null_redesign_geom_s08_r3_20260807 | succeeded | v3_null_redesign_geometry_s08_v1_r3 | a6000 | 98928 | runs/v3_null_redesign_geometry_r3_20260807/shard_08.log |
| v3_null_redesign_geom_s09_r3_20260807 | succeeded | v3_null_redesign_geometry_s09_v1_r3 | a6000 | 98929 | runs/v3_null_redesign_geometry_r3_20260807/shard_09.log |
| v3_null_redesign_geom_s10_r3_20260807 | succeeded | v3_null_redesign_geometry_s10_v1_r3 | a6000 | 98930 | runs/v3_null_redesign_geometry_r3_20260807/shard_10.log |
| v3_null_redesign_geom_s11_r3_20260807 | succeeded | v3_null_redesign_geometry_s11_v1_r3 | a6000 | 98931 | runs/v3_null_redesign_geometry_r3_20260807/shard_11.log |
| v3_null_redesign_geom_s12_r3_20260807 | succeeded | v3_null_redesign_geometry_s12_v1_r3 | a6000 | 98932 | runs/v3_null_redesign_geometry_r3_20260807/shard_12.log |
| v3_null_redesign_geom_s13_r3_20260807 | succeeded | v3_null_redesign_geometry_s13_v1_r3 | a6000 | 98933 | runs/v3_null_redesign_geometry_r3_20260807/shard_13.log |
| v3_null_redesign_geom_s14_r3_20260807 | succeeded | v3_null_redesign_geometry_s14_v1_r3 | a6000 | 98934 | runs/v3_null_redesign_geometry_r3_20260807/shard_14.log |
| v3_null_redesign_geom_s15_r3_20260807 | succeeded | v3_null_redesign_geometry_s15_v1_r3 | a6000 | 98935 | runs/v3_null_redesign_geometry_r3_20260807/shard_15.log |
| v3_null_redesign_geom_s16_r3_20260807 | succeeded | v3_null_redesign_geometry_s16_v1_r3 | a6000 | 98940 | runs/v3_null_redesign_geometry_r3_20260807/shard_16.log |
| v3_null_redesign_geom_s17_r3_20260807 | succeeded | v3_null_redesign_geometry_s17_v1_r3 | a6000 | 98941 | runs/v3_null_redesign_geometry_r3_20260807/shard_17.log |
| v3_null_redesign_geom_s18_r3_20260807 | succeeded | v3_null_redesign_geometry_s18_v1_r3 | a6000 | 98942 | runs/v3_null_redesign_geometry_r3_20260807/shard_18.log |
| v3_null_redesign_geom_s19_r3_20260807 | succeeded | v3_null_redesign_geometry_s19_v1_r3 | a6000 | 98943 | runs/v3_null_redesign_geometry_r3_20260807/shard_19.log |
| v3_null_redesign_merge_20260807 | failed | v3_null_redesign_geometry_merge_v1 | a4000 | 98956 | runs/v3_null_redesign_merge_20260807/slurm.log |
| v3_null_redesign_merge_r2_20260807 | succeeded | v3_null_redesign_geometry_merge_v1_r2 | a4000 | 98957 | runs/v3_null_redesign_merge_r2_20260807/slurm.log |
| v3_null_redesign_donors_20260807 | succeeded | v3_null_redesign_donor_audit_v1 | a4000 | 98959 | runs/v3_null_redesign_donors_20260807/slurm.log |
| v3_null_redesign_covariance_20260807 | succeeded | v3_null_redesign_covariance_comparison_v1 | a6000 | 98961 | runs/v3_null_redesign_covariance_20260807/slurm.log |
| v3_null_redesign_pool_v2_20260807 | succeeded | v3_null_redesign_pool_v2 | a4000 | 98963 | runs/v3_null_redesign_pool_v2_20260807/slurm.log |
| v3_null_redesign_c_rank_20260807 | succeeded | v3_null_redesign_c_rank1024_v1 | a6000 | 98964 | runs/v3_null_redesign_c_rank_20260807/slurm.log |
| v3_null_redesign_delta_s00_20260807 | succeeded | v3_null_redesign_delta_s00_v1 | a6000 | 98965 | runs/v3_null_redesign_geometry_delta_20260807/shard_00.log |
| v3_null_redesign_delta_s01_20260807 | succeeded | v3_null_redesign_delta_s01_v1 | a6000 | 98966 | runs/v3_null_redesign_geometry_delta_20260807/shard_01.log |
| v3_null_redesign_delta_s02_20260807 | succeeded | v3_null_redesign_delta_s02_v1 | a6000 | 98967 | runs/v3_null_redesign_geometry_delta_20260807/shard_02.log |
| v3_null_redesign_delta_s03_20260807 | succeeded | v3_null_redesign_delta_s03_v1 | a6000 | 98968 | runs/v3_null_redesign_geometry_delta_20260807/shard_03.log |
| v3_null_redesign_delta_s04_20260807 | succeeded | v3_null_redesign_delta_s04_v1 | a6000 | 98969 | runs/v3_null_redesign_geometry_delta_20260807/shard_04.log |
| v3_null_redesign_delta_s05_20260807 | succeeded | v3_null_redesign_delta_s05_v1 | a6000 | 98970 | runs/v3_null_redesign_geometry_delta_20260807/shard_05.log |
| v3_null_redesign_delta_s06_20260807 | succeeded | v3_null_redesign_delta_s06_v1 | a6000 | 98971 | runs/v3_null_redesign_geometry_delta_20260807/shard_06.log |
| v3_null_redesign_delta_s07_20260807 | succeeded | v3_null_redesign_delta_s07_v1 | a6000 | 98972 | runs/v3_null_redesign_geometry_delta_20260807/shard_07.log |
| v3_null_redesign_delta_s08_20260807 | succeeded | v3_null_redesign_delta_s08_v1 | a6000 | 98973 | runs/v3_null_redesign_geometry_delta_20260807/shard_08.log |
| v3_null_redesign_delta_s09_20260807 | succeeded | v3_null_redesign_delta_s09_v1 | a6000 | 98974 | runs/v3_null_redesign_geometry_delta_20260807/shard_09.log |
| v3_null_redesign_delta_s10_20260807 | succeeded | v3_null_redesign_delta_s10_v1 | a6000 | 98975 | runs/v3_null_redesign_geometry_delta_20260807/shard_10.log |
| v3_null_redesign_delta_s11_20260807 | succeeded | v3_null_redesign_delta_s11_v1 | a6000 | 98976 | runs/v3_null_redesign_geometry_delta_20260807/shard_11.log |
| v3_null_redesign_delta_s12_20260807 | succeeded | v3_null_redesign_delta_s12_v1 | a6000 | 98977 | runs/v3_null_redesign_geometry_delta_20260807/shard_12.log |
| v3_null_redesign_delta_s13_20260807 | succeeded | v3_null_redesign_delta_s13_v1 | a6000 | 98978 | runs/v3_null_redesign_geometry_delta_20260807/shard_13.log |
| v3_null_redesign_delta_s14_20260807 | succeeded | v3_null_redesign_delta_s14_v1 | a6000 | 98979 | runs/v3_null_redesign_geometry_delta_20260807/shard_14.log |
| v3_null_redesign_delta_s15_20260807 | succeeded | v3_null_redesign_delta_s15_v1 | a6000 | 98980 | runs/v3_null_redesign_geometry_delta_20260807/shard_15.log |
| v3_null_redesign_delta_s16_20260807 | succeeded | v3_null_redesign_delta_s16_v1 | a6000 | 98981 | runs/v3_null_redesign_geometry_delta_20260807/shard_16.log |
| v3_null_redesign_delta_s17_20260807 | succeeded | v3_null_redesign_delta_s17_v1 | a6000 | 98982 | runs/v3_null_redesign_geometry_delta_20260807/shard_17.log |
| v3_null_redesign_delta_s18_20260807 | succeeded | v3_null_redesign_delta_s18_v1 | a6000 | 98983 | runs/v3_null_redesign_geometry_delta_20260807/shard_18.log |
| v3_null_redesign_delta_s19_20260807 | succeeded | v3_null_redesign_delta_s19_v1 | a6000 | 98984 | runs/v3_null_redesign_geometry_delta_20260807/shard_19.log |
| v3_null_redesign_delta_merge_20260807 | succeeded | v3_null_redesign_delta_merge_v1 | a4000 | 98985 | runs/v3_null_redesign_delta_merge_20260807/slurm.log |
| v3_null_redesign_donors_v2_20260807 | succeeded | v3_null_redesign_donor_audit_v2 | a4000 | 98986 | runs/v3_null_redesign_donors_v2_20260807/slurm.log |
| v4_manifest_freeze_20260807 | succeeded | v4_manifest_freeze_20260807 | a4000 | 99105 | runs/v4_manifest_freeze_20260807/slurm.log |
| v4_preflight_20260807 | failed | v4_preflight_20260807 | a6000 | 99107 | runs/v4_preflight_20260807/slurm.log |
| v4_preflight_r2_20260807 | succeeded | v4_preflight_r2_20260807 | a6000 | 99109 | runs/v4_preflight_r2_20260807/slurm.log |
| v4_discovery_s00_20260807 | succeeded | v4_discovery_s00_20260807 | a6000 | 99117 | runs/v4_discovery_s00_20260807/slurm.log |
| v4_discovery_s01_20260807 | succeeded | v4_discovery_s01_20260807 | a6000 | 99118 | runs/v4_discovery_s01_20260807/slurm.log |
| v4_discovery_s02_20260807 | failed | v4_discovery_s02_20260807 | a6000 | 99119 | runs/v4_discovery_s02_20260807/slurm.log |
| v4_discovery_s03_20260807 | succeeded | v4_discovery_s03_20260807 | a6000 | 99120 | runs/v4_discovery_s03_20260807/slurm.log |
| v4_discovery_s02_r2_20260807 | failed | v4_discovery_s02_r2_20260807 | a6000 |  | runs/v4_discovery_s02_r2_20260807/slurm.log |
| v4_discovery_s02_r3_20260807 | succeeded | v4_discovery_s02_r3_20260807 | a6000 | 99121 | runs/v4_discovery_s02_r3_20260807/slurm.log |
| v4_discovery_analysis_20260807 | failed | v4_discovery_analysis_20260807 | a4000 | 99123 | runs/v4_discovery_analysis_20260807/slurm.log |
| v4_discovery_analysis_r2_20260807 | failed | v4_discovery_analysis_r2_20260807 | a4000 | 99124 | runs/v4_discovery_analysis_r2_20260807/slurm.log |
| v4_discovery_analysis_r3_20260807 | succeeded | v4_discovery_analysis_r3_20260807 | a4000 | 99125 | runs/v4_discovery_analysis_r3_20260807/slurm.log |
| v4_discovery_analysis_r4_20260807 | succeeded | v4_discovery_analysis_r4_20260807 | a4000 | 99127 | runs/v4_discovery_analysis_r4_20260807/slurm.log |
| v4_cost_utility_frontier_20260807 | failed | v4_cost_utility_frontier_20260807 | a4000 | 99128 | runs/v4_cost_utility_frontier_20260807/slurm.log |
| v4_cost_utility_frontier_r2_20260807 | succeeded | v4_cost_utility_frontier_r2_20260807 | a4000 | 99129 | runs/v4_cost_utility_frontier_r2_20260807/slurm.log |
| v4_cost_utility_frontier_r3_20260807 | succeeded | v4_cost_utility_frontier_r3_20260807 | a4000 | 99131 | runs/v4_cost_utility_frontier_r3_20260807/slurm.log |
| query_refinement_manifest_20260807 | cancelled | query_refinement_manifest_v1 | auto |  | runs/query_refinement_manifest_v1/slurm.log |
| query_refinement_manifest_a4000_20260807 | succeeded | query_refinement_manifest_v1 | a4000 | 99169 | runs/query_refinement_manifest_v1/slurm.log |
| query_refinement_manifest_audit_fix_20260807 | succeeded | query_refinement_manifest_v1 | a4000 | 99173 | runs/query_refinement_manifest_v1/audit_fix.log |
| query_refinement_preflight_20260807 | failed | query_refinement_preflight_v1 | a6000 | 99174 | runs/query_refinement_preflight_v1/slurm.log |
| query_refinement_preflight_r2_20260807 | succeeded | query_refinement_preflight_v1 | a6000 | 99175 | runs/query_refinement_preflight_v1/retry2.log |
| query_refinement_full_s00_20260807 | succeeded | query_refinement_full_v1 | a6000 | 99176 | runs/query_refinement_full_v1/shard_00.log |
| query_refinement_full_s02_20260807 | failed | query_refinement_full_v1 | a6000 | 99177 | runs/query_refinement_full_v1/shard_02.log |
| query_refinement_full_s03_20260807 | failed | query_refinement_full_v1 | a6000 | 99178 | runs/query_refinement_full_v1/shard_03.log |
| query_refinement_full_s01_20260807 | cancelled | query_refinement_full_v1 | a6000 | 99179 | runs/query_refinement_full_v1/shard_01.log |
| query_refinement_full_s01_r2_20260807 | failed | query_refinement_full_v1 | a6000 | 99180 | runs/query_refinement_full_v1/shard_01.log |
| query_refinement_full_s01_r3_20260807 | succeeded | query_refinement_full_v1 | a6000 | 99182 | runs/query_refinement_full_v1/shard_01_r3.log |
| query_refinement_full_s02_r2_20260807 | cancelled | query_refinement_full_v1 | a6000 | 99183 | runs/query_refinement_full_v1/shard_02_r2.log |
| query_refinement_full_s02_r3_20260807 | succeeded | query_refinement_full_v1 | a6000 | 99184 | runs/query_refinement_full_v1/shard_02_r3.log |
| query_refinement_full_s03_r2_20260807 | succeeded | query_refinement_full_v1 | a6000 | 99185 | runs/query_refinement_full_v1/shard_03_r2.log |
| query_refinement_analysis_20260807 | failed | query_refinement_analysis_v1 | a4000 | 99187 | runs/query_refinement_analysis_v1/slurm.log |
| query_refinement_analysis_r2_20260807 | succeeded | query_refinement_analysis_v1 | a4000 | 99189 | runs/query_refinement_analysis_v1/retry2.log |
| query_refinement_analysis_r3_20260807 | succeeded | query_refinement_analysis_v1 | a4000 | 99191 | runs/query_refinement_analysis_v1/retry3.log |
| binary_env_tf53_20260809 | succeeded | binary_env_tf53_20260809 | a4000 | 99717 | runs/binary_env_tf53_20260809/slurm.log |
| binary_label_audit_bp0_20260809 | succeeded | binary_label_audit_bp0_v1 | a4000 | 99718 | runs/binary_label_audit_bp0_20260809/slurm.log |
| binary_bp1_fixture_freeze_20260809 | succeeded | binary_bp1_fixture_freeze_v1 | a4000 | 99719 | runs/binary_bp1_fixture_freeze_20260809/slurm.log |
| binary_bp1_executor_preflight_20260809 | failed | binary_bp1_executor_preflight_v1 | a6000 | 99721 | runs/binary_bp1_executor_preflight_20260809/slurm.log |
| binary_bp1_executor_preflight_cublas_20260809 | failed | binary_bp1_executor_preflight_v1 | a6000 | 99722 | runs/binary_bp1_executor_preflight_cublas_20260809/slurm.log |
| binary_bp1_executor_diagnostic_20260809 | failed | binary_bp1_executor_diagnostic_v1 | a6000 | 99724 | runs/experiments/binary_bp1_executor_diagnostic_v1/run.log |
| binary_bp1_equivalence_trace_20260809 | succeeded | binary_bp1_equivalence_trace_v1 | a6000 | 99728 | runs/experiments/binary_bp1_equivalence_trace_v1/run.log |
| binary_bp1_executor_preflight_r2_20260809 | failed | binary_bp1_executor_preflight_v2 | a6000 | 99729 | runs/experiments/binary_bp1_executor_preflight_v2/run.log |
| binary_label_runtime_contract_v1 | succeeded | binary_label_runtime_contract_v1 | a6000 | 99730 | runs/experiments/binary_label_runtime_contract_v1/run.log |
| binary_bp1_executor_preflight_r3_20260810 | failed | binary_bp1_executor_preflight_v3 | a6000 | 99737 | runs/experiments/binary_bp1_executor_preflight_v3/run.log |
| label_regeneration_p2_smoke_20260810 | succeeded | label_regeneration_p2_smoke_v1 | a6000 | 99740 | runs/label_regeneration/p2_smoke.log |
| label_regeneration_p3_mcts_8k_20260810 | cancelled | label_regeneration_p3_mcts_8k_v1 | a6000 | 99741 | runs/label_regeneration/p3_mcts_8k.log |
| label_regeneration_p3_mcts_8k_resume_8gpu_20260810 | succeeded | label_regeneration_p3_mcts_8k_v1 | a6000 | 99758 | runs/label_regeneration/p3_mcts_8k_resume_8gpu.log |
| wemath20_standard_download_20260811 | succeeded | wemath20_standard_download_v1 | a4000 | 99824 | runs/dataset_downloads/wemath20_standard_v1.log |
| wemath20_pro_download_20260811 | succeeded | wemath20_pro_download_v1 | a4000 | 99825 | runs/dataset_downloads/wemath20_pro_v1.log |
| wemath20_mathruler_env_20260811 | failed | wemath20_mathruler_env_v1 | a4000 | 99832 | runs/env/wemath20_mathruler_env_v1.log |
| wemath20_pro_manifest_20260811 | failed | wemath20_pro_manifest_v1 | a4000 | 99833 | runs/label_regeneration/wemath2pro_manifest_v1.log |
| wemath20_pro_manifest_v2_20260811 | succeeded | wemath20_pro_manifest_v2 | a4000 | 99847 | runs/label_regeneration/wemath2pro_manifest_v2.log |
| wemath20_pro_mcts_8gpu_20260811 | failed | wemath20_pro_mcts_v1 | a6000 | 99848 | runs/label_regeneration/wemath2pro_mcts_v1.log |
| wemath20_pro_mcts_8gpu_r2_20260811 | failed | wemath20_pro_mcts_v1 | a6000 | 99849 | runs/label_regeneration/wemath2pro_mcts_r2.log |
| wemath20_pro_mcts_8gpu_r3_20260811 | cancelled | wemath20_pro_mcts_v1 | a6000 | 99850 | runs/label_regeneration/wemath2pro_mcts_r3.log |
| label_regeneration_p4_cache_audit_20260812 | succeeded | label_regeneration_p4_cache_audit_v1 | a4000 | 100342 | runs/label_regeneration/p4_cache_audit_v1.log |
| label_regeneration_p5_summary_20260812 | succeeded | label_regeneration_p5_summary_v1 | a4000 | 100344 | runs/label_regeneration/p5_summary_v1.log |
| label_regeneration_p6_diversity_20260812 | succeeded | label_regeneration_p6_diversity_v1 | a4000 | 100345 | runs/label_regeneration/p6_diversity_v1.log |
| binary_router_external_overlap_p7_design_20260812 | succeeded | binary_router_external_overlap_p7_design_v1 | a4000 | 100347 | runs/label_regeneration/external_eval_overlap_split_audit_v1.log |
| binary_router_eval_suite_audit_20260812 | succeeded | p7_eval_suite_audit_v1 | a4000 | 100357 | runs/label_regeneration/eval_suite_overlap_audit_v1.log |
| label_regeneration_p7_split_freeze_20260812 | succeeded | label_regeneration_p7_split_v1 | a4000 | 100358 | runs/label_regeneration/p7_split_v1.log |
| label_regeneration_p7_split_freeze_r2_20260812 | succeeded | label_regeneration_p7_split_v1 | a4000 | 100359 | runs/label_regeneration/p7_split_v1_r2.log |
| label_regeneration_p8_derived_20260812 | cancelled | label_regeneration_p8_derived_v1 | a4000 | 100360 | runs/label_regeneration/p8_derived_v1.log |
| label_regeneration_p8_derived_r2_20260812 | cancelled | label_regeneration_p8_derived_v1 | a4000 | 100361 | runs/label_regeneration/p8_derived_v1_r2.log |
| label_regeneration_p8_derived_r3_20260812 | cancelled | label_regeneration_p8_derived_v1 | a4000 | 100363 | runs/label_regeneration/p8_derived_v1_r3.log |
| label_regeneration_p8_derived_r4_20260812 | succeeded | label_regeneration_p8_derived_v1 | a4000 | 100364 | runs/label_regeneration/p8_derived_v1_r4.log |
| label_regeneration_p8_verify_20260812 | succeeded | label_regeneration_p8_verify_v1 | a4000 | 100365 | runs/label_regeneration/p8_verify_v1.log |
| label_regeneration_p9_finalize_20260812 | failed | label_regeneration_p9_finalize_v1 | a4000 | 100369 | runs/label_regeneration/p9_finalize_v1.log |
| label_regeneration_p9_finalize_repair_20260812 | succeeded | label_regeneration_p9_finalize_v1 | a4000 | 100370 | runs/label_regeneration/p9_finalize_repair_v1.log |
| label_regeneration_p9_verify_checksums_20260812 | succeeded | label_regeneration_p9_verify_v1 | a4000 | 100371 | runs/label_regeneration/p9_verify_checksums_v1.log |
| label_regeneration_p9_finalize_selfcontained_20260812 | succeeded | label_regeneration_p9_finalize_v1 | a4000 | 100372 | runs/label_regeneration/p9_finalize_selfcontained_v1.log |
| label_regeneration_p9_verify_final_20260812 | succeeded | label_regeneration_p9_verify_final_v1 | a4000 | 100373 | runs/label_regeneration/p9_verify_final_v1.log |
| binary_polar_p10_readiness_audit_20260812 | succeeded | binary_polar_p10_readiness_v1 | a4000 | 100374 | runs/binary_polar/p10_readiness_v1.log |
| binary_polar_p10_readiness_final_20260812 | succeeded | binary_polar_p10_readiness_v1 | a4000 | 100380 | runs/binary_polar/p10_readiness_final_v1.log |
| binary_polar_p10_real_encoder_preflight_20260812 | succeeded | binary_polar_p10_real_encoder_preflight_v1 | a4000 | 100386 | runs/binary_polar/p10_real_encoder_preflight_v1.log |
| binary_polar_p10_freeze_readiness_20260812 | succeeded | binary_polar_p10_readiness_gate_v1 | a4000 | 100390 | runs/binary_polar/p10_freeze_readiness_v1.log |
| binary_polar_p10_refreeze_readiness_20260812 | succeeded | binary_polar_p10_readiness_gate_v1 | a4000 | 100391 | runs/binary_polar/p10_refreeze_readiness_v1.log |
| wemath20_resume_audit_v2_20260812 | cancelled | wemath20_resume_audit_v2 | a4000 | 100395 | runs/label_regeneration/wemath2pro_resume_audit_v2.log |
| wemath20_resume_audit_v2_r2_20260812 | failed | wemath20_resume_audit_v2 | a6000 | 100396 | runs/label_regeneration/wemath2pro_resume_audit_v2.log |
| wemath20_resume_audit_v2_r3_20260812 | succeeded | wemath20_resume_audit_v2 | a6000 | 100397 | runs/label_regeneration/wemath2pro_resume_audit_v2_r3.log |
| wemath20_pro_mcts_8gpu_r4_20260812 | cancelled | wemath20_pro_mcts_v2_timeout_repair | a6000 |  | runs/label_regeneration/wemath2pro_mcts_r4.log |
| wemath20_pro_mcts_6gpu_resume_v2_20260812 | cancelled | wemath20_pro_mcts_v2_timeout_repair | a6000 | 100398 | runs/label_regeneration/wemath2pro_mcts_r5.log |
| wemath20_extension_yield_20260812 | cancelled | wemath20_extension_yield_v1 | a4000 | 100400 | runs/label_regeneration/wemath2pro_extension_yield.log |
| wemath20_extension_yield_r2_20260812 | succeeded | wemath20_extension_yield_v1 | a6000 | 100401 | runs/label_regeneration/wemath2pro_extension_yield_r2.log |
| wemath20_cap400_stage_20260813 | cancelled | wemath20_cap400_stage_v1 | a4000 | 100402 | runs/label_regeneration/wemath2pro_cap400_stage.log |
| wemath20_cap400_stage_r2_20260813 | succeeded | wemath20_cap400_stage_v1 | a6000 | 100403 | runs/label_regeneration/wemath2pro_cap400_stage_r2.log |
| wemath20_cap400_stage_v2_20260813 | succeeded | wemath20_cap400_stage_v2 | a6000 | 100404 | runs/label_regeneration/wemath2pro_cap400_stage_v2.log |
| wemath20_pro_mcts_cap400_7gpu_20260813 | cancelled | wemath20_pro_mcts_cap400_v1 | a6000 | 100405 | runs/label_regeneration/wemath2pro_cap400_r1.log |
| wemath20_pro_mcts_cap400_7gpu_r2_20260813 | failed | wemath20_pro_mcts_cap400_v1 | a6000 | 100406 | runs/label_regeneration/wemath2pro_cap400_r2.log |
| wemath20_pro_mcts_cap400_7gpu_r3_20260813 | succeeded | wemath20_pro_mcts_cap400_v1 | a6000 | 100407 | runs/label_regeneration/wemath2pro_cap400_r3.log |
| binary_polar_p10_smoke_duplicated_bce_20260813 | failed | binary_polar_p10_smoke_duplicated_bce_v1 | a6000 | 100408 | runs/binary_polar/p10_smoke_duplicated_bce_v1.log |
| binary_polar_p10_smoke_exact_set_nll_20260813 | failed | binary_polar_p10_smoke_exact_set_nll_v1 | a6000 | 100409 | runs/binary_polar/p10_smoke_exact_set_nll_v1.log |
| binary_polar_p10_smoke_duplicated_bce_r2_20260813 | failed | binary_polar_p10_smoke_duplicated_bce_v1 | a6000 | 100410 | runs/binary_polar/p10_smoke_duplicated_bce_r2_v1.log |
| binary_polar_p10_smoke_exact_set_nll_r2_20260813 | failed | binary_polar_p10_smoke_exact_set_nll_v1 | a6000 | 100411 | runs/binary_polar/p10_smoke_exact_set_nll_r2_v1.log |
| binary_polar_p10_static_repair_v2_20260813 | succeeded | binary_polar_p10_static_repair_v2 | a6000 | 100413 | runs/binary_polar/p10_static_repair_v2.log |
| binary_polar_p10_runtime_repair_v2_20260813 | succeeded | binary_polar_p10_runtime_repair_v2 | a6000 | 100412 | runs/binary_polar/p10_runtime_repair_v2.log |
| binary_polar_p10_smoke_duplicated_bce_r3_20260813 | succeeded | binary_polar_p10_smoke_duplicated_bce_v2 | a6000 | 100414 | runs/binary_polar/p10_smoke_duplicated_bce_r3_v2.log |
| binary_polar_p10_smoke_exact_set_nll_r3_20260813 | succeeded | binary_polar_p10_smoke_exact_set_nll_v2 | a6000 | 100415 | runs/binary_polar/p10_smoke_exact_set_nll_r3_v2.log |
| binary_polar_p11_prepare_20260813 | cancelled | binary_polar_p11_prepare_v1 | a4000 | 100417 | runs/binary_polar/p11_prepare_v1.log |
| binary_polar_p11_prepare_r2_20260813 | succeeded | binary_polar_p11_prepare_v1 | a6000 | 100418 | runs/binary_polar/p11_prepare_r2_v1.log |
| binary_polar_p11_bias_baselines_20260813 | succeeded | binary_polar_p11_bias_baselines_v1 | a6000 | 100419 | runs/binary_polar/p11_bias_v1.log |
| binary_polar_p11_weighted_bce_20260813 | succeeded | binary_polar_p11_weighted_bce_v1 | a6000 | 100420 | runs/binary_polar/p11_weighted_bce_v1.log |
| binary_polar_p11_weighted_setnll_20260813 | succeeded | binary_polar_p11_weighted_setnll_v1 | a6000 | 100421 | runs/binary_polar/p11_weighted_setnll_v1.log |
| binary_polar_p11_execute_bce_20260813 | succeeded | binary_polar_p11_execute_bce_v1 | a6000 | 100422 | runs/binary_polar/p11_execute_bce_v1.log |
| binary_polar_p11_execute_setnll_20260813 | succeeded | binary_polar_p11_execute_setnll_v1 | a6000 | 100423 | runs/binary_polar/p11_execute_setnll_v1.log |
| binary_polar_p12_geometry_20260813 | cancelled | binary_polar_p12_geometry_v1 | a4000 | 100509 | runs/binary_polar/p12_geometry_v1.log |
| binary_polar_p12_geometry_r2_20260813 | succeeded | binary_polar_p12_geometry_v1 | a6000 | 100510 | runs/binary_polar/p12_geometry_r2_v1.log |
| binary_polar_p12_bf16_preflight_20260813 | succeeded | binary_polar_p12_bf16_preflight_v1 | a6000 | 100525 | runs/binary_polar/p12/bf16_preflight_%j.log |
| binary_polar_p12_structured_smoke_20260813 | succeeded | binary_polar_p12_structured_smoke_v1 | a6000 | 100544 | runs/binary_polar/p12/structured_smoke_%j.log |
| binary_polar_p12_conditioning_20260813 | succeeded | binary_polar_p12_conditioning_v1 | a6000 | 100550 | runs/binary_polar/p12/conditioning_%j.log |
| binary_polar_p12_execution_20260813 | succeeded | binary_polar_p12_execution_v1 | a6000 | 100554 | runs/binary_polar/p12/execution_%j.log |
| binary_polar_p13_visual_cache_20260813 | succeeded | binary_polar_p13_visual_cache_v1 | a6000 | 100734 | runs/binary_polar/p13/visual_cache_%j.log |
| binary_polar_p13_bf16_preflight_20260813 | succeeded | binary_polar_p13_bf16_preflight_v1 | a6000 | 100736 | runs/binary_polar/p13/bf16_preflight_%j.log |
| binary_polar_p13_freeze_readiness_20260813 | cancelled | binary_polar_p13_readiness_v1 | a4000 | 100737 | runs/binary_polar/p13/readiness_%j.log |
| binary_polar_p13_freeze_readiness_r2_20260813 | cancelled | binary_polar_p13_readiness_v2 | a4000 | 100738 | runs/binary_polar/p13_readiness_v2_%j.log |
| binary_polar_p13_freeze_readiness_r3_20260813 | succeeded | binary_polar_p13_readiness_v2 | a6000 | 100739 | runs/binary_polar/p13_readiness_v2_r3_%j.log |
| binary_polar_p13_smoke_question_20260813 | succeeded | binary_polar_p13_smoke_question_v1 | a6000 | 100744 | runs/binary_polar/p13_question_%j.log |
| binary_polar_p13_smoke_image_20260813 | succeeded | binary_polar_p13_smoke_image_v1 | a6000 | 100742 | runs/binary_polar/p13_image_%j.log |
| binary_polar_p13_smoke_image_question_20260813 | succeeded | binary_polar_p13_smoke_image_question_v1 | a6000 | 100743 | runs/binary_polar/p13_image_question_%j.log |
| binary_polar_p13_conditioning_20260813 | succeeded | binary_polar_p13_conditioning_v1 | a6000 | 100745 | runs/binary_polar/p13_conditioning_%j.log |
| binary_polar_full10_visual_cache_20260813 | succeeded | binary_polar_full10_visual_cache_v1 | a6000 | 100746 | runs/binary_polar/full10_visual_cache_%j.log |
| binary_polar_full10_preflight_20260813 | succeeded | binary_polar_full10_preflight_v1 | a6000 | 100753 | runs/binary_polar/full10_preflight_%j.log |
| binary_polar_full10_question_20260813 | succeeded | binary_polar_full10_question_v1 | a6000 | 100757 | runs/binary_polar/full10_question_%j.log |
| binary_polar_full10_image_question_20260813 | succeeded | binary_polar_full10_image_question_v1 | a6000 | 100756 | runs/binary_polar/full10_image_question_%j.log |
| binary_polar_full10_conditioning_20260813 | succeeded | full10_conditioning | a6000 | 100760 | runs/binary_polar/full10_conditioning_%j.log |
| binary_polar_full10_exec_q_best_20260813 | failed | full10_exec_q_best | a6000 | 100762 | runs/binary_polar/full10_exec_q_best_%j.log |
| binary_polar_full10_exec_iq_best_20260813 | failed | full10_exec_iq_best | a6000 | 100761 | runs/binary_polar/full10_exec_iq_best_%j.log |
| binary_polar_full10_exec_q_best_r2_20260813 | succeeded | full10_exec_q_best_r2 | a6000 | 100764 | runs/binary_polar/full10_exec_q_best_r2_%j.log |
| binary_polar_full10_exec_iq_best_r2_20260813 | succeeded | full10_exec_iq_best_r2 | a6000 | 100763 | runs/binary_polar/full10_exec_iq_best_r2_%j.log |
| binary_polar_full10_exec_q_final_20260813 | succeeded | full10_exec_q_final | a6000 | 100767 | runs/binary_polar/full10_exec_q_final_%j.log |
| binary_polar_full10_exec_iq_final_20260813 | succeeded | full10_exec_iq_final | a6000 | 100766 | runs/binary_polar/full10_exec_iq_final_%j.log |
| bp_external_preflight_v1 | failed | bp_external_preflight_v1 | a6000 | 100778 | runs/binary_polar/external_eval/preflight_%j.log |
| bp_external_env_qwen_vl_utils_v1 | succeeded | bp_external_env_qwen_vl_utils_v1 | a4000 | 100779 | runs/binary_polar/external_eval/env_qwen_vl_utils_%j.log |
| bp_external_preflight_v1_r2 | failed | bp_external_preflight_v1 | a6000 | 100780 | runs/binary_polar/external_eval/preflight_r2_%j.log |
| bp_external_freeze_contract_v1 | succeeded | bp_external_freeze_contract_v1 | a4000 | 100781 | runs/binary_polar/external_eval/freeze_contract_%j.log |
| bp_external_preflight_v1_r3 | succeeded | bp_external_preflight_v1 | a6000 | 100782 | runs/binary_polar/external_eval/preflight_r3_%j.log |
| bp_external_preflight_question_v1 | succeeded | bp_external_question_v1 | a6000 | 100784 | runs/binary_polar/external_eval/question_check_%j.log |
| bp_external_preflight_image_question_v1 | succeeded | bp_external_image_question_v1 | a6000 | 100783 | runs/binary_polar/external_eval/image_question_check_%j.log |
| bp_external_full_question_v1 | cancelled | bp_external_question_v1 | a6000 | 100786 | runs/binary_polar/external_eval/question_full_%j.log |
| bp_external_full_image_question_v1 | cancelled | bp_external_image_question_v1 | a6000 | 100785 | runs/binary_polar/external_eval/image_question_full_%j.log |
| bp_external_full_question_v1_r2 | succeeded | bp_external_question_v1 | a6000 | 100788 | runs/binary_polar/external_eval/question_full_r2_%j.log |
| bp_external_full_image_question_v1_r2 | succeeded | bp_external_image_question_v1 | a6000 | 100787 | runs/binary_polar/external_eval/image_question_full_r2_%j.log |
| bp_external_wait_merge_v1 | succeeded | bp_external_analysis_v1 | a4000 | 100790 | runs/binary_polar/external_eval/wait_merge_%j.log |
| binary_polar_full10_bce_preflight_v1 | succeeded | binary_polar_full10_bce_preflight_v1 | a6000 | 101019 | runs/binary_polar_full10_bce_preflight_v1/slurm.log |
| binary_polar_full10_bce_question_v1 | failed | binary_polar_full10_bce_question_v1 | a6000 | 101021 | runs/binary_polar_full10_bce_question_v1/slurm.log |
| binary_polar_full10_bce_image_question_v1 | failed | binary_polar_full10_bce_image_question_v1 | a6000 | 101020 | runs/binary_polar_full10_bce_image_question_v1/slurm.log |
| binary_polar_full10_bce_question_v1_r2 | succeeded | binary_polar_full10_bce_question_v1 | a6000 | 101023 | runs/binary_polar_full10_bce_question_v1/slurm_r2.log |
| binary_polar_full10_bce_image_question_v1_r2 | succeeded | binary_polar_full10_bce_image_question_v1 | a6000 | 101022 | runs/binary_polar_full10_bce_image_question_v1/slurm_r2.log |
| mcts-bce-label-analysis-v1 | cancelled | mcts-bce-label-analysis-v1 | a4000 | 101273 | runs/mcts_bce_label_analysis_v1/slurm.log |
| mcts-bce-label-analysis-v2 | failed | mcts-bce-label-analysis-v1 | a4000 | 101274 | runs/mcts_bce_label_analysis_v2/slurm.log |
| mcts-bce-label-analysis-v3 | failed | mcts-bce-label-analysis-v1 | a4000 | 101275 | runs/mcts_bce_label_analysis_v3/slurm.log |
| mcts-bce-label-analysis-v4 | succeeded | mcts-bce-label-analysis-v1 | a4000 | 101276 | runs/mcts_bce_label_analysis_v4/slurm.log |
| wemath2pro-label-analysis-v1 | failed | wemath2pro-label-analysis-v1 | a4000 | 101422 | runs/wemath2pro_label_analysis_v1/slurm.log |
| wemath2pro-label-analysis-v2 | succeeded | wemath2pro-label-analysis-v1 | a4000 | 101423 | runs/wemath2pro_label_analysis_v1/slurm_v2.log |
| binary_pareto_manifest_prepare_v1 | succeeded | binary_pareto_manifest_prepare_v1 | a4000 | 101470 | runs/binary_pareto_manifest_prepare_v1/slurm.log |
| binary_pareto_full10_bce_train_eval_v1 | running | binary_pareto_full10_bce_train_eval_v1 | a6000 | 101489 | runs/binary_pareto_v1/bce/slurm.log |
| binary_pareto_full10_nll_train_eval_v1 | running | binary_pareto_full10_nll_train_eval_v1 | a6000 | 101511 | runs/binary_pareto_v1/nll/slurm.log |
| binary_pareto_training_fit_smoke_v1 | failed | binary_pareto_training_fit_smoke_v1 | a4000 | 101688 | runs/binary_pareto_v1/training_fit_analysis_v1/smoke_bce.log |
| binary_pareto_training_fit_smoke_v2 | succeeded | binary_pareto_training_fit_smoke_v2 | a4000 | 101689 | runs/binary_pareto_v1/training_fit_analysis_v1/smoke_bce_v2.log |
| binary_pareto_bce_training_fit_v1 | cancelled | binary_pareto_bce_training_fit_v1 | a4000 | 101690 | runs/binary_pareto_v1/training_fit_analysis_v1/bce.log |
| binary_pareto_nll_training_fit_v1 | cancelled | binary_pareto_nll_training_fit_v1 | a4000 | 101691 | runs/binary_pareto_v1/training_fit_analysis_v1/nll.log |
| binary_pareto_bce_training_fit_v2 | failed | binary_pareto_bce_training_fit_v2 | a4000 | 101692 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v2.log |
| binary_pareto_nll_training_fit_v2 | cancelled | binary_pareto_nll_training_fit_v2 | a6000 | 101693 | runs/binary_pareto_v1/training_fit_analysis_v1/nll_v2.log |
| binary_pareto_bce_training_fit_v3 | cancelled | binary_pareto_bce_training_fit_v3 | a6000 | 101694 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v3.log |
| binary_pareto_bce_training_fit_v4 | cancelled | binary_pareto_bce_training_fit_v4 | a6000 | 101695 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v4.log |
| binary_pareto_bce_training_fit_v5 | failed | binary_pareto_bce_training_fit_v5 | a4000 | 101696 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v5.log |
| binary_pareto_nll_training_fit_v3 | failed | binary_pareto_nll_training_fit_v3 | a4000 | 101697 | runs/binary_pareto_v1/training_fit_analysis_v1/nll_v3.log |
| binary_pareto_nll_training_fit_v4 | failed | binary_pareto_nll_training_fit_v4 | a4000 | 101698 | runs/binary_pareto_v1/training_fit_analysis_v1/nll_v4.log |
| binary_pareto_nll_training_fit_v5 | succeeded | binary_pareto_nll_training_fit_v5 | a4000 | 101699 | runs/binary_pareto_v1/training_fit_analysis_v1/nll_v5.log |
| binary_pareto_bce_training_fit_v6 | succeeded | binary_pareto_bce_training_fit_v6 | a4000 | 101700 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v6.log |
| wemath-greedy-manifest-v1 | succeeded | wemath2pro-greedy-recovery-v1 | a4000 | 101705 | runs/wemath_greedy_recovery/manifest.log |
| wemath-greedy-preflight-v1 | failed | wemath2pro-greedy-recovery-preflight-v1 | a6000 | 101706 | runs/wemath_greedy_recovery/preflight.log |
| wemath-greedy-preflight-v2 | succeeded | wemath2pro-greedy-recovery-preflight-v2 | a6000 | 101707 | runs/wemath_greedy_recovery/preflight_r2.log |
| wemath-greedy-phase1-node06-v1 | running | wemath2pro-greedy-phase1-node06-v1 | a6000 | 101708 | runs/wemath_greedy_recovery/phase1_node06.log |
| wemath-greedy-phase1-node07-v1 | running | wemath2pro-greedy-phase1-node07-v1 | a6000 | 101709 | runs/wemath_greedy_recovery/phase1_node07.log |

## Run States

| exp_id | status | current_step | log | result |
|---|---|---|---|---|
| binary_bp1_equivalence_trace_v1 | running | Running GPU job binary_bp1_equivalence_trace_20260809 | runs/experiments/binary_bp1_equivalence_trace_v1/run.log | outputs/binary_polar/preflight/executor_equivalence_trace_v1.json |
| binary_bp1_executor_diagnostic_v1 | running | Running GPU job binary_bp1_executor_diagnostic_20260809 | runs/experiments/binary_bp1_executor_diagnostic_v1/run.log | outputs/binary_polar/preflight/executor_diagnostic_v1.json |
| binary_bp1_executor_preflight_v1 | running | Running GPU job binary_bp1_executor_preflight_cublas_20260809 | runs/binary_bp1_executor_preflight_cublas_20260809/slurm.log | outputs/binary_polar/preflight/executor_preflight_v1.json |
| binary_bp1_executor_preflight_v2 | running | Running GPU job binary_bp1_executor_preflight_r2_20260809 | runs/experiments/binary_bp1_executor_preflight_v2/run.log | outputs/binary_polar/preflight/executor_preflight_v2.json |
| binary_bp1_executor_preflight_v3 | failed | BP-1 input-contract repair rerun failed the frozen cached-ID gate; training blocked | runs/experiments/binary_bp1_executor_preflight_v3/run.log | outputs/binary_polar/preflight/executor_preflight_v3.json |
| binary_bp1_fixture_freeze_v1 | running | Running GPU job binary_bp1_fixture_freeze_20260809 | runs/binary_bp1_fixture_freeze_20260809/slurm.log | outputs/binary_polar/preflight/executor_fixtures_v1.json |
| binary_env_tf53_20260809 | running | Running GPU job binary_env_tf53_20260809 | runs/binary_env_tf53_20260809/slurm.log | outputs/env_migrations/transformers_5_3_0_v1.json |
| binary_label_audit_bp0_v1 | running | Running GPU job binary_label_audit_bp0_20260809 | runs/binary_label_audit_bp0_20260809/slurm.log | /data/dataset/dynamic_mllm/binary_polar_v1/binary_polar_label_geometry_audit_v1.json |
| binary_label_runtime_contract_v1 | running | Running GPU job binary_label_runtime_contract_v1 | runs/experiments/binary_label_runtime_contract_v1/run.log | outputs/binary_polar/preflight/label_runtime_contract_v1.json |
| binary_pareto_bce_training_fit_v1 | running | Running GPU job binary_pareto_bce_training_fit_v1 | runs/binary_pareto_v1/training_fit_analysis_v1/bce.log | outputs/binary_pareto_v1/training_fit_analysis_v1/bce_training_fit_v1.json |
| binary_pareto_bce_training_fit_v2 | running | Running GPU job binary_pareto_bce_training_fit_v2 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v2.log | outputs/binary_pareto_v1/training_fit_analysis_v1/bce_training_fit_v1.json |
| binary_pareto_bce_training_fit_v3 | running | Running GPU job binary_pareto_bce_training_fit_v3 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v3.log | outputs/binary_pareto_v1/training_fit_analysis_v1/bce_training_fit_v1.json |
| binary_pareto_bce_training_fit_v4 | running | Running GPU job binary_pareto_bce_training_fit_v4 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v4.log | outputs/binary_pareto_v1/training_fit_analysis_v1/bce_training_fit_v1.json |
| binary_pareto_bce_training_fit_v5 | running | Running GPU job binary_pareto_bce_training_fit_v5 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v5.log | outputs/binary_pareto_v1/training_fit_analysis_v1/bce_training_fit_v1.json |
| binary_pareto_bce_training_fit_v6 | running | Running GPU job binary_pareto_bce_training_fit_v6 | runs/binary_pareto_v1/training_fit_analysis_v1/bce_v6.log | outputs/binary_pareto_v1/training_fit_analysis_v1/bce_training_fit_v1.json |
| binary_pareto_full10_bce_train_eval_v1 | running | Running GPU job binary_pareto_full10_bce_train_eval_v1 | runs/binary_pareto_v1/bce/slurm.log | outputs/binary_pareto_v1/bce_pipeline_complete.json |
| binary_pareto_full10_nll_train_eval_v1 | running | Running GPU job binary_pareto_full10_nll_train_eval_v1 | runs/binary_pareto_v1/nll/slurm.log | outputs/binary_pareto_v1/nll_pipeline_complete.json |
| binary_pareto_manifest_prepare_v1 | running | Running GPU job binary_pareto_manifest_prepare_v1 | runs/binary_pareto_manifest_prepare_v1/slurm.log | outputs/binary_pareto_v1/audits/pareto_integrity_audit_v1.json |
| binary_pareto_nll_training_fit_v1 | running | Running GPU job binary_pareto_nll_training_fit_v1 | runs/binary_pareto_v1/training_fit_analysis_v1/nll.log | outputs/binary_pareto_v1/training_fit_analysis_v1/nll_training_fit_v1.json |
| binary_pareto_nll_training_fit_v2 | running | Running GPU job binary_pareto_nll_training_fit_v2 | runs/binary_pareto_v1/training_fit_analysis_v1/nll_v2.log | outputs/binary_pareto_v1/training_fit_analysis_v1/nll_training_fit_v1.json |
| binary_pareto_nll_training_fit_v3 | running | Running GPU job binary_pareto_nll_training_fit_v3 | runs/binary_pareto_v1/training_fit_analysis_v1/nll_v3.log | outputs/binary_pareto_v1/training_fit_analysis_v1/nll_training_fit_v1.json |
| binary_pareto_nll_training_fit_v4 | running | Running GPU job binary_pareto_nll_training_fit_v4 | runs/binary_pareto_v1/training_fit_analysis_v1/nll_v4.log | outputs/binary_pareto_v1/training_fit_analysis_v1/nll_training_fit_v1.json |
| binary_pareto_nll_training_fit_v5 | running | Running GPU job binary_pareto_nll_training_fit_v5 | runs/binary_pareto_v1/training_fit_analysis_v1/nll_v5.log | outputs/binary_pareto_v1/training_fit_analysis_v1/nll_training_fit_v1.json |
| binary_pareto_training_fit_smoke_v1 | running | Running GPU job binary_pareto_training_fit_smoke_v1 | runs/binary_pareto_v1/training_fit_analysis_v1/smoke_bce.log | outputs/binary_pareto_v1/training_fit_analysis_v1/smoke_bce_epoch1.json |
| binary_pareto_training_fit_smoke_v2 | running | Running GPU job binary_pareto_training_fit_smoke_v2 | runs/binary_pareto_v1/training_fit_analysis_v1/smoke_bce_v2.log | outputs/binary_pareto_v1/training_fit_analysis_v1/smoke_bce_epoch1.json |
| binary_polar_full10_bce_image_question_v1 | running | Running GPU job binary_polar_full10_bce_image_question_v1_r2 | runs/binary_polar_full10_bce_image_question_v1/slurm_r2.log | outputs/binary_polar/external_eval/full10_bce_v1/image_question_pipeline_complete.json |
| binary_polar_full10_bce_preflight_v1 | running | Running GPU job binary_polar_full10_bce_preflight_v1 | runs/binary_polar_full10_bce_preflight_v1/slurm.log | outputs/binary_polar/full10_bce/preflight_v1.json |
| binary_polar_full10_bce_question_v1 | running | Running GPU job binary_polar_full10_bce_question_v1_r2 | runs/binary_polar_full10_bce_question_v1/slurm_r2.log | outputs/binary_polar/external_eval/full10_bce_v1/question_pipeline_complete.json |
| binary_polar_full10_image_question_v1 | running | Running GPU job binary_polar_full10_image_question_20260813 | runs/binary_polar/full10_image_question_%j.log | outputs/binary_polar/full10/image_question_v1/training_summary.json |
| binary_polar_full10_preflight_v1 | running | Running GPU job binary_polar_full10_preflight_20260813 | runs/binary_polar/full10_preflight_%j.log | outputs/binary_polar/full10/preflight_v1.json |
| binary_polar_full10_question_v1 | running | Running GPU job binary_polar_full10_question_20260813 | runs/binary_polar/full10_question_%j.log | outputs/binary_polar/full10/question_v1/training_summary.json |
| binary_polar_full10_visual_cache_v1 | running | Running GPU job binary_polar_full10_visual_cache_20260813 | runs/binary_polar/full10_visual_cache_%j.log | outputs/binary_polar/full10/visual_features_v1/cache_audit_v1.json |
| binary_polar_p10_readiness_gate_v1 | running | Running GPU job binary_polar_p10_refreeze_readiness_20260812 | runs/binary_polar/p10_refreeze_readiness_v1.log | outputs/binary_polar/preflight/p10_readiness_gate_v1.json |
| binary_polar_p10_readiness_v1 | running | Running GPU job binary_polar_p10_readiness_final_20260812 | runs/binary_polar/p10_readiness_final_v1.log | outputs/binary_polar/preflight/p10_training_readiness_v1.json |
| binary_polar_p10_real_encoder_preflight_v1 | running | Running GPU job binary_polar_p10_real_encoder_preflight_20260812 | runs/binary_polar/p10_real_encoder_preflight_v1.log | outputs/binary_polar/preflight/p10_real_encoder_preflight_v1.json |
| binary_polar_p10_runtime_repair_v2 | running | Running GPU job binary_polar_p10_runtime_repair_v2_20260813 | runs/binary_polar/p10_runtime_repair_v2.log | outputs/binary_polar/preflight/repair_v2/p10_real_encoder_preflight_v2.json |
| binary_polar_p10_smoke_duplicated_bce_v1 | running | Running GPU job binary_polar_p10_smoke_duplicated_bce_r2_20260813 | runs/binary_polar/p10_smoke_duplicated_bce_r2_v1.log | outputs/binary_polar/p10_smoke/duplicated_bce_execution_v1.json |
| binary_polar_p10_smoke_duplicated_bce_v2 | running | Running GPU job binary_polar_p10_smoke_duplicated_bce_r3_20260813 | runs/binary_polar/p10_smoke_duplicated_bce_r3_v2.log | outputs/binary_polar/p10_smoke/duplicated_bce_execution_v2.json |
| binary_polar_p10_smoke_exact_set_nll_v1 | running | Running GPU job binary_polar_p10_smoke_exact_set_nll_r2_20260813 | runs/binary_polar/p10_smoke_exact_set_nll_r2_v1.log | outputs/binary_polar/p10_smoke/exact_set_nll_execution_v1.json |
| binary_polar_p10_smoke_exact_set_nll_v2 | running | Running GPU job binary_polar_p10_smoke_exact_set_nll_r3_20260813 | runs/binary_polar/p10_smoke_exact_set_nll_r3_v2.log | outputs/binary_polar/p10_smoke/exact_set_nll_execution_v2.json |
| binary_polar_p10_static_repair_v2 | running | Running GPU job binary_polar_p10_static_repair_v2_20260813 | runs/binary_polar/p10_static_repair_v2.log | outputs/binary_polar/preflight/repair_v2/p10_training_readiness_v2.json |
| binary_polar_p11_bias_baselines_v1 | running | Running GPU job binary_polar_p11_bias_baselines_20260813 | runs/binary_polar/p11_bias_v1.log | outputs/binary_polar/p11/bias |
| binary_polar_p11_execute_bce_v1 | running | Running GPU job binary_polar_p11_execute_bce_20260813 | runs/binary_polar/p11_execute_bce_v1.log | outputs/binary_polar/p11/execution/duplicated_bce_v1.json |
| binary_polar_p11_execute_setnll_v1 | running | Running GPU job binary_polar_p11_execute_setnll_20260813 | runs/binary_polar/p11_execute_setnll_v1.log | outputs/binary_polar/p11/execution/exact_set_nll_v1.json |
| binary_polar_p11_prepare_v1 | running | Running GPU job binary_polar_p11_prepare_r2_20260813 | runs/binary_polar/p11_prepare_r2_v1.log | outputs/binary_polar/p11/label_geometry_v1.json |
| binary_polar_p11_weighted_bce_v1 | running | Running GPU job binary_polar_p11_weighted_bce_20260813 | runs/binary_polar/p11_weighted_bce_v1.log | outputs/binary_polar/p11/question/duplicated_bce_v1/best_checkpoint.json |
| binary_polar_p11_weighted_setnll_v1 | running | Running GPU job binary_polar_p11_weighted_setnll_20260813 | runs/binary_polar/p11_weighted_setnll_v1.log | outputs/binary_polar/p11/question/exact_set_nll_v1/best_checkpoint.json |
| binary_polar_p12_bf16_preflight_v1 | running | Running GPU job binary_polar_p12_bf16_preflight_20260813 | runs/binary_polar/p12/bf16_preflight_%j.log | outputs/binary_polar/preflight/p12_bf16_preflight_v1.json |
| binary_polar_p12_conditioning_v1 | running | Running GPU job binary_polar_p12_conditioning_20260813 | runs/binary_polar/p12/conditioning_%j.log | outputs/binary_polar/p12/structured_conditioning_v1.json |
| binary_polar_p12_execution_v1 | running | Running GPU job binary_polar_p12_execution_20260813 | runs/binary_polar/p12/execution_%j.log | outputs/binary_polar/p12/structured_execution_v1.json |
| binary_polar_p12_geometry_v1 | running | Running GPU job binary_polar_p12_geometry_r2_20260813 | runs/binary_polar/p12_geometry_r2_v1.log | outputs/binary_polar/p12/segment_geometry_v1.json |
| binary_polar_p12_structured_smoke_v1 | running | Running GPU job binary_polar_p12_structured_smoke_20260813 | runs/binary_polar/p12/structured_smoke_%j.log | outputs/binary_polar/p12/structured_exact_set_v1/best_checkpoint.json |
| binary_polar_p13_bf16_preflight_v1 | running | Running GPU job binary_polar_p13_bf16_preflight_20260813 | runs/binary_polar/p13/bf16_preflight_%j.log | outputs/binary_polar/preflight/p13_bf16_preflight_v1.json |
| binary_polar_p13_conditioning_v1 | running | Running GPU job binary_polar_p13_conditioning_20260813 | runs/binary_polar/p13_conditioning_%j.log | outputs/binary_polar/p13/conditioning_diagnostic_v1.json |
| binary_polar_p13_readiness_v1 | running | Running GPU job binary_polar_p13_freeze_readiness_20260813 | runs/binary_polar/p13/readiness_%j.log | outputs/binary_polar/preflight/p13_readiness_gate_v1.json |
| binary_polar_p13_readiness_v2 | running | Running GPU job binary_polar_p13_freeze_readiness_r3_20260813 | runs/binary_polar/p13_readiness_v2_r3_%j.log | outputs/binary_polar/preflight/p13_readiness_gate_v2.json |
| binary_polar_p13_smoke_image_question_v1 | running | Running GPU job binary_polar_p13_smoke_image_question_20260813 | runs/binary_polar/p13_image_question_%j.log | outputs/binary_polar/p13/image_question_v1/best_checkpoint.json |
| binary_polar_p13_smoke_image_v1 | running | Running GPU job binary_polar_p13_smoke_image_20260813 | runs/binary_polar/p13_image_%j.log | outputs/binary_polar/p13/image_v1/best_checkpoint.json |
| binary_polar_p13_smoke_question_v1 | running | Running GPU job binary_polar_p13_smoke_question_20260813 | runs/binary_polar/p13_question_%j.log | outputs/binary_polar/p13/question_v1/best_checkpoint.json |
| binary_polar_p13_visual_cache_v1 | running | Running GPU job binary_polar_p13_visual_cache_20260813 | runs/binary_polar/p13/visual_cache_%j.log | outputs/binary_polar/p13/visual_features_v1/cache_audit_v1.json |
| binary_router_external_overlap_p7_design_v1 | running | Running GPU job binary_router_external_overlap_p7_design_20260812 | runs/label_regeneration/external_eval_overlap_split_audit_v1.log | outputs/label_regeneration/v1/post_generation/external_eval_overlap_split_audit_v1.json |
| bp_external_analysis_v1 | running | Running GPU job bp_external_wait_merge_v1 | runs/binary_polar/external_eval/wait_merge_%j.log | outputs/binary_polar/external_eval/full10_best_v1/analysis_manifest_v1.json |
| bp_external_env_qwen_vl_utils_v1 | running | Running GPU job bp_external_env_qwen_vl_utils_v1 | runs/binary_polar/external_eval/env_qwen_vl_utils_%j.log | .venv/lib/python3.12/site-packages/qwen_vl_utils/__init__.py |
| bp_external_freeze_contract_v1 | running | Running GPU job bp_external_freeze_contract_v1 | runs/binary_polar/external_eval/freeze_contract_%j.log | outputs/binary_polar/external_eval/full10_best_v1/evaluation_contract_v1.json |
| bp_external_image_question_v1 | running | Running GPU job bp_external_full_image_question_v1_r2 | runs/binary_polar/external_eval/image_question_full_r2_%j.log | outputs/binary_polar/external_eval/full10_best_v1/image_question/shard_000_of_001/metadata.json |
| bp_external_preflight_v1 | running | Running GPU job bp_external_preflight_v1_r3 | runs/binary_polar/external_eval/preflight_r3_%j.log | outputs/binary_polar/external_eval/full10_best_v1/preflight_v1.json |
| bp_external_question_v1 | running | Running GPU job bp_external_full_question_v1_r2 | runs/binary_polar/external_eval/question_full_r2_%j.log | outputs/binary_polar/external_eval/full10_best_v1/question/shard_000_of_001/metadata.json |
| full10_conditioning | running | Running GPU job binary_polar_full10_conditioning_20260813 | runs/binary_polar/full10_conditioning_%j.log | outputs/binary_polar/full10/conditioning_v1.json |
| full10_exec_iq_best | running | Running GPU job binary_polar_full10_exec_iq_best_20260813 | runs/binary_polar/full10_exec_iq_best_%j.log | outputs/binary_polar/full10/execution_image_question_best_hit_at_1_v1.json |
| full10_exec_iq_best_r2 | running | Running GPU job binary_polar_full10_exec_iq_best_r2_20260813 | runs/binary_polar/full10_exec_iq_best_r2_%j.log | outputs/binary_polar/full10/execution_image_question_best_hit_at_1_v1.json |
| full10_exec_iq_final | running | Running GPU job binary_polar_full10_exec_iq_final_20260813 | runs/binary_polar/full10_exec_iq_final_%j.log | outputs/binary_polar/full10/execution_image_question_final_v1.json |
| full10_exec_q_best | running | Running GPU job binary_polar_full10_exec_q_best_20260813 | runs/binary_polar/full10_exec_q_best_%j.log | outputs/binary_polar/full10/execution_question_best_hit_at_1_v1.json |
| full10_exec_q_best_r2 | running | Running GPU job binary_polar_full10_exec_q_best_r2_20260813 | runs/binary_polar/full10_exec_q_best_r2_%j.log | outputs/binary_polar/full10/execution_question_best_hit_at_1_v1.json |
| full10_exec_q_final | running | Running GPU job binary_polar_full10_exec_q_final_20260813 | runs/binary_polar/full10_exec_q_final_%j.log | outputs/binary_polar/full10/execution_question_final_v1.json |
| label_regeneration_p2_smoke_v1 | running | Running GPU job label_regeneration_p2_smoke_20260810 | runs/label_regeneration/p2_smoke.log | outputs/label_regeneration/v1/smoke_report_v1.json |
| label_regeneration_p3_mcts_8k_v1 | running | Running GPU job label_regeneration_p3_mcts_8k_resume_8gpu_20260810 | runs/label_regeneration/p3_mcts_8k_resume_8gpu.log | outputs/label_regeneration/v1/raw_route_cache |
| label_regeneration_p4_cache_audit_v1 | running | Running GPU job label_regeneration_p4_cache_audit_20260812 | runs/label_regeneration/p4_cache_audit_v1.log | outputs/label_regeneration/v1/post_generation/cache_audit_v1.json |
| label_regeneration_p5_summary_v1 | running | Running GPU job label_regeneration_p5_summary_20260812 | runs/label_regeneration/p5_summary_v1.log | outputs/label_regeneration/v1/post_generation/label_quality_summary_p5_v1.json |
| label_regeneration_p6_diversity_v1 | running | Running GPU job label_regeneration_p6_diversity_20260812 | runs/label_regeneration/p6_diversity_v1.log | outputs/label_regeneration/v1/post_generation/route_diversity_summary_p6_v1.json |
| label_regeneration_p7_split_v1 | running | Running GPU job label_regeneration_p7_split_freeze_r2_20260812 | runs/label_regeneration/p7_split_v1_r2.log | outputs/label_regeneration/v1/post_generation/predictor_split_audit_v1.json |
| label_regeneration_p8_derived_v1 | running | Running GPU job label_regeneration_p8_derived_r4_20260812 | runs/label_regeneration/p8_derived_v1_r4.log | outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json |
| label_regeneration_p8_verify_v1 | running | Running GPU job label_regeneration_p8_verify_20260812 | runs/label_regeneration/p8_verify_v1.log | outputs/label_regeneration/v1/post_generation/derived_supervision_verification_v1.json |
| label_regeneration_p9_finalize_v1 | running | Running GPU job label_regeneration_p9_finalize_selfcontained_20260812 | runs/label_regeneration/p9_finalize_selfcontained_v1.log | outputs/label_regeneration/v1/post_generation/p9_final_audit_v1.json |
| label_regeneration_p9_verify_final_v1 | running | Running GPU job label_regeneration_p9_verify_final_20260812 | runs/label_regeneration/p9_verify_final_v1.log | outputs/label_regeneration/v1/post_generation/P9_SHA256SUMS |
| label_regeneration_p9_verify_v1 | running | Running GPU job label_regeneration_p9_verify_checksums_20260812 | runs/label_regeneration/p9_verify_checksums_v1.log | outputs/label_regeneration/v1/post_generation/P9_SHA256SUMS |
| mcts-bce-label-analysis-v1 | running | Running GPU job mcts-bce-label-analysis-v4 | runs/mcts_bce_label_analysis_v4/slurm.log | outputs/binary_mcts_label_geometry_v1/analysis_manifest.json |
| p7_eval_suite_audit_v1 | running | Running GPU job binary_router_eval_suite_audit_20260812 | runs/label_regeneration/eval_suite_overlap_audit_v1.log | outputs/label_regeneration/v1/post_generation/eval_suite_overlap_audit_v1.json |
| query_refinement_analysis_v1 | running | Running GPU job query_refinement_analysis_r3_20260807 | runs/query_refinement_analysis_v1/retry3.log | outputs/query_refinement/analysis_manifest.json |
| query_refinement_full_v1 | running | Running GPU job query_refinement_full_s03_r2_20260807 | runs/query_refinement_full_v1/shard_03_r2.log | outputs/query_refinement/shards_v1/shard_03/completion.json |
| query_refinement_manifest_v1 | running | Running GPU job query_refinement_manifest_audit_fix_20260807 | runs/query_refinement_manifest_v1/audit_fix.log | outputs/query_refinement/gqa_discovery_manifest_audit_v1.json |
| query_refinement_preflight_v1 | running | Running GPU job query_refinement_preflight_r2_20260807 | runs/query_refinement_preflight_v1/retry2.log | outputs/query_refinement/preflight_v1/summary.json |
| stage_a_chunked_eager_probe_1024_20260804 | running | Running GPU job stage_a_chunked_eager_probe_1024_20260804 | runs/stage_a_chunked_eager_probe_1024_20260804/slurm.log | outputs/stage_a_chunked_eager_probe/stage_a_summary.json |
| stage_a_chunked_eager_probe_20260804 | running | Running GPU job stage_a_chunked_eager_probe_20260804 | runs/stage_a_chunked_eager_probe_20260804/slurm.log | outputs/stage_a_chunked_eager_probe/stage_a_summary.json |
| stage_a_chunked_eager_probe_256_20260804 | running | Running GPU job stage_a_chunked_eager_probe_256_20260804 | runs/stage_a_chunked_eager_probe_256_20260804/slurm.log | outputs/stage_a_chunked_eager_probe/stage_a_summary.json |
| stage_a_chunked_stock_equivalence_20260804 | running | Running GPU job stage_a_chunked_stock_equivalence_20260804 | runs/stage_a_chunked_stock_equivalence_20260804/slurm.log | outputs/stage_a_chunked_stock_equivalence/chunked_eager_equivalence.json |
| stage_a_chunked_stock_equivalence_boundary_20260804 | running | Running GPU job stage_a_chunked_stock_equivalence_boundary_20260804 | runs/stage_a_chunked_stock_equivalence_boundary_20260804/slurm.log | outputs/stage_a_chunked_stock_equivalence_boundary/chunked_eager_equivalence.json |
| stage_a_env_20260804 | running | Running GPU job stage_a_env_20260804 | runs/stage_a_env_20260804/slurm.log | outputs/stage_a/env_ready.json |
| stage_a_env_localcache_20260804 | running | Running GPU job stage_a_env_localcache_20260804 | runs/stage_a_env_localcache_20260804/slurm.log | outputs/stage_a/env_ready.json |
| stage_a_probe_20260804 | running | Running GPU job stage_a_probe_20260804 | runs/stage_a_probe_20260804/slurm.log | outputs/stage_a_probe/stage_a_summary.json |
| stage_a_probe_cublas_20260804 | running | Running GPU job stage_a_probe_cublas_20260804 | runs/stage_a_probe_cublas_20260804/slurm.log | outputs/stage_a_probe_cublas/stage_a_summary.json |
| stage_a_probe_readfix_20260804 | running | Running GPU job stage_a_probe_readfix_20260804 | runs/stage_a_probe_readfix_20260804/slurm.log | outputs/stage_a_probe_readfix/stage_a_summary.json |
| stage_a_sdpa_reference_probe_20260804 | running | Running GPU job stage_a_sdpa_reference_probe_20260804 | runs/stage_a_sdpa_reference_probe_20260804/slurm.log | outputs/stage_a_sdpa_reference_probe/stage_a_summary.json |
| stage_a_sdpa_reference_probe_causal_20260804 | running | Running GPU job stage_a_sdpa_reference_probe_causal_20260804 | runs/stage_a_sdpa_reference_probe_causal_20260804/slurm.log | outputs/stage_a_sdpa_reference_probe/stage_a_summary.json |
| stage_a_sdpa_reference_probe_chunked_20260804 | running | Running GPU job stage_a_sdpa_reference_probe_chunked_20260804 | runs/stage_a_sdpa_reference_probe_chunked_20260804/slurm.log | outputs/stage_a_sdpa_reference_probe/stage_a_summary.json |
| stage_a_stock_eager_validity_23_20260804 | running | Running GPU job stage_a_stock_eager_validity_23_20260804 | runs/stage_a_stock_eager_validity_23_20260804/slurm.log | outputs/stage_a/stage_a_summary.json |
| stage_a_validity_20260804 | running | Running GPU job stage_a_validity_20260804 | runs/stage_a_validity_20260804/slurm.log | outputs/stage_a/stage_a_summary.json |
| stage_a_validity_final_20260804 | running | Running GPU job stage_a_validity_final_20260804 | runs/stage_a_validity_final_20260804/slurm.log | outputs/stage_a/stage_a_summary.json |
| stage_a_validity_memory_bounded_20260804 | running | Running GPU job stage_a_validity_memory_bounded_20260804 | runs/stage_a_validity_memory_bounded_20260804/slurm.log | outputs/stage_a/stage_a_summary.json |
| stage_a_validity_vision_sdpa_20260804 | running | Running GPU job stage_a_validity_vision_sdpa_20260804 | runs/stage_a_validity_vision_sdpa_20260804/slurm.log | outputs/stage_a/stage_a_summary.json |
| stage_b_c_archive_outcome_b_20260805 | running | Running GPU job stage_b_c_archive_outcome_b_retry_20260805 | runs/stage_b_c_archive_outcome_b_retry_20260805/slurm.log | archives/stage_b_stage_c_frozen_outcome_b_v1/archive_summary_v1.json |
| stage_b_reference_analysis_20260804 | running | Running GPU job stage_b_reference_analysis_20260804 | runs/stage_b_reference_analysis_20260804/slurm.log | outputs/stage_b/analysis_v1/analysis_manifest.json |
| stage_b_reference_full_20260804 | running | Running GPU job stage_b_reference_full_20260804 | runs/stage_b_reference_full_20260804/slurm.log | outputs/stage_b/stage_b_results_v1.jsonl |
| stage_b_reference_full_v2_20260804 | running | Running GPU job stage_b_reference_full_v2_20260804 | runs/stage_b_reference_full_v2_20260804/slurm.log | outputs/stage_b/stage_b_results_v1.jsonl |
| stage_b_reference_validity_20260804 | running | Running GPU job stage_b_reference_validity_20260804 | runs/stage_b_reference_validity_20260804/slurm.log | outputs/stage_b_validity/stage_b_validity_summary.json |
| stage_b_reference_validity_v2_20260804 | running | Running GPU job stage_b_reference_validity_v2_20260804 | runs/stage_b_reference_validity_v2_20260804/slurm.log | outputs/stage_b_validity_v2/stage_b_validity_summary.json |
| stage_b_reference_validity_v3_20260804 | running | Running GPU job stage_b_reference_validity_v3_20260804 | runs/stage_b_reference_validity_v3_20260804/slurm.log | outputs/stage_b_validity_v3/stage_b_validity_summary.json |
| stage_b_reference_validity_v4_20260804 | running | Running GPU job stage_b_reference_validity_v4_20260804 | runs/stage_b_reference_validity_v4_20260804/slurm.log | outputs/stage_b_validity_v4/stage_b_validity_summary.json |
| stage_c_datasets_env_20260805 | running | Running GPU job stage_c_datasets_env_20260805 | runs/stage_c_datasets_env_20260805/slurm.log | workspace/env_state.md |
| stage_c_donor_coverage_audit_v1_20260805 | running | Running GPU job stage_c_donor_audit_s07_20260805 | runs/stage_c_donor_coverage_audit_v1_20260805/shard_07/slurm.log | outputs/stage_c/preflight/donor_coverage_v1/shards/shard_07/geometry.jsonl |
| stage_c_entry_gate_20260805 | running | Running GPU job stage_c_entry_gate_20260805 | runs/stage_c_entry_gate_20260805/slurm.log | outputs/stage_c/nulls/null_calibration_and_smoke_v1.json |
| stage_c_entry_gate_v2_20260805 | running | Running GPU job stage_c_entry_gate_v2_20260805 | runs/stage_c_entry_gate_v2_20260805/slurm.log | outputs/stage_c/nulls/null_calibration_and_smoke_v1.json |
| stage_c_entry_gate_v3_20260805 | running | Running GPU job stage_c_entry_gate_v3_20260805 | runs/stage_c_entry_gate_v3_20260805/slurm.log | outputs/stage_c/nulls/null_calibration_and_smoke_v1.json |
| stage_c_full_preflight_20260805 | running | Running GPU job stage_c_full_preflight_20260805 | runs/stage_c_full_preflight_20260805/slurm-%j.out | outputs/stage_c/stage_c_execution_freeze_v1.json |
| stage_c_full_v1_20260805 | running | Running GPU job stage_c_full_s7_20260805 | runs/stage_c_full_v1_20260805/shard_7/slurm-%j.out | outputs/stage_c/shards_v1/shard_07/results.jsonl |
| stage_c_full_v2_20260805 | running | Running GPU job stage_c_analysis_v1_retry_20260805 | runs/stage_c_full_v2_20260805/analysis_retry/slurm.log | outputs/stage_c/analysis_v1/final_decision.json |
| stage_c_manifest_freeze_20260805 | running | Running GPU job stage_c_manifest_freeze_20260805 | runs/stage_c_manifest_freeze_20260805/slurm.log | outputs/stage_c/manifest/stage_c_manifest_v1.jsonl |
| stage_c_manifest_freeze_v2_20260805 | running | Running GPU job stage_c_manifest_freeze_v2_20260805 | runs/stage_c_manifest_freeze_v2_20260805/slurm.log | outputs/stage_c/manifest/stage_c_manifest_v1.jsonl |
| stage_c_manifest_freeze_v3_20260805 | running | Running GPU job stage_c_manifest_freeze_v3_20260805 | runs/stage_c_manifest_freeze_v3_20260805/slurm.log | outputs/stage_c/manifest/stage_c_manifest_v1.jsonl |
| stage_c_prefix_preflight_v1_20260805 | running | Running GPU job stage_c_prefix_preflight_merge_20260805 | runs/stage_c_prefix_preflight_merge_20260805/slurm-%j.out | outputs/stage_c/prefix_preflight_v1/summary.json |
| stage_c_prefix_score_preflight_v1_20260805 | running | Running GPU job stage_c_prefix_score_preflight_s7_20260805 | runs/stage_c_prefix_score_preflight_v1_20260805/shard_7/slurm-%j.out | outputs/stage_c/prefix_preflight_v1/shards/shard_07/results.jsonl |
| stage_c_prefix_static_preflight_v2_20260805 | running | Running GPU job stage_c_prefix_static_preflight_v2_20260805 | runs/stage_c_prefix_static_preflight_v2_20260805/slurm-%j.out | outputs/stage_c/prefix_preflight_v1/static_preflight.json |
| stage_c_textvqa_validation_download_20260805 | running | Running GPU job stage_c_textvqa_validation_download_20260805 | runs/stage_c_textvqa_validation_download_20260805/slurm.log | outputs/datasets/lmms_lab_textvqa_validation/download_report.json |
| v3_confirmation_preflight_v1 | running | Running GPU job v3_query_invariance_diag_20260806 | runs/v3_query_invariance_diag_20260806/slurm.log | outputs/v3_preflight/query_invariance_equal_length_diagnostic.json |
| v3_gqa_scenegraphs_download_20260806 | running | Running GPU job v3_gqa_scenegraphs_download_20260806 | runs/v3_gqa_scenegraphs_download_20260806/slurm.log | /data/dataset/GQA/sceneGraphs_v1.1/train_sceneGraphs.json |
| v3_grounding_audit_20260806 | running | Running GPU job v3_grounding_audit_20260806 | runs/v3_grounding_audit_20260806/slurm.log | outputs/v3_preflight/grounding_eligibility_audit_v1.json |
| v3_grounding_audit_r2_20260806 | running | Running GPU job v3_grounding_audit_r2_20260806 | runs/v3_grounding_audit_r2_20260806/slurm.log | outputs/v3_preflight/grounding_eligibility_audit_v1.json |
| v3_null_geometry_merge_20260806 | running | Running GPU job v3_null_geometry_merge_20260806 | runs/v3_null_geometry_merge_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/manifest.json |
| v3_null_geometry_merge_r2_20260806 | running | Running GPU job v3_null_geometry_merge_r2_20260806 | runs/v3_null_geometry_merge_r2_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/manifest.json |
| v3_null_geometry_s00_20260806 | running | Running GPU job v3_null_geometry_s00_20260806 | runs/v3_null_geometry_s00_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_00/shard_manifest.json |
| v3_null_geometry_s01_20260806 | running | Running GPU job v3_null_geometry_s01_20260806 | runs/v3_null_geometry_s01_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_01/shard_manifest.json |
| v3_null_geometry_s02_20260806 | running | Running GPU job v3_null_geometry_s02_20260806 | runs/v3_null_geometry_s02_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_02/shard_manifest.json |
| v3_null_geometry_s03_20260806 | running | Running GPU job v3_null_geometry_s03_20260806 | runs/v3_null_geometry_s03_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_03/shard_manifest.json |
| v3_null_geometry_s03_r3_20260806 | running | Running GPU job v3_null_geometry_s03_r3_20260806 | runs/v3_null_geometry_s03_r3_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_03/shard_manifest.json |
| v3_null_geometry_s04_20260806 | running | Running GPU job v3_null_geometry_s04_20260806 | runs/v3_null_geometry_s04_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_04/shard_manifest.json |
| v3_null_geometry_s04_r3_20260806 | running | Running GPU job v3_null_geometry_s04_r3_20260806 | runs/v3_null_geometry_s04_r3_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_04/shard_manifest.json |
| v3_null_geometry_s05_20260806 | running | Running GPU job v3_null_geometry_s05_20260806 | runs/v3_null_geometry_s05_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_05/shard_manifest.json |
| v3_null_geometry_s05_r3_20260806 | running | Running GPU job v3_null_geometry_s05_r3_20260806 | runs/v3_null_geometry_s05_r3_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_05/shard_manifest.json |
| v3_null_geometry_s06_20260806 | running | Running GPU job v3_null_geometry_s06_20260806 | runs/v3_null_geometry_s06_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_06/shard_manifest.json |
| v3_null_geometry_s06_r3_20260806 | running | Running GPU job v3_null_geometry_s06_r3_20260806 | runs/v3_null_geometry_s06_r3_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_06/shard_manifest.json |
| v3_null_geometry_s07_20260806 | running | Running GPU job v3_null_geometry_s07_20260806 | runs/v3_null_geometry_s07_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_07/shard_manifest.json |
| v3_null_geometry_s07_r3_20260806 | running | Running GPU job v3_null_geometry_s07_r3_20260806 | runs/v3_null_geometry_s07_r3_20260806/slurm.log | artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_07/shard_manifest.json |
| v3_null_geometry_smoke_20260806 | running | Running GPU job v3_null_geometry_smoke_20260806 | runs/v3_null_geometry_smoke_20260806/slurm.log | outputs/v3_preflight/null_geometry_smoke_v1/shards/shard_00/shard_manifest.json |
| v3_null_geometry_smoke_r2_20260806 | running | Running GPU job v3_null_geometry_smoke_r2_20260806 | runs/v3_null_geometry_smoke_r2_20260806/slurm.log | outputs/v3_preflight/null_geometry_smoke_v2/shards/shard_00/shard_manifest.json |
| v3_null_geometry_smoke_r3_20260806 | running | Running GPU job v3_null_geometry_smoke_r3_20260806 | runs/v3_null_geometry_smoke_r3_20260806/slurm.log | outputs/v3_preflight/null_geometry_smoke_v3/shards/shard_00/shard_manifest.json |
| v3_null_geometry_smoke_r4_20260806 | running | Running GPU job v3_null_geometry_smoke_r4_20260806 | runs/v3_null_geometry_smoke_r4_20260806/slurm.log | outputs/v3_preflight/null_geometry_smoke_v4/shards/shard_00/shard_manifest.json |
| v3_null_model_fit_20260806 | running | Running GPU job v3_null_model_fit_20260806 | runs/v3_null_model_fit_20260806/slurm.log | artifacts/v3_null_calibration/joint_covariance_model_v1/manifest.json |
| v3_null_model_fit_r2_20260806 | running | Running GPU job v3_null_model_fit_r2_20260806 | runs/v3_null_model_fit_r2_20260806/slurm.log | artifacts/v3_null_calibration/joint_covariance_model_v1/manifest.json |
| v3_null_redesign_c_rank1024_v1 | running | Running GPU job v3_null_redesign_c_rank_20260807 | runs/v3_null_redesign_c_rank_20260807/slurm.log | outputs/v3_null_redesign/covariance_representation_c_rank_extension.json |
| v3_null_redesign_covariance_comparison_v1 | running | Running GPU job v3_null_redesign_covariance_20260807 | runs/v3_null_redesign_covariance_20260807/slurm.log | outputs/v3_null_redesign/covariance_representation_comparison.json |
| v3_null_redesign_delta_merge_v1 | running | Running GPU job v3_null_redesign_delta_merge_20260807 | runs/v3_null_redesign_delta_merge_20260807/slurm.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/manifest.json |
| v3_null_redesign_delta_s00_v1 | running | Running GPU job v3_null_redesign_delta_s00_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_00.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_00/shard_manifest.json |
| v3_null_redesign_delta_s01_v1 | running | Running GPU job v3_null_redesign_delta_s01_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_01.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_01/shard_manifest.json |
| v3_null_redesign_delta_s02_v1 | running | Running GPU job v3_null_redesign_delta_s02_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_02.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_02/shard_manifest.json |
| v3_null_redesign_delta_s03_v1 | running | Running GPU job v3_null_redesign_delta_s03_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_03.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_03/shard_manifest.json |
| v3_null_redesign_delta_s04_v1 | running | Running GPU job v3_null_redesign_delta_s04_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_04.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_04/shard_manifest.json |
| v3_null_redesign_delta_s05_v1 | running | Running GPU job v3_null_redesign_delta_s05_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_05.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_05/shard_manifest.json |
| v3_null_redesign_delta_s06_v1 | running | Running GPU job v3_null_redesign_delta_s06_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_06.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_06/shard_manifest.json |
| v3_null_redesign_delta_s07_v1 | running | Running GPU job v3_null_redesign_delta_s07_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_07.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_07/shard_manifest.json |
| v3_null_redesign_delta_s08_v1 | running | Running GPU job v3_null_redesign_delta_s08_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_08.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_08/shard_manifest.json |
| v3_null_redesign_delta_s09_v1 | running | Running GPU job v3_null_redesign_delta_s09_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_09.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_09/shard_manifest.json |
| v3_null_redesign_delta_s10_v1 | running | Running GPU job v3_null_redesign_delta_s10_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_10.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_10/shard_manifest.json |
| v3_null_redesign_delta_s11_v1 | running | Running GPU job v3_null_redesign_delta_s11_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_11.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_11/shard_manifest.json |
| v3_null_redesign_delta_s12_v1 | running | Running GPU job v3_null_redesign_delta_s12_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_12.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_12/shard_manifest.json |
| v3_null_redesign_delta_s13_v1 | running | Running GPU job v3_null_redesign_delta_s13_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_13.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_13/shard_manifest.json |
| v3_null_redesign_delta_s14_v1 | running | Running GPU job v3_null_redesign_delta_s14_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_14.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_14/shard_manifest.json |
| v3_null_redesign_delta_s15_v1 | running | Running GPU job v3_null_redesign_delta_s15_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_15.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_15/shard_manifest.json |
| v3_null_redesign_delta_s16_v1 | running | Running GPU job v3_null_redesign_delta_s16_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_16.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_16/shard_manifest.json |
| v3_null_redesign_delta_s17_v1 | running | Running GPU job v3_null_redesign_delta_s17_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_17.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_17/shard_manifest.json |
| v3_null_redesign_delta_s18_v1 | running | Running GPU job v3_null_redesign_delta_s18_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_18.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_18/shard_manifest.json |
| v3_null_redesign_delta_s19_v1 | running | Running GPU job v3_null_redesign_delta_s19_20260807 | runs/v3_null_redesign_geometry_delta_20260807/shard_19.log | artifacts/v3_null_redesign/read_write_geometry_delta_v2/shards/shard_19/shard_manifest.json |
| v3_null_redesign_donor_audit_v1 | running | Running GPU job v3_null_redesign_donors_20260807 | runs/v3_null_redesign_donors_20260807/slurm.log | outputs/v3_null_redesign/donor_coverage.json |
| v3_null_redesign_donor_audit_v2 | running | Running GPU job v3_null_redesign_donors_v2_20260807 | runs/v3_null_redesign_donors_v2_20260807/slurm.log | outputs/v3_null_redesign/donor_coverage_v2.json |
| v3_null_redesign_geometry_merge_v1 | running | Running GPU job v3_null_redesign_merge_20260807 | runs/v3_null_redesign_merge_20260807/slurm.log | artifacts/v3_null_redesign/read_write_geometry_v2/manifest.json |
| v3_null_redesign_geometry_merge_v1_r2 | running | Running GPU job v3_null_redesign_merge_r2_20260807 | runs/v3_null_redesign_merge_r2_20260807/slurm.log | artifacts/v3_null_redesign/read_write_geometry_v2/manifest.json |
| v3_null_redesign_geometry_s00_v1 | running | Running GPU job v3_null_redesign_geom_s00_20260807 | runs/v3_null_redesign_geometry_20260807/shard_00.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_00/shard_manifest.json |
| v3_null_redesign_geometry_s00_v1_r2 | running | Running GPU job v3_null_redesign_geom_s00_r2_20260807 | runs/v3_null_redesign_geometry_r2_20260807/shard_00.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_00/shard_manifest.json |
| v3_null_redesign_geometry_s00_v1_r3 | running | Running GPU job v3_null_redesign_geom_s00_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_00.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_00/shard_manifest.json |
| v3_null_redesign_geometry_s01_v1 | running | Running GPU job v3_null_redesign_geom_s01_20260807 | runs/v3_null_redesign_geometry_20260807/shard_01.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_01/shard_manifest.json |
| v3_null_redesign_geometry_s01_v1_r2 | running | Running GPU job v3_null_redesign_geom_s01_r2_20260807 | runs/v3_null_redesign_geometry_r2_20260807/shard_01.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_01/shard_manifest.json |
| v3_null_redesign_geometry_s01_v1_r3 | running | Running GPU job v3_null_redesign_geom_s01_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_01.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_01/shard_manifest.json |
| v3_null_redesign_geometry_s02_v1 | running | Running GPU job v3_null_redesign_geom_s02_20260807 | runs/v3_null_redesign_geometry_20260807/shard_02.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_02/shard_manifest.json |
| v3_null_redesign_geometry_s02_v1_r2 | running | Running GPU job v3_null_redesign_geom_s02_r2_20260807 | runs/v3_null_redesign_geometry_r2_20260807/shard_02.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_02/shard_manifest.json |
| v3_null_redesign_geometry_s02_v1_r3 | running | Running GPU job v3_null_redesign_geom_s02_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_02.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_02/shard_manifest.json |
| v3_null_redesign_geometry_s03_v1 | running | Running GPU job v3_null_redesign_geom_s03_20260807 | runs/v3_null_redesign_geometry_20260807/shard_03.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_03/shard_manifest.json |
| v3_null_redesign_geometry_s03_v1_r2 | running | Running GPU job v3_null_redesign_geom_s03_r2_20260807 | runs/v3_null_redesign_geometry_r2_20260807/shard_03.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_03/shard_manifest.json |
| v3_null_redesign_geometry_s03_v1_r3 | running | Running GPU job v3_null_redesign_geom_s03_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_03.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_03/shard_manifest.json |
| v3_null_redesign_geometry_s04_v1 | running | Running GPU job v3_null_redesign_geom_s04_20260807 | runs/v3_null_redesign_geometry_20260807/shard_04.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_04/shard_manifest.json |
| v3_null_redesign_geometry_s04_v1_r2 | running | Running GPU job v3_null_redesign_geom_s04_r2_20260807 | runs/v3_null_redesign_geometry_r2_20260807/shard_04.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_04/shard_manifest.json |
| v3_null_redesign_geometry_s04_v1_r3 | running | Running GPU job v3_null_redesign_geom_s04_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_04.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_04/shard_manifest.json |
| v3_null_redesign_geometry_s05_v1 | running | Running GPU job v3_null_redesign_geom_s05_20260807 | runs/v3_null_redesign_geometry_20260807/shard_05.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_05/shard_manifest.json |
| v3_null_redesign_geometry_s05_v1_r2 | running | Running GPU job v3_null_redesign_geom_s05_r2_20260807 | runs/v3_null_redesign_geometry_r2_20260807/shard_05.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_05/shard_manifest.json |
| v3_null_redesign_geometry_s05_v1_r3 | running | Running GPU job v3_null_redesign_geom_s05_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_05.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_05/shard_manifest.json |
| v3_null_redesign_geometry_s06_v1 | running | Running GPU job v3_null_redesign_geom_s06_20260807 | runs/v3_null_redesign_geometry_20260807/shard_06.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_06/shard_manifest.json |
| v3_null_redesign_geometry_s06_v1_r2 | running | Running GPU job v3_null_redesign_geom_s06_r2_20260807 | runs/v3_null_redesign_geometry_r2_20260807/shard_06.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_06/shard_manifest.json |
| v3_null_redesign_geometry_s06_v1_r3 | running | Running GPU job v3_null_redesign_geom_s06_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_06.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_06/shard_manifest.json |
| v3_null_redesign_geometry_s07_v1 | running | Running GPU job v3_null_redesign_geom_s07_20260807 | runs/v3_null_redesign_geometry_20260807/shard_07.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_07/shard_manifest.json |
| v3_null_redesign_geometry_s07_v1_r2 | running | Running GPU job v3_null_redesign_geom_s07_r2_20260807 | runs/v3_null_redesign_geometry_r2_20260807/shard_07.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_07/shard_manifest.json |
| v3_null_redesign_geometry_s07_v1_r3 | running | Running GPU job v3_null_redesign_geom_s07_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_07.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_07/shard_manifest.json |
| v3_null_redesign_geometry_s08_v1_r3 | running | Running GPU job v3_null_redesign_geom_s08_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_08.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_08/shard_manifest.json |
| v3_null_redesign_geometry_s09_v1_r3 | running | Running GPU job v3_null_redesign_geom_s09_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_09.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_09/shard_manifest.json |
| v3_null_redesign_geometry_s10_v1_r3 | running | Running GPU job v3_null_redesign_geom_s10_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_10.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_10/shard_manifest.json |
| v3_null_redesign_geometry_s11_v1_r3 | running | Running GPU job v3_null_redesign_geom_s11_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_11.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_11/shard_manifest.json |
| v3_null_redesign_geometry_s12_v1_r3 | running | Running GPU job v3_null_redesign_geom_s12_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_12.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_12/shard_manifest.json |
| v3_null_redesign_geometry_s13_v1_r3 | running | Running GPU job v3_null_redesign_geom_s13_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_13.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_13/shard_manifest.json |
| v3_null_redesign_geometry_s14_v1_r3 | running | Running GPU job v3_null_redesign_geom_s14_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_14.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_14/shard_manifest.json |
| v3_null_redesign_geometry_s15_v1_r3 | running | Running GPU job v3_null_redesign_geom_s15_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_15.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_15/shard_manifest.json |
| v3_null_redesign_geometry_s16_v1_r3 | running | Running GPU job v3_null_redesign_geom_s16_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_16.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_16/shard_manifest.json |
| v3_null_redesign_geometry_s17_v1_r3 | running | Running GPU job v3_null_redesign_geom_s17_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_17.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_17/shard_manifest.json |
| v3_null_redesign_geometry_s18_v1_r3 | running | Running GPU job v3_null_redesign_geom_s18_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_18.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_18/shard_manifest.json |
| v3_null_redesign_geometry_s19_v1_r3 | running | Running GPU job v3_null_redesign_geom_s19_r3_20260807 | runs/v3_null_redesign_geometry_r3_20260807/shard_19.log | artifacts/v3_null_redesign/read_write_geometry_v2/shards/shard_19/shard_manifest.json |
| v3_null_redesign_pool_v1 | running | Running GPU job v3_null_redesign_pool_20260807 | runs/v3_null_redesign_pool_20260807/slurm.log | outputs/v3_null_redesign/calibration_pool_manifest.json |
| v3_null_redesign_pool_v2 | running | Running GPU job v3_null_redesign_pool_v2_20260807 | runs/v3_null_redesign_pool_v2_20260807/slurm.log | outputs/v3_null_redesign/calibration_pool_manifest_v2.json |
| v3_preflight_pool_audit_v1 | running | Running GPU job v3_preflight_pool_audit_20260806_r2 | runs/v3_preflight_pool_audit_20260806_r2/slurm.log | outputs/v3_preflight/candidate_pool_audit.json |
| v3_stage_b_reanalysis_v1 | running | Running GPU job v3_stage_b_reanalysis_20260806_r4 | runs/v3_stage_b_reanalysis_20260806_r4/slurm.log | outputs/v3_discovery/analysis_manifest.json |
| v3_textocr_annotations_download_20260806 | running | Running GPU job v3_textocr_annotations_download_20260806 | runs/v3_textocr_annotations_download_20260806/slurm.log | /data/dataset/TextOCR/annotations_v0.1/TextOCR_0.1_train.json |
| v4_cost_utility_frontier_20260807 | running | Running GPU job v4_cost_utility_frontier_20260807 | runs/v4_cost_utility_frontier_20260807/slurm.log | outputs/v4_discovery/cost_utility_frontier_summary_v1.json |
| v4_cost_utility_frontier_r2_20260807 | running | Running GPU job v4_cost_utility_frontier_r2_20260807 | runs/v4_cost_utility_frontier_r2_20260807/slurm.log | outputs/v4_discovery/cost_utility_frontier_summary_v1.json |
| v4_cost_utility_frontier_r3_20260807 | running | Running GPU job v4_cost_utility_frontier_r3_20260807 | runs/v4_cost_utility_frontier_r3_20260807/slurm.log | outputs/v4_discovery/cost_utility_frontier_summary_v1.json |
| v4_discovery_analysis_20260807 | running | Running GPU job v4_discovery_analysis_20260807 | runs/v4_discovery_analysis_20260807/slurm.log | outputs/v4_discovery/analysis_v1/analysis_manifest.json |
| v4_discovery_analysis_r2_20260807 | running | Running GPU job v4_discovery_analysis_r2_20260807 | runs/v4_discovery_analysis_r2_20260807/slurm.log | outputs/v4_discovery/analysis_v1/analysis_manifest.json |
| v4_discovery_analysis_r3_20260807 | running | Running GPU job v4_discovery_analysis_r3_20260807 | runs/v4_discovery_analysis_r3_20260807/slurm.log | outputs/v4_discovery/analysis_v1/analysis_manifest.json |
| v4_discovery_analysis_r4_20260807 | running | Running GPU job v4_discovery_analysis_r4_20260807 | runs/v4_discovery_analysis_r4_20260807/slurm.log | outputs/v4_discovery/analysis_v1/analysis_manifest.json |
| v4_discovery_s00_20260807 | running | Running GPU job v4_discovery_s00_20260807 | runs/v4_discovery_s00_20260807/slurm.log | outputs/v4_discovery/shards/shard_00/completion.json |
| v4_discovery_s01_20260807 | running | Running GPU job v4_discovery_s01_20260807 | runs/v4_discovery_s01_20260807/slurm.log | outputs/v4_discovery/shards/shard_01/completion.json |
| v4_discovery_s02_20260807 | running | Running GPU job v4_discovery_s02_20260807 | runs/v4_discovery_s02_20260807/slurm.log | outputs/v4_discovery/shards/shard_02/completion.json |
| v4_discovery_s02_r3_20260807 | running | Running GPU job v4_discovery_s02_r3_20260807 | runs/v4_discovery_s02_r3_20260807/slurm.log | outputs/v4_discovery/shards/shard_02/completion.json |
| v4_discovery_s03_20260807 | running | Running GPU job v4_discovery_s03_20260807 | runs/v4_discovery_s03_20260807/slurm.log | outputs/v4_discovery/shards/shard_03/completion.json |
| v4_manifest_freeze_20260807 | running | Running GPU job v4_manifest_freeze_20260807 | runs/v4_manifest_freeze_20260807/slurm.log | outputs/v4_discovery/manifest/v4_gqa_discovery_manifest_audit_v1.json |
| v4_preflight_20260807 | running | Running GPU job v4_preflight_20260807 | runs/v4_preflight_20260807/slurm.log | outputs/v4_discovery/preflight/v4_common_padding_preflight_v1.json |
| v4_preflight_r2_20260807 | running | Running GPU job v4_preflight_r2_20260807 | runs/v4_preflight_r2_20260807/slurm.log | outputs/v4_discovery/preflight/v4_common_padding_preflight_v1.json |
| wemath20_cap400_stage_v1 | running | Running GPU job wemath20_cap400_stage_r2_20260813 | runs/label_regeneration/wemath2pro_cap400_stage_r2.log | outputs/label_regeneration/wemath2pro_cap400_v1/cap400_resume_audit_v1.json |
| wemath20_cap400_stage_v2 | running | Running GPU job wemath20_cap400_stage_v2_20260813 | runs/label_regeneration/wemath2pro_cap400_stage_v2.log | outputs/label_regeneration/wemath2pro_cap400_v2/cap400_resume_audit_v1.json |
| wemath20_extension_yield_v1 | running | Running GPU job wemath20_extension_yield_r2_20260812 | runs/label_regeneration/wemath2pro_extension_yield_r2.log | outputs/label_regeneration/wemath2pro_v1/extension_yield_snapshot_v1.json |
| wemath20_mathruler_env_v1 | running | Running GPU job wemath20_mathruler_env_20260811 | runs/env/wemath20_mathruler_env_v1.log | workspace/env_state.md |
| wemath20_pro_download_v1 | running | Running GPU job wemath20_pro_download_20260811 | runs/dataset_downloads/wemath20_pro_v1.log | outputs/dataset_downloads/wemath20_pro_v1.json |
| wemath20_pro_manifest_v1 | running | Running GPU job wemath20_pro_manifest_20260811 | runs/label_regeneration/wemath2pro_manifest_v1.log | outputs/label_regeneration/wemath2pro_v1/manifest/manifest_summary_v1.json |
| wemath20_pro_manifest_v2 | running | Running GPU job wemath20_pro_manifest_v2_20260811 | runs/label_regeneration/wemath2pro_manifest_v2.log | outputs/label_regeneration/wemath2pro_v1/manifest/manifest_summary_v1.json |
| wemath20_pro_mcts_cap400_v1 | running | Running GPU job wemath20_pro_mcts_cap400_7gpu_r3_20260813 | runs/label_regeneration/wemath2pro_cap400_r3.log | outputs/label_regeneration/wemath2pro_cap400_v2/raw_route_cache |
| wemath20_pro_mcts_v1 | running | Running GPU job wemath20_pro_mcts_8gpu_r3_20260811 | runs/label_regeneration/wemath2pro_mcts_r3.log | outputs/label_regeneration/wemath2pro_v1/raw_route_cache |
| wemath20_pro_mcts_v2_timeout_repair | running | Running GPU job wemath20_pro_mcts_6gpu_resume_v2_20260812 | runs/label_regeneration/wemath2pro_mcts_r5.log | outputs/label_regeneration/wemath2pro_v1/raw_route_cache |
| wemath20_resume_audit_v2 | running | Running GPU job wemath20_resume_audit_v2_r3_20260812 | runs/label_regeneration/wemath2pro_resume_audit_v2_r3.log | outputs/label_regeneration/wemath2pro_v1/resume_compatibility_audit_v2.json |
| wemath20_standard_download_v1 | running | Running GPU job wemath20_standard_download_20260811 | runs/dataset_downloads/wemath20_standard_v1.log | outputs/dataset_downloads/wemath20_standard_v1.json |
| wemath2pro-greedy-phase1-node06-v1 | running | Running GPU job wemath-greedy-phase1-node06-v1 | runs/wemath_greedy_recovery/phase1_node06.log | outputs/label_regeneration/wemath2pro_greedy_recovery_v1/phase1/shard_000_of_004/summary.json |
| wemath2pro-greedy-phase1-node07-v1 | running | Running GPU job wemath-greedy-phase1-node07-v1 | runs/wemath_greedy_recovery/phase1_node07.log | outputs/label_regeneration/wemath2pro_greedy_recovery_v1/phase1/shard_002_of_004/summary.json |
| wemath2pro-greedy-recovery-preflight-v1 | running | Running GPU job wemath-greedy-preflight-v1 | runs/wemath_greedy_recovery/preflight.log | outputs/label_regeneration/wemath2pro_greedy_recovery_v1/preflight/preflight_report_v1.json |
| wemath2pro-greedy-recovery-preflight-v2 | running | Running GPU job wemath-greedy-preflight-v2 | runs/wemath_greedy_recovery/preflight_r2.log | outputs/label_regeneration/wemath2pro_greedy_recovery_v1/preflight/preflight_report_v1.json |
| wemath2pro-greedy-recovery-v1 | running | Running GPU job wemath-greedy-manifest-v1 | runs/wemath_greedy_recovery/manifest.log | outputs/label_regeneration/wemath2pro_greedy_recovery_v1/manifest/recovery_manifest_audit_v1.json |
| wemath2pro-label-analysis-v1 | running | Running GPU job wemath2pro-label-analysis-v2 | runs/wemath2pro_label_analysis_v1/slurm_v2.log | outputs/wemath2pro_mcts_label_analysis_v1/analysis_manifest.json |
