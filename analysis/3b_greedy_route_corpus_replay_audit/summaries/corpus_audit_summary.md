Canonical 3B corpus census; GPU replay pending.

{
  "unique_samples": 10000,
  "unique_image_groups": 9273,
  "unique_routes": 3907717,
  "raw_records": 3907717,
  "duplicate_records": 0,
  "duplicate_canonical_routes": 0,
  "routes_per_sample": {
    "mean": 390.7717,
    "std": 18.69644830201715,
    "min": 218,
    "p10": 355.0,
    "p25": 381.0,
    "median": 400.0,
    "p75": 405.0,
    "p90": 407.0,
    "p95": 407.0,
    "p99": 409.0,
    "max": 409
  },
  "histogram": {
    "1": 0,
    "2-4": 0,
    "5-9": 0,
    "10-19": 0,
    "20-49": 0,
    "50+": 10000
  },
  "saved_correct": 1880345,
  "saved_wrong": 2027372,
  "saved_unknown": 0,
  "sample_composition": {
    "MIXED_CW": 8481,
    "ALL_W": 1486,
    "ALL_C": 33
  },
  "geometry_eligibility": {
    "C_W": 8481,
    "C_C": 8496,
    "W_W": 9933,
    "C_C_plus_C_W": 8463,
    "W_W_plus_C_W": 8447,
    "full_pairwise": 8429
  },
  "dense_audit": {
    "exactly_one": 10000,
    "zero": 0,
    "multiple": 0
  },
  "phase_counts": {
    "phase1": 3559195,
    "phase2": 348522
  },
  "rank_route_loads": [
    488464,
    488353,
    488484,
    488484,
    488484,
    488484,
    488483,
    488481
  ],
  "macro_saved_correct_fraction": 0.4625506927596368
}

Dataset-specific counts and distributions: census/dataset_breakdown.csv. Canonical labels remain historical.
