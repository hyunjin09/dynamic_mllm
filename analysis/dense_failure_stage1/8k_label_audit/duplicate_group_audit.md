# Stage-1 Dense-Failure Duplicate and Group Audit

## Physical 8K image audit

All 8,000 image files were SHA-256 hashed on this server.

| Metric | Count |
|---|---:|
| Physical image records | 8,000 |
| Unique content SHA-256 groups | 7,477 |
| Content groups with more than one record | 513 |
| Records in repeated-content groups | 1,036 |
| Duplicate occurrences beyond the first | 523 |
| Largest content group | 4 |
| Cross-dataset content groups | 0 |

These repeated image payloads make record-random splitting unsafe.

## Available 6,917-row metadata audit

| Metric | Count |
|---|---:|
| Unique UIDs | 6,917 |
| Duplicate UID rows | 0 |
| Unique source image-group IDs | 6,574 |
| Image groups with more than one question | 340 |
| Rows in multi-question image groups | 683 |
| Maximum questions per source image group | 3 |
| Repeated exact question-text values | 250 |
| Rows carrying repeated exact question text | 747 |

The complete question-duplication and exact-record audit is unresolved because
the 1,083 zero-positive source rows are absent.

## Required future grouping rule

Use image-group-disjoint splitting. Build equivalence classes as the union of:

1. the authoritative source `image_group_id`; and
2. exact image-content SHA-256 equality.

This prevents both multiple questions for one source image and byte-identical
images stored under different sample filenames from crossing splits. A UID
must appear exactly once after grouping.

