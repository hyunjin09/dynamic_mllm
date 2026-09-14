# Invalid Phase-54 Freeze

This metadata-only freeze is invalid and must not be used.

- Failure: `KeyError: 'static_config'` while rendering `protocol.md`.
- Cause: the first frozen-contract constructor copied projected top-level fields but omitted the nested static configuration required by runtime validators.
- Scope: no score worker, Qwen forward pass, manifest aggregation, or scientific computation ran.
- Resolution: added a regression test for frozen-contract construction and refroze from an empty canonical output root.
