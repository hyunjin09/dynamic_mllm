# Invalid Phase-56 smoke attempt

- Frozen contract: `0931226b2c6823f2e4cd6721844163393a08df8c9f5d6c8904bf4e288da491dd`
- Status: **invalid; do not use for scientific results or training**
- Direct observation: three smoke ranks completed dense parity/search or preservation and then stopped in route-manifest serialization with `KeyError: 'trigger_layer'`; the fourth rank was interrupted after the global smoke had already failed.
- Supported cause: imported Phase-55 result rows expose `trigger_layer`, while fresh Phase-56 manifest rows expose the frozen source field `first_trigger_layer`. The serializer accepted only the imported alias.
- Scope: implementation-only smoke. No full 1,800-row fresh workload was launched, no accepted smoke completion was emitted, and no Stage-2 training/evaluation occurred.
- Repair: accept both frozen row schemas through one explicit trigger-layer accessor and add a regression test before freezing a replacement contract.
