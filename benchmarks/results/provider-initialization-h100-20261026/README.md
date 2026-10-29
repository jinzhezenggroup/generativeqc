# Historical initialization-only engineering snapshot

All seven original published files, including the reader, source/input archive,
observations, original README and file attributes, remain recoverable byte for
byte from the already-merged commit
`11b7bda884901e94c132433691106100e54b1f55` (#2815).
`snapshot.json` pins every path, length, SHA256 and published Git mode.

```console
python tools/restore_retained_evidence.py --all --manifest benchmarks/results/provider-initialization-h100-20261026/snapshot.json --output .artifacts/recovered-provider-initialization
python .artifacts/recovered-provider-initialization/benchmarks/results/provider-initialization-h100-20261026/reader.py
```

The stock restorer validates every Git blob before publishing a new complete
directory. It does not fetch, execute operational scripts, load CUDA or run
chemistry. Explicitly fetch the pinned commit first in a shallow checkout that
lacks it. All seven restored files were compared to their original published
bytes before this migration.

This is a task-owned historical engineering storage migration to admit the
[current-input owner receipt](../h100-provider-owner-20261027/v9/README.md) under
the unchanged retention limits. No independent scientific oracle or molecular
array is removed. The original status stays
`PASS_ENGINEERING_INIT_SUBCONTRACT_ONLY`; it never established a full installed
SDK, RHF, solver or scientific result. Its original privacy limits, legacy
refusal and failure evidence remain in the recovered bytes. Snapshot restoration
does not recover previously unavailable pre-scrub bytes or transfer its
qualification to a later build.
