# CUDA residency receipts

`tools/audit_residency_receipts.py` consumes declared transfer and synchronization
evidence for one CUDA endpoint. It does not measure CUDA operations. The native
`cuda_component_trace` JSONL and KS transport diagnostics remain the owners of
their existing measurements. The `#1668` source-work scanner remains an advisory
static inventory; neither its sites nor GPU-related filenames prove residency.

## Contract

The contract and receipt use `generativeqc.residency_receipt` version 1. Both
bind source commit, native source identity, build/library SHA-256, backend,
device, endpoint, problem, resource, owner, and dependency domain. A contract
declares every covered region with one role: `prepare`, `replay`, `iteration`,
`tile`, `publication`, `oracle`, or `compatibility`. The first three hot roles
(`replay`, `iteration`, `tile`) have explicit uint64 ratchets for H2D, D2H, D2D
bytes, stream/device synchronizations, event waits, and paired round trips.
Other roles are reported by role and cannot acquire a path-based exemption.
Publication can therefore be legal yet visible beside a replay sync regression.

A complete receipt requires `completed: true`, `execution: stream`, full region,
transfer, synchronization and payload-link coverage, an event count matching the
ordered `seq` stream, and every event's declared role/owner/domain. A transfer
records direction, byte count, payload identity, and dependency domain. An H2D
claiming a round trip must cite a preceding D2H sequence with the same payload
and dependency domain. Equal or summed byte counts alone never prove a pair.
Unknown, duplicated or reordered events, duplicate JSON members, integer decode
overflow, non-finite JSON numbers, excessive nesting, capture-only execution and
partial coverage return `INCOMPLETE`. A complete stream exceeding a hot ratchet
returns `FAIL`; only a complete stream within every ratchet returns `PASS`.
Receipts are assertions from a producer and must be evaluated with that
producer's coverage and provenance, not inferred from absence of events.

```console
python tools/audit_residency_receipts.py --contract expected.json --receipt observed.json
```

Exit status is 0 for `PASS`, 1 for `FAIL`, and 2 for `INCOMPLETE`. The output
separates `candidate_replay_syncs` from the `publication` role totals.

## Existing DF Trace Adapter

The historical adapter reads real `vibeqc.df_trace` JSONL with its retained
manifest. It checks the manifest schema, source/build identity,
contract-selected retained record, and LF-normalized SHA-256; rejects invalid,
dropped, or capture records; and aggregates only explicitly named owner counters.
`profiler_event_count` counts diagnostic CUDA events, not
production synchronization. The trace itself synchronizes its final event at
operation teardown, and optional progress tracing adds diagnostic region fences.
Those waits are separate from `explicit_synchronizations`,
`stream_synchronizations`, and `raw_panel_event_synchronizations` production
counters. `final_synchronization_ms` is a duration and cannot be converted to a
production wait count.

```console
python tools/audit_residency_receipts.py \
  --contract benchmarks/results/residency-receipts-1629/historical-oh-df-contract.json \
  --df-trace benchmarks/results/issue206-practical-auxiliary/diagnosis/control-diagnosis-v1/oh-def2-svp-spherical-uhf-auto.jsonl \
  --trace-manifest benchmarks/results/issue206-practical-auxiliary/manifest.json
```

The adapter always returns `INCOMPLETE`: the component trace has operation-wide
aggregates, no complete H2D/D2H/D2D or wait enumeration, no per-event payload
links, and no proof that every declared hot region is covered. It is useful for
auditing present counters and identifying precisely what the owner must supply
before a zero-unexpected-transfer claim can pass. The retained OH case is
historical evidence, not a current source or GPU run. See
`benchmarks/results/residency-receipts-1629/README.md` for its exact identity and
independent totals.

The current `generativeqc.df_trace` writer has the same construction/capture
distinction and diagnostic event synchronization. KS `KsTransportDiagnostic`
exposes cumulative setup/density H2D, scalar/matrix D2H, synchronization and
iteration counts; its public ABI does not expose D2D, per-event identity or
complete hot-region coverage. The richer `_ks_snapshot.py` owner diagnostics
cover specific action/preparation paths, not a universal transfer ledger. They
cannot be converted into a complete receipt by subtracting unrelated cumulative
snapshots. A future owner-specific instrumented producer must explicitly close
these coverage gaps before this consumer can pass a resident pathway.
