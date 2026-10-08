# Semantic producer-work receipts

`tools/audit_producer_work.py` compares exact semantic producer counts for the
same scientific problem, resource regime, producer domain, dependency identity,
reuse owner, and invalidation owner. It is separate from memory budget and peak
storage accounting. A static schedule census is a ratchet for possible work;
it is not a completed GPU execution, wall-time result, or numerical acceptance.

The first production adapter calls
`generativeqc_compiler.method.df_exchange_schedule.projected_exchange_schedule`
and enumerates outer and prefix panel visits independently of its
`generated_rows` formula. It rejects disagreement, including the final short
row block, and binds the exact source file SHA-256. The policy requires the
loaded package to come from the declared source root. It reads the schedule
source once, then hashes and executes that same captured byte string in an
isolated module namespace. A long-lived process therefore cannot combine a
stale imported function with a newer on-disk source identity. The native owner
uses the generated `visit_projected_exchange` two-slot traversal; its project
callback increments `streamed_occupied_raw_generation_rows`; a successful host
traversal then records `streamed_occupied_row_blocks`.

Generate a shape census from a source checkout, with a separately reviewed
build digest in `REVIEWED_BUILD_SHA256`:

```sh
python tools/audit_producer_work.py schedule --root . --n 12 --auxiliaries 5 \
  --rank 2 --capacity 48 --dense-row-blocks 3 --dense-output-blocks 3 \
  --triangular --scientific-problem shape-12x5-rank-2 \
  --dependency-identity geometry-basis-coefficients-v1 \
  --build-sha256 "$REVIEWED_BUILD_SHA256" > candidate.json
python tools/audit_producer_work.py compare baseline.json candidate.json \
  --baseline-source <baseline-snapshot>/python/generativeqc_compiler/method/df_exchange_schedule.py \
  --candidate-source python/generativeqc_compiler/method/df_exchange_schedule.py
```

The baseline must be generated and reviewed against its own source snapshot,
not copied from the candidate. `compare` requires both source files and rejects
a stale hash. The caller must separately verify the claimed build digest,
scientific inputs, dependency closure, resource budget, and execution identity.
The receipt has no authority to authenticate those claims by itself.

The strict v1 receipt has exactly `schema`, `identity`, `work`, and `evidence`.
Identity contains `scientific_problem`, `resource_regime`, `producer_domain`,
`dependency_identity`, `reuse_owner`, `invalidation_owner`, `source_sha256`,
`build_sha256`, and `execution_identity`. Work contains integer
`logical_elements`, `executed_elements`, `producer_callbacks`, and
`outer_consumer_multiplicity`, plus `memory_budget_bytes` and nullable
`peak_bytes`. `executed_elements` means enumerated schedule work for static
evidence; it means submitted producer work for the trace adapter. The evidence
fields are `kind`, `phase`, `coverage_complete`, `execution_complete`,
`source_matched`, `reusable_dependency_proven`, and nullable
`reuse_proof_sha256`. A true proof flag requires a reviewed proof digest.

Integers are exact unsigned 64-bit values; booleans, floats, negatives,
overflow, missing/extra fields, and duplicate JSON keys fail. Ratios are
reduced rational numbers. Comparisons require the same domain and complete
coverage. Increased producer elements or callbacks return `FAIL` as a work
ratchet, while classification remains `work change; reuse unproven` until an
independent reusable-dependency
proof exists. Repeated work can be legitimate under a bounded budget or changed
dependencies. `PASS` for a static census always reports runtime acceptance as
`INCOMPLETE`.

The trace adapter reads the existing `generativeqc.df_trace` JSONL operation
record selected by `--trace-id`. Missing or duplicate IDs and malformed lines
fail. It reconciles selected tile production counts and value bytes, or the
native streamed occupied generation-row, projection-callback, and row-block
counters. It rejects invalid and truncated records, duplicate tile keys,
and counter disagreement. These
counters are submitted before or around launches. A valid `stream` record and
its final CUDA event do not prove that the scientific endpoint returned
success or that all required source tiles were observed. The adapter therefore
sets `coverage_complete`, `execution_complete`, and `source_matched` to false;
`graph_capture` remains capture evidence only. Such comparisons report
`INCOMPLETE`; setting `execution_complete` on a trace receipt is invalid. A
later endpoint receipt needs independent complete-call status, source-matched
installed build, and expected-domain coverage before runtime
acceptance can be claimed.

For the native streamed-row counter, select one operation from the production
JSONL trace with `trace --trace-id <id> --tile-kind streamed_rows` and supply
`--source src/scf/cuda/df_occupied_exchange.cpp`, the scientific/problem and
resource identities, dependency and execution identities, and the reviewed
build digest. The output is diagnostic submission evidence only.

## Comparable-source CI ratchet

The PR work-audit workflow runs `tools/ratchet_producer_schedule.py` with
`--fail-on-work-growth`, matching the immutable PR base source to the current
candidate under the same five frozen DF production schedule shapes:

```sh
python3 tools/ratchet_producer_schedule.py --base-sha "$BASE_SHA" \
  --output .artifacts/producer-work-ratchet.json --fail-on-work-growth
```

- A complete, source-bound, same-domain increase in producer elements **or**
  callback count is a CI failure. This flags a **work regression requiring
  review**, not a proven scientific bug or permission to change memory budgets.
- Changed imported dependencies, changed producer-work analyzer bytes, unavailable base objects, unsupported shapes
  or missing receipts retain explicit `INCOMPLETE` JSON status and a warning;
  this scoped CI mode does not block on unknown observations or relabel them
  `PASS`. The default CLI still fails closed for both `FAIL` and
  `INCOMPLETE` unless `--fail-on-work-growth` is selected.
- Counts are static schedule visits, not completed GPU execution, measured
  bytes, throughput, or reusable-dependency proof. The uploaded artifact records
  both source identities and each shape's baseline/candidate evidence.

This slice does not modify native scheduling, install a new counter system,
claim GPU execution, or complete issue #1628's runtime/scientific gates.
