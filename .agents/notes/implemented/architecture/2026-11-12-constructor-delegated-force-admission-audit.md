# Decision: discover delegated constructor force admission

Status: implemented
Date: 2026-11-12

The Calculator constructor now delegates execution admission to a helper.
Offline public force qualification and source-only route tests still searched
only the constructor body, causing 220 route failures, one nonlocal-force test
failure and 46 capacity-report failures on source 6f6969a6. This was not evidence
that those numerical force predicates had changed: all five predicate ASTs were
unchanged from the parent revision.

Discover the unique force admission in constructor-reachable direct instance
method calls instead of binding the consumer to a new private helper name.
Unreachable lookalikes, nested callback definitions alone and ambiguous owners
fail closed. Discovery is not an interpreter or proof of every runtime path;
the existing executable route matrices, capacity guards and force qualification
checks remain required. Preserve all existing predicate and promotion gates and
provenance, without changing precision thresholds or introducing new permanent
source fingerprints. Regression fixtures exercise both admission outcomes,
renamed/delegated helpers, multiple delegation levels and rejection cases.

Regenerate the Libxc snapshot and rendered report using the official checker.
Only the two current input identities change. The retained campaign remains
stale, with the same summary/endpoints and measured identities. This is archival
consistency, not new numerical qualification. CUDA/Fast performance and numerical
acceptance remain separate from these mechanical CI repairs.

## Bounded review evidence

The combined repair passed 758 targeted host cases in two runs (448 capacity
cases and 310 route/catalog/anchor cases); 16 device-only cases were skipped.
The discovery fixtures live in the existing capacity module, preserving the
core-shard module ordering. The native probes used verified ccache 4.14.1.
Fourteen catalog/architecture/inventory checks passed; no full GPU qualification
was performed for this repair. Checkpoint regression coverage now distinguishes
shared defaults, matched explicit controls, and deliberately different controls;
the latter still rejects exact restart and requires explicit warm import.

Hosted CI run 43298950791 at source 6f6969a6 builds its actual Release CPU
libgenerativeqc.so and passes all 62 preliminary-initial-guess cases, including
all ten hf/lda-by-RHF/KS complete energy/fallback cases. This is not a skip or a
policy-adapter-only test. The author's separate CUDA-enabled-library host run
reported eight KS failures; its raw traceback is not available in this review.
The passing hosted build does not explain that local build/source difference or
qualify GPU execution. Keep these distinct evidence contexts rather than
relabeling either result.
