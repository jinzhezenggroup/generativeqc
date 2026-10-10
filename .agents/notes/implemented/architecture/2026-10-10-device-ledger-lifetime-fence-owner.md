# Decision: declare shared device-ledger fences at their real conditional calls

Status: implemented
Date: 2026-10-10

## Problem

The source-owned graph/resource qualification retains six unannotated
`cudaStreamSynchronize` records per RHF/UHF history. Their adjacent asynchronous
frees identify an investigation target, not scientific ownership. Inspection of
`resource_cuda_free_async` finds a pre-existing conditional ledger fence: a tracked
allocation's logical reservation cannot be retired before the queued physical
release completes, because a subsequent owner may use another stream. This
policy is distinct from the bucket resource destructor's final stream drain.
The async-allocation metadata-rollback path has another existing lifetime drain.

## Decision

Declare the existing fences under a shared `device_resource_ledger` source owner,
with distinct release and metadata-rollback sites and conservative `lifetime`
roles. Create a source execution only when a fence actually runs. Keep the
original CUDA API, stream, conditional calls, returned/ignored status and
reservation-retirement order. This is observation, not a change to release or
allocation policy; adding, coalescing or removing fences is outside this slice.

Append owner/site IDs without changing boundary layout 2 or existing tag IDs.
The source owner is the shared ledger helper, not the enclosing HF scientific
owner. Allocation generations, byte sizes, adjacent operations and enclosing
public phases do not establish producer/transform/consumer dependencies. No
allocation/tensor identity is asserted for a fence. Lifetime work stays higher
risk even inside public close; it cannot automatically receive a hot-path
publication or setup exemption.

The source pinning procedure explicitly requires the ledger helper and graph,
bucket and eigensolver implementation files as manifest inputs. Native code
still supplies all tag names; Python does not duplicate the registry.

## Invariants

- An untracked async free remains asynchronous and emits no invented fence.
- A failed async free skips synchronization and keeps its logical reservation.
- A tracked free observes its allocation owner even after the active scope ends.
- A failed release fence returns its exact CUDA status and retains bookkeeping.
- Metadata rollback still returns host OOM and ignores its fence return status.
- Disabled or failed observation never changes CUDA calls, accounting or errors.
- Explicit API/activity coverage alone cannot certify all implicit/graph waits.

## Validation

CPU doubles compile the production helper and real CUPTI 28/CUDA SDK headers.
They check tracked/untracked release, release after scope end, failed free,
failed fence, metadata rollback (including failed rollback fence), disabled
observation and throwing observer callbacks. They verify exactly-once CUDA work,
native source correlation, independent owner scopes and preserved ledger
retirement/status semantics. Diagnostic tests keep both ledger sites higher risk
and treat `by_owner` and `by_role` as non-additive projections. Real-device
qualification is separate from these CPU tests.

## Scheduled qualification (appended after source freeze)

Slurm job **6933** completes with exit 0 on 2026-10-10 using `main`,
`gpu:5090:1`, `node1`, a finite ten-minute allocation and Slurm's unchanged device
visibility. The pre-evidence frozen tree is
`549a3e89bf8ce8baf6ff290f089b5b4570ee52f2` (9,195 source entries); native library
SHA-256 is `fe88aed090f749f45d1914ff5cc792b3b0ed113bdbf2c81aae5430cf0ecd8cfd`.
The unchanged CUPTI collector is
`72d1fef064e9910c50f5dd45df5b6139c72cdb00a02ca768df7cec373913499b`, built from
the same collector bytes retained in this source manifest. Release sm_120 and
AOT-disabled configuration remain pinned. The selected ccache 4.5.1 launcher,
actual compiler command and shared before/after cache statistics are retained;
the statistics are not exclusive per-build performance measurements.

Both RHF and UHF histories now have **369 matched source operations and zero
unannotated production transfer/synchronization records**. Their six ledger
release fences match actual CUDA APIs and CUPTI synchronization records through
the independent source correlation domain: two in force-first and four in close.
All six remain higher-risk lifetime work. The ten total lifetime fences include
the four bucket-resource drains; the **40 total explicit source stream fences
still execute**, not zero fences or zero round trips. Existing owner projections
and all observed per-public-phase semantic work counts match job 6931 exactly.
This is ownership qualification, not a performance improvement.

Activity record counts are 3,131/3,191; each source store has 786 records and
graph stores have 106/110 records. All stores report zero loss, errors and
outstanding work, and source dispatch errors are zero. Each method retains four
graph lifetimes and ten accepted host submissions, not graph execution counts.
The graph-only observer keeps intact inventory and an independent CPU H2 energy
error of `4.440892098500626e-16`. Metadata rollback and the ordinary eigensolver
resource owner are **not exercised** in these real-GPU histories; their code
presence and CPU tests cannot establish real-device coverage.

H2D bytes remain 6,649/8,041 and D2H bytes 3,306/5,586 (RHF/UHF). Matched ordinary-CUDA
energy error maxima are `1.4210854715202004e-14` / `2.842170943040401e-14`, and
force maxima are `8.881784197001252e-15` / `3.164135620181696e-15`. All five
endpoint iteration histories match exactly; numerical errors satisfy the
1e-10 energy / 1e-9 force gates.
The scheduled resource suite passes 39 cases, including two explicitly enabled
CUDA regressions. The allocation receipts pass for both methods on this same
source/library and the independently pinned native marker. Local CPU suites
pass 90 focused cases and 295 broader cases (57 subtests). An initial local
resource-suite attempt lacked a native library; those library-dependent cases
pass in the scheduled suite. Its prerequisite-failure log remains retained.

Ignored `.artifacts/issue1629-ledger-fences/` holds the frozen archive and source
manifest, exact library/collector/marker, build provenance, Slurm records, raw
CUPTI/graph/source journals, numerical histories and Memray allocation evidence.
`aggregate.py` reimports the archived source, verifies its tree, artifacts and
independent pins, and recomputes all saved diagnostics and allocation acceptance;
`qualification-summary.json` retains the result. `compare-previous.py` and
`comparison.json` retain the observed work-count comparison with job 6931.
Archive and binary checksums are reverified before this append. The evidence
append changes the current note, not the tree that generated these measurements.

Numerical-reference clarification: RHF/UHF history errors above compare
observed and ordinary CUDA histories, not CPU execution. The graph-only H2
comparison does use CPU. The earlier history CPU label was incorrect; retained
source/raw data are unchanged. The subsequent observed-work ratchet slice adds
separately labeled native-CPU backend energy/force comparisons, without equating
different backend iteration histories.

Even with `observed_production_activity_fully_annotated = true` for these selected
histories, every residency result remains **INCOMPLETE**. Cross-owner scientific
payload/dependency links, other consumers, dynamic/device-tail graph execution,
implicit blocking and owner-qualified zero-unexpected-round-trip acceptance are
unresolved. `zero_round_trip_assertion` remains null; no endpoint timing claim
follows from the job runtime, static graph inventory or this annotation coverage.

## References

- [Graph and resource fences](2026-10-10-hf-graph-resource-fence-owners.md)
- [Requested-byte ledger semantics](../compatibility/2026-10-10-device-ledger-requested-bytes.md)
- Current contract: `docs/maintainer/residency_receipts.md`
