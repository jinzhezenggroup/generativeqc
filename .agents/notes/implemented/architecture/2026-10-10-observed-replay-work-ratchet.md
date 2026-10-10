# Decision: gate observed replay work without manufacturing residency PASS

Status: implemented
Date: 2026-10-10

## Problem

The qualified RHF/UHF source journals now annotate all observed explicit
transfer/synchronization work, but still cannot prove scientific dependencies,
all graph executions or implicit blocking. Correctly retaining an unconditional
full-residency `INCOMPLETE` is necessary; treating that status as approval could
nevertheless hide a new, definitively observed replay copy or fence. A one-time
before/after artifact comparison is not a reusable review gate.

## Decision

Add an explicitly named observed-work ratchet, independently of full residency
acceptance. Native source tags remain authoritative. Extend the shared source
counter projection with public phase, owner and source role, using exactly the
same joined API/activity work as `by_owner` and `by_role`. Keep API calls,
zero-byte calls, stream fences and CUPTI activity quantities distinct; projections
cannot be added as extra physical work.

Policy lives under `manifests/` and is selected by a SHA-256 of the complete fixed
workload, including numerical gates. Limits are reviewed inputs derived from the
previous independent scheduled qualification (job 6933, frozen source tree
`549a3e89bf8ce8baf6ff290f089b5b4570ee52f2`), not learned from the run being checked.
The prepared producer pins the whole policy file and selected profile before
work, rechecks afterward and retains a separate verdict. Unlisted owner/role
pairs and unowned blocking work receive no allowance in selected hot scopes.
Explicit oracle/compatibility limits can remain legal; their labels never create
a blanket exemption. Known nonblocking queries are retained outside the blocking
work budget, not mistaken for fences. Preparation and observer completion work
outside selected replay are not counted as hot regressions.

`WITHIN_OBSERVED_RATCHET` means only that the named observed quantities meet
limits. `VIOLATION` requires review; invalid/lost source evidence remains
`INCOMPLETE`. A configured gate that is not within limits causes capture exit 1,
without deleting collected diagnostics. Even within limits, the full residency
result stays `INCOMPLETE`, exit 2, with a null zero-round-trip assertion.

## Rejected alternatives

- Claiming PASS from complete explicit annotations would ignore dependency,
  implicit-wait and graph-execution gaps.
- Checking only total bytes hides sync amplification, zero-byte call overhead
  and migration between owner/role contexts.
- Reusing publication/preparation allowances for every call in the same file or
  enclosing public endpoint would allow an unreviewed hot-path exemption.
- Raising limits automatically during `--pin` would ratchet in the regression
  being tested rather than compare against independent accepted evidence.
- Counting nonblocking query records as waits would create false sync failures.

## Invariants and remaining scope

The native ABI, CUDA calls, scientific execution routes, fallback policy and
numerics do not change. Counter integrity is required before an observed gate
can be accepted. Policy decoding is bounded and rejects duplicate/non-finite
JSON, ambiguous selectors, incomplete fields and invalid uint64 limits. The
retained profiles cover only the fixed direct-FP64 H2/water/H2 warm energy and
force cases. Missing observed API provenance is not a zero-work proof.

This does not complete the broader residency issue: cross-owner scientific
links (including the retained host-staged DF adapter), other workloads/owners,
dynamic/device-tail graph execution, implicit blocking and owner-qualified
zero-unexpected-round-trip acceptance remain unproven. Same-count/byte CPU
fallbacks can escape this observed-work gate. No endpoint timing or scientific
residency claim follows from a budget verdict or Slurm job runtime.

## Validation

CPU fixtures verify contextual limits, fixed independent policy selection,
source loss, missing phase provenance, known nonblocking queries, conservative
lifetime risk, explicit oracle/compatibility admission and extra zero-byte calls.
The producer's pinned policy changes when limits change. Real-device positive
and controlled extra-sync qualification are separate from these synthetic tests;
scheduled evidence is appended only after source freeze and successful runs.

Qualification evidence correction: inspection found that previous RHF/UHF
`numerical` rows compare observed with ordinary CUDA histories, despite CPU
labels in two historical notes. Those labels are corrected explicitly without
changing their frozen source/raw evidence. Only the separately named graph-only
H2 case actually used CPU in those jobs. The producer now retains separately
labeled native-CPU backend histories outside capture, keeps energy/force gates
mandatory and enforces exact ordinary-CUDA iteration equality. Independent
backend iteration counts are retained but need not agree. This strengthens the
observer qualification; it does not change scientific production code or claim
an independent PySCF/theory validation from a native-backend comparison.

## Scheduled qualification (appended after source freeze)

Slurm job **6942** completes with exit 0 on 2026-10-10 using `main`,
`gpu:5090:1`, `node1`, a finite ten-minute allocation and unchanged assigned
device visibility. The qualified pre-evidence tree is
`adabeda138c4d336de20fe46fd6efd67d02f6e74` (9,199 manifest entries), exported as
`source-oracle.json`. The earlier `source.json` in this artifact directory is
an unqualified initial freeze, not the source for these measurements.
Native library SHA-256 remains
`fe88aed090f749f45d1914ff5cc792b3b0ed113bdbf2c81aae5430cf0ecd8cfd`; collector
SHA-256 remains `72d1fef064e9910c50f5dd45df5b6139c72cdb00a02ca768df7cec373913499b`.
Scientific native bytes and CUDA calls are unchanged. Release sm_120 and
AOT-disabled build configuration, verified ccache 4.5.1 launcher/commands and
shared cache statistics are retained. Statistics do not imply an exclusive
per-build cache-hit or performance measurement.

Both positive RHF/UHF histories report `WITHIN_OBSERVED_RATCHET` with no issues
or violations. They retain 369 source operations, zero unannotated production
transfer/sync records and 40 actual explicit source stream fences per method.
The policy covers warm energy and force cases, not arbitrary workloads. Source
stores have 786 records each; activity stores have 3,131/3,191 records, with no
loss, parse, dispatch or outstanding-operation errors. H2D totals remain
6,649/8,041 bytes and D2H totals 3,306/5,586. The graph-only observer also keeps
intact inventory; its independent native-CPU H2 energy error remains
`4.440892098500626e-16`.

The separate controlled RHF negative process adds exactly one real
`cudaDeviceSynchronize` while public `energy-warm` (region 3) is still active.
This call is outside production instrumentation and deliberately has no source
ownership exemption. Actual CUPTI activity records its successful return, API
identity, device synchronization kind and public correlation. The observed gate
reports `VIOLATION`, one unannotated selected activity over limit zero, and the
capture exits 1 as required. All source/API/graph journals remain intact; source
operation counts and the numerical gates do not change. This is an actual
scheduled extra-sync regression check, not merely a synthetic JSON mutation.
The injection harness is retained and checksummed separately from production
source; it adds no production test hook or scientific fallback.

Ordinary CUDA histories retain exact iteration equality. Positive RHF/UHF
ordinary-CUDA energy error maxima are `1.4210854715202004e-14` for each; force
maxima are `5.162537064506978e-15` / `6.5503158452884236e-15`. Separately labeled
native-CPU reference energy errors are `9.947598300641403e-14` for each; force
errors are `8.375189430864793e-11` / `8.369266391028418e-11`. The controlled
negative also passes both numerical comparisons (native-CPU energy
`8.526512829121202e-14`, force `8.376027649248385e-11`). All energy/force gates
remain 1e-10/1e-9; CPU convergence histories are recorded but not equated with
CUDA histories. This is native-backend cross-checking, not an independent
PySCF/theory validation or a new scientific-method qualification.

Local final CPU suites pass **346 tests and 57 subtests**; the focused ratchet
and source suite passes 120 tests before the numerical-reference extension.
The scheduled resource suite passes 39 cases, including two explicit CUDA
regressions. Source-matched RHF/UHF allocation receipts continue to pass under
the strengthened ordinary iteration gate.

Ignored `.artifacts/issue1629-work-ratchet/` retains the qualified frozen archive,
manifest, exact library/collector/marker and build inputs, independent contracts,
positive and controlled-negative raw histories/verdicts, reference-kind-labeled
numerical evidence, allocation/Memray receipts, Slurm records and logs.
`aggregate.py` reimports archived source, verifies its tree and pinned artifacts,
recomputes all source/graph/activity/ratchet diagnostics and allocation acceptance,
and verifies both numerical-reference kinds and the controlled negative.
`qualification-summary.json` retains the result. Archive/binary checksums are
reverified before this evidence append; the append does not change the frozen
tree. Every full residency result remains `INCOMPLETE`, with a null zero-round-
trip assertion and no endpoint timing claim.

## Workload compatibility (2026-10-10)

The shared audit workload now pins an internal final-atom displacement, rather
than translating every atom equally. The residency producer consumes those
exact pinned coordinates too. Retained v1 limits remain bound to their original
rigid-translation workload and source/binary identities: changing their workload
hash without fresh measurements would invent qualification for different work.
Tests retain the historical selector and explicitly reject the new workload;
current-policy pinning is tested with clearly synthetic CPU metadata, not a
relabeled GPU baseline. A fresh non-rigid observed-work policy remains required
before that optional assertion can be used on current captures.

## References

- [Device-ledger lifetime source qualification](2026-10-10-device-ledger-lifetime-fence-owner.md)
- Current contract: `docs/maintainer/residency_receipts.md`
- Reviewed policy: `manifests/residency_work_ratchets/hf_prepared_direct_fp64.v1.json`
