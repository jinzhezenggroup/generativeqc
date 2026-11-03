# Decision: reuse the bounded CPU MP2 prepared source through forces

Status: implemented, scoped numerical/resource validation; endpoint performance unqualified
Date: 2026-11-04

## Problem

The incomplete policy-retirement campaign exposed repeated CPU AO regeneration
inside MP2 force/response transformations. Its diagnosis and instrumented
feasibility observations are preserved in
`../../proposed/2026-11-04-cpu-mp2-force-source-residency.md`. They did not
qualify a production implementation or replace the failed physical CBS campaign.

Conventional CPU MP2 already constructs `cpu_exact_plan_` for SCF and borrows its
integrals for energy. Retiring that owner unconditionally before forces discarded
usable resident storage. Increasing source tiles alone did not remove AO
regeneration, and adding another unconditional molecular ERI cache would have
duplicated storage and changed resource admission.

## Decision

Keep the existing CPU prepared owner and borrowing source view through the force
endpoint when optional reference-source sharing is enabled. Pass that source to
the existing conventional force owner, whose provider charges actual retained
numeric capacity. Apply the existing compiler-generated checked source-tile
planner to host-readable sources as well as device-readable CUDA sources. Every
candidate covers the streamed rotation-block lifetimes, response and derivative
reservations. Neither the energy schedule nor its work guard changes.

The existing `GENERATIVEQC_MP2_REFERENCE_SOURCE=0` switch explicitly preserves
the owner-retiring streaming force route. The existing
`GENERATIVEQC_MP2_FORCE_SOURCE_TILING=0` switch preserves legacy force tiling.
Neither switch changes the scientific model or removes complete-force planning.

If a phase raises `std::length_error` or `std::bad_alloc` while the optional
prepared view is live, unwind phase-local work, clear the active source pointer,
destroy the borrowing view before its prepared owner, and retry once with the
existing bounded raw source. Subsequent phases use that raw source too; the next
reference execution may prepare its own owner again. The same optional-residency
policy covers companion/reference-force and energy phases, not only the final
correlated force. Other scientific and runtime failures are not retried.

Retained reference and warm-state storage is not measured by the force-local
allocation tracker. Whole-endpoint measured peak remains unknown when those
owners were live, including an unsuccessful resident attempt before fallback.
Checked planned/numeric peaks remain separately reported. Do not relabel an
admitted bound as a measured allocation peak.

## Evidence and limits

`tests/python/test_mp2_cpu_reference_source.py` executes actual CPU complete
energy/force publications against the pinned independent physical H2 DZ MP2
component in `tests/reference_data/cbs/h2-dz-tz.json`. It covers default/explicit
resident and streaming selection, legacy/generated tiling, repeated prepared
warm execution, genuine optional-source admission failure/retry, and a scientific
denominator error that must not become a memory fallback. Energy tolerance is
1e-9 hartree and complete-force maximum absolute tolerance is 2e-6; route parity
is an additional gate, not the independent oracle.

The compiled native MP2 contract locates the actual complete-force admission
boundary for its small physical fixture, checks force/response parity and
reported capacity for admitted candidates, and requires zero source reads on
every rejected candidate and at exactly one byte below admission. The earlier
test assumed that one byte below a roomy plan's peak must reject; adaptive CPU
tiling can legitimately admit a different plan there. The pre-work rejection
requirement is retained at the real boundary, not removed.

The final CPU Release build uses verified sccache 0.16.0. Its actual native
source query matches the complete input inventory:
`2ec35ab1a42783fdfcaa5ab3f7fe103dec4130175455a213771a2632de36f460`.
Validation passes all 133 native CTests, 101 affected MP2/gradient/prepared-batch/
CBS Python tests (40 explicit skips), and 821 policy/precision/retained-evidence/
ownership tests (313 explicit skips). Skips are not device execution evidence.

A bounded reconnaissance run on the first source-matched CPU build admitted
both routes at 9,000,000 bytes. The resident attempt advertised 571,248 retained
source bytes, reported a memory fallback, then completed using the 1,632-byte
streaming source with a 8,998,496-byte planned/numeric peak. These are observations
of that build, not portable byte thresholds. Regression admission uses actual
native planning on the selected build rather than pinning those numbers.

The `bad_alloc` catch is implemented but has not been independently fault-injection
qualified. This change does not select packaged CPU derivatives: the existing
single-channel reference derivative route is unchanged. It does not establish
physical full-CBS E/F throughput, ABBA/BAAB or the unchanged 1.05 endpoint gate.
The earlier diagnostic's approximately 41-second result must not be attributed
to this clean implementation. No CUDA runtime qualification or parent #2227
completion is implied.

## Revisit when

Consider further derivative routing only with an owner-provided capability
contract, independent full-observable references and supported-domain/resource
fallback gates. Consider source or tile changes only with complete endpoint and
semantic-work measurements; fewer transform operations alone are insufficient.

## References

- Issue #2227; draft PR #2972.
- `../../rejected/2026-11-04-cbs-generated-derivative-shortcut.md`.
- `2026-10-12-mp2-cuda-reference-source-ef.md` for the existing optional CUDA
  source and bounded-retry policy.
- `benchmarks/results/required-native-policy-20261104/` retains the earlier
  incomplete campaign. Its immutable source/binary provenance is not changed or
  reinterpreted by this implementation.
