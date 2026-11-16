# Proposal: isolate geometry-only composite source preparation

Status: proposed; no prototype or compiler/runtime advantage claimed
Date: 2026-11-16

## Observed remaining bottleneck

The complete clean sm120 native build at
`bd9d6cc30ec70c8fe05fbbc948ef6cd535a69247` spends 2,130.439 task-wall seconds
in `direct_jk_kernels.cu`, versus 1,514.488 in bounded fallback. These tasks
overlap other work; they are not isolated cold-wheel measurements. The retained
Direct-JK object has 565 device-function resource records. They include 156
`canonical_jk_kernel` specializations: 13 angular orders times 12 combinations
of spin, Cartesian layout, paired-range, mixed and composite-source modes.
There are also 59 materialized canonical kernels and other scientific owners.

The composite-source mode was added by `9d9083606` on November 7, 2026
(PR #3033). It contributes two spin variants across all 13 orders. Its runtime
benefit is qualified separately in
`../implemented/performance/2026-11-07-direct-composite-source-build.md`.
Neither its extra specializations nor their names establish removable science.

## Distinct hypothesis

For a nonzero full-range coefficient with a nonnull `source_values` destination,
the composite mode computes a geometry-only paired radial value, stores
`selected + coefficient * full`, then continues before density and Fock scatter.
The generic kernel nevertheless also compiles its ordinary J/K branch because
the source pointer is a runtime argument. This is a potential ownership boundary
for a dedicated source-only kernel, shared across spins, rather than another
noinline extraction of density preparation or per-order range dispatch.

Keep the existing ordinary, paired-Fock, mixed and materialized consumers. Do
not route zero-coefficient source preparation through a paired evaluation:
that would add scientific work. Do not remove the qualified composite-source
optimization or its bounded raw fallback merely to shrink the binary.

## Contracts to establish before implementation

- Audit every caller, especially raw/component fallbacks and source-window
  builders. Keep existing null-pointer behavior: a nonzero coefficient with no
  source destination currently selects ordinary Cartesian J/K work. A new
  geometry-only path must not accidentally discard that behavior.
- Preserve ordered AO rows, item/angular domains, screened pair enumeration,
  prefix clipping, bounded work windows, `work_begin`, output indexing and
  all existing finite/error handling. Never add density-based skipping to
  geometry-only source preparation.
- Preserve paired scientific evaluation and FP64 expression order, including
  runtime range/omega, zero and signed zero, finite signed coefficients,
  repeated-shell/component domains and admitted constant-AOT cases.
- Retain diagnostic semantics explicitly: the existing source store continues
  before ordinary evaluated-work accounting. Do not silently change work
  counts or mistake candidate-domain counts for executed recurrence work.
- Keep the nonzero-coefficient admission and null-pointer fallback explicit;
  do not depend on undocumented source-lease lifetime or a hidden host oracle.

## Required qualification and rejection conditions

Freeze the complete affected owner and verified compiler-cache provenance.
Compare full compiler-worker resources and all retained architectures; helper
text sizes and an empty-compile sample are insufficient. Keep independent
native source construction/replay, bounded-window, screening, failure-injection
and lifetime tests, then use scheduled real-GPU complete source and public
cold/warm endpoint evidence with semantic work counts. Pure forwarding mocks
cannot qualify scientific arithmetic or the resource/lifetime contract.

Reject if removing the generic branches does not materially reduce complete
owner cost, or if the new kernel harms complete source construction/replay or
method behavior. The rejected density-preparation and shared range-dispatch
experiments remain rejected; see
`../rejected/2026-11-16-direct-shared-density-preparation.md` and
`../rejected/2026-11-16-direct-range-order-sharing.md`.

The original resource dump, demangled census and build timing context are
retained in `/tmp/gqc-wheel-g-value-orbits/merged-direct-jk-specializations.json`,
`merged-direct-jk-resources.txt` and `merged-native-build-census.json`.
