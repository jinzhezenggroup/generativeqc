# Decision: specialize COSX primitive headers to its existing through-f domain

Status: implemented
Date: 2026-11-16

## Problem

The full-DF SCF owners require g-capable primitive headers, but COSX included
those same headers despite its existing through-f basis contract. Its value
translation unit parsed a 14,904,718-byte value header; its derivative owner
parsed that header plus 26,657,437 bytes of gradients. Dynamic component dispatch
kept unsupported g cases reachable to the compiler even though native entry
points reject those bases.

`AoBasis(const core::System&)` rejects angular momentum above three. The COSX
staging constructor, molecular response, point response and response diagnostic
all construct that basis from the input system. AO spatial differentiation can
raise intermediate polynomial powers without raising the public basis bound;
the ESP primitive components still come from the original basis expansion.

## Decision

Generate separate through-f COSX value and gradient headers. CMake owns this
consumer bound. The generic compiler accepts an optional lower-snake-case
`owner`, which isolates emitted namespaces and guards without changing the
mathematical IR. COSX uses `owner="cosx"`; the full-DF owners keep their existing
default symbols and g bound. No policy or backend selection moves into the
primitive compiler.

This intentionally adds two generated files rather than reducing total generated
files or on-disk bytes. It reduces what two NVCC consumers must compile. The
gradient owner's combined primitive header input falls from 41,562,155 to
10,748,387 bytes (74.14%); the value owner's input becomes 3,100,663 bytes.
Combining the two consumers' inputs is not a whole-wheel source-size metric.

## Invariants

- Keep SCF g values, all 150 derivative programs, all supported architectures,
  scientific acceptance gates and bounded native execution unchanged.
- Never link different bounds with identical inline C++ names. Independent
  namespace and include-guard ownership must both survive refactoring.
- Keep source ownership separate from scientific inventories. Emitted source
  identity changes deliberately; mathematical serialization does not.
- Reject unsupported COSX bases before primitive dispatch. The native regression
  exercises the value, point-response and molecular-response entry points.
- Retain the existing independent through-f Libcint header tests, now exercising
  the actual COSX symbols with the verified compiler cache. A small compiled
  two-owner test checks coexistence and different invalid-component bounds
  without adding another required-CI full-header compilation fixture.

## Evidence

The comparison starts at `0402c2e4e3dd6fe7b4df9e5e75daf2a7454767cf`.
Local evidence and reproducible drivers are retained under
`/tmp/gqc-wheel-cosx-scope/`.

- `emission-report.json`: one-off byte preservation of both default headers at
  maximum angular momenta three and four. Renaming only the namespace/guard of
  the original f artifacts exactly reproduces the COSX artifacts. These are
  investigation proofs, not permanent source-fingerprint tests.
  `f-scalar-preservation.json` additionally verifies that all 32 supported value
  helpers and 32 supported gradient helpers already present in the full g
  artifacts have byte-identical scalar bodies in the specialized artifacts.
- `candidate/multiarch/`: both complete production COSX CUDA owners compiled
  with Release `-O3`, NVCC 12.9.86, sccache 0.16.0, sm80/sm90/sm120 and compute80
  PTX. Two top-level requests record zero hits and sixteen cacheable misses.
  The objects are 10,405,048 and 15,436,960 bytes. The cache server's reaped-worker
  maximum RSS is 3,625,192 KiB. Its timing/resource scope is these two owners,
  not a cold wheel build.
- `raw/oracle-report.json`: Slurm job 17504 on an RTX 5090 executes the actual
  scoped value and gradient headers, 22,994 primitive records per launch, with
  64/128/256 threads. All 498 independently Libcint-recontracted Cartesian and
  spherical fixtures pass at atol 3e-11 / rtol 6e-12. The maximum absolute error
  is 7.14e-14. This qualifies primitive arithmetic, not a complete COSX endpoint.
- Focused Python gates pass: 148 value/gradient/ownership cases, 121 COSX cases
  with one existing skip, and 189 compiler/catalog/CI-resource contracts.
  An additional 214 independent default g-header regressions pass. The pinned
  `ty==0.0.82` check and compiler-structure audit pass.
- The complete sm120 native production library and CLI build successfully with
  automatic portable XC AOT enabled. The patch-stage library is retained as
  `candidate/libgenerativeqc.so`. Slurm job 17506 passes all 16 independent
  PySCF Direct-HF endpoints (RHF/UHF, Cartesian/spherical, cold/warm/moved/restored).
  Compared with the frozen parent, all iteration and Fock-build counts match;
  maximum energy/force errors are 6.22e-14 / 8.04e-11. This is a complete native
  product regression, not a COSX-method or installed-wheel gate.
- `catalog-preservation.json`: refreshing dependency audit inputs preserves the
  Libxc summary, qualified endpoints, historical receipts and transition list.
  It does not relabel stale evidence as current acceptance.

## Existing endpoint blocker

Do not describe the primitive or compiler evidence as a passing complete COSX
endpoint. The unmodified baseline standalone target first fails to link because
it omits current CUDA tracing/device-facts and full-range CPU derivative objects.
The exact diagnostic link command adds those existing production objects, without
changing their code or suppressing unresolved symbols.

That original binary then fails on Slurm job 17503 with
`native/Python grid plan mismatch`, before its COSX endpoint comparisons. The
COSX callers retain an older duplicated grid-byte formula, whereas
`cuda_grid_storage` additionally reserves the current local-XC panel. No guard is
disabled, expected-byte identity overridden, or resource accounting weakened to
obtain a passing result. Those inherited runtime/link defects are outside this
compiler-size change. Complete COSX endpoint timing and qualification remain
explicitly unverified; no runtime or whole-wheel speedup is claimed here.
The specialized standalone binary reaches the same original guard on Slurm
job 17505, after its new unsupported-g entry-point checks pass.

## Rejected alternatives

- Lower the shared full-DF bound: that removes real SCF g capability.
- Reuse default namespaces for different bounds: include guards can hide a
  header in one TU, and inline definitions can violate ODR across TUs.
- Add macro aliases or import shims: these obscure ownership and do not establish
  a single definition for differing generated programs.
- Depend on dead-code elimination of the full g artifact: runtime component IDs
  keep the oversized dispatch compiled in the COSX owners.
- Add another full f host-compilation matrix to required CI: retargeting the
  existing independent oracle fixtures retains coverage without duplicating its
  heaviest compilation work.

## Revisit when

If `AoBasis` gains public g support, extend the COSX emission bound together with
the packed-basis domain, native resource limits and independent value/response
endpoint gates. A generic generator's ability to emit g is not sufficient.
Resolve the inherited grid-storage and standalone link defects before claiming
complete COSX endpoint qualification or measuring a runtime speed change.
