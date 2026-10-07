# Decision: share weighted-Gram lowering without changing incumbent arithmetic

Status: implemented
Date: 2026-10-07

## Problem

Gaussian SCF and GFN2 already used the same `tensor.scf` density and
energy-weighted-density equations, but their actual lowering was split between
Gaussian emitters, four independently built GFN2 scalar graphs, embedded CPU
column scaling/DGEMM, and two embedded CUDA paired reductions. Shared equation
hashes alone did not establish common ownership of those implementations.

## Decision

`tensor.weighted_gram` recognizes only the existing unit-coefficient ternary
contraction and optional occupation/energy multiply. It binds coefficient aliases,
scientific domains, FP64 arithmetic and the original weight node. Checked stages
are scalarizations of this recognized region using the existing TensorIR and
scalar emitter. There is no new scientific IR or density operator.

`tensor.native_lowering.weighted_gram_candidate` projects the same original
nodes through `TensorLoweringAdapter` and owns the retained implementation,
physical views, explicit/occupied-prefix weight representation, checks and
provider choice. `tensor.weighted_gram_emit` consumes the selected candidates
and owns actual generated arithmetic and loops. Native CPU execution consumes
the generated candidate type; only its implemented checked-column-scale/LP64
DGEMM specialization is admitted. The implementation lives in
`src/tensor/weighted_gram.hpp`, with only a primitive DGEMM function pointer
crossing the provider boundary.

Gaussian CPU/CUDA generators now use this common owner. GFN2's four duplicate
scalar graph builders and embedded CPU density function are deleted. The two
GFN2 CUDA kernels retain their method-owned envelopes and include a shared
compiler-emitted contraction loop. No new runtime helper call replaces either
CUDA loop. Legacy generated scalar symbol names are retained as physical source
bindings, with their definitions emitted by the common lowerer.

## Retained schedules and invariants

- Gaussian CPU P keeps its occupied prefix, product order and canonical-weight
  1/2 mirroring. General scalar weights retain full-square computation. W keeps
  its full-square, left-associated weight/energy/coefficient evaluation.
- Gaussian CUDA keeps occupation templates 1/2, column-major views, the original
  full-square launch with upper-triangle P ownership, and full-square W. Its
  representative graph describes the complete n-by-n eigenframe: occupied[] is
  a virtual occupation-prefix specialization, not a shorter physical orbital
  dimension. Rectangular occupied-CUDA candidates are rejected; state stride
  remains n*n. Existing active-mask handling remains unchanged.
- GFN2 CPU keeps checked column scaling followed by its existing LP64 DGEMM:
  column-major, N/T, n/n/n, all leading dimensions n, alpha 1, beta 0. The existing
  scratch panel and weight vector are borrowed, with no additional allocation.
- Restricted CPU still computes f*epsilon for band energy, forms P, repeats the
  checked f*epsilon multiplication in place, then forms W. Unrestricted CPU still
  performs alpha weights/P/W then beta weights/P/W. No multiply or reduction is
  hoisted, merged or dropped.
- GFN2 CUDA keeps its paired lower-triangle P/W schedule, orbital traversal,
  grid-stride tiling, separate nonfused product overflow check, checked FMA,
  accumulator-update order and joint scratch stores.
- Fractional occupations and signed W remain supported. Method owners retain
  occupation construction/admission, ragged/spin offsets, active/sequence state,
  error recording, diagnostics and final transactional publication. A failed beta
  channel continues to suppress publication of both physical channels.
- All planning, arenas, scratch offsets and bindings are unchanged. Candidate
  resource bytes describe *additional* storage; existing borrowed panels and
  method-owned transaction buffers are not allocated again or hidden.

## Source and identity evidence

The baseline is ca98c41e (tree bb11434a), before this migration. Frozen SHA-256
gates cover both complete Gaussian generated headers, both complete GFN2
scalar/electronic generated headers, and the complete GFN2 density CUDA source
with its generated loop includes expanded. All five remain byte-identical.
`tests/python/test_weighted_gram_lowering.py` records the baseline hashes.

Generated CPU density/scalar bodies, generated CUDA helpers and expanded CUDA
kernel/function bodies are unchanged. The extracted CPU native executor changes
its owner/type binding but retains the exact primitive operation/dispatch
contract, covered by independent tests rather than claiming identical source.

The compiler/source inventory changes deliberately, including new generator
files and new common modules. Therefore build/source identities change; this is
not claimed to preserve an artifact identity. Scientific equation, existing
scalar graph and existing generated-source identities remain unchanged.

## Validation

- Graph mutations changing factors, coefficients, aliases, domains, precision or
  paired input identity are rejected. Unsupported rectangular checked and
  occupied-CUDA schedules,
  unqualified candidates and candidate/emitter mismatches fail closed.
- Independent CPU tests cover multiple dimensions, fractional f, signed energy
  weights, unchanged coefficient/input storage, exact DGEMM ABI/panel/pointer
  traces, checked overflow, no allocation, exact per-spin call order and the
  real solver's late-beta failure/publication behavior.
- A host harness executes the actual generated paired CUDA loop with synthetic
  thread indices. It checks exact ordered FP64/FMA results, an independent
  long-double mathematical oracle, mirrored stores, signed zero, scale/product/
  accumulator overflow and the separately checked product despite a finite FMA.
  This is scalar/order evidence, not CUDA device/concurrency qualification.
- Source-matched Release CPU build and 14 Gaussian native/CLI gates passed.
  GFN2 independent molecular energy/force, finite-difference, covariance and
  transaction endpoints passed (28 tests; CUDA-only tests skipped).
- The real CUDA fixture is extended for late-beta W-only failure, graph replay
  recovery and pre-existing sequence-error suppression. NVIDIA/CuMetal compile
  and real-device execution remain separate required qualification; no local
  device execution is claimed by this change.

## Rejected alternatives and remaining scope

`PreparedSymmetricProduct` implements a symmetric cross product, not this
weighted Gram. Binary `PreparedContractions` would change materialization or
paired publication. Neither is a drop-in replacement. New BLAS/CUDA providers,
reassociation, bucket ordering and launch changes are outside this migration.

The additional `method.mean_field_setup_cuda.weighted_projector` consumer is
explicitly outside this slice. Its setup/overlap reconstruction emission has not
been migrated here; this change does not claim complete retirement of all
weighted-projector emitters.

Review corrected the Gaussian CUDA candidate's former rectangular shape probe:
its concrete dense state stride did not describe the retained full eigenframe.
The square representative changes candidate metadata identities, while all five
frozen generated/expanded-source gates remain exact. A production-binding test
asserts the actual n*n state stride and virtual prefix representation.

This work does not resolve the separate 768-AO tblite discrepancy or reopen
#1879's parked performance campaign. Endpoint/device qualification remains
necessary; source equality is not a substitute for integration evidence.
