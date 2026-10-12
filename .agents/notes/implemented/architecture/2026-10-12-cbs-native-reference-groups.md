# Decision: native same-reference CBS component groups

Status: implemented
Date: 2026-10-12

## Problem

The initial CBS composition deduplicated equal method/basis identities but ran
HF, MP2 and CCSD(T) as independent native endpoints. That bounded sequential
ownership correctly qualified the composite, but repeated the same-basis RHF
solve and failed to share its integral owner across methods. Clearing caches
between independent methods was not evidence of scientific reuse.

This note supersedes the execution/ownership policy in
[the initial composition note](2026-10-12-mp2-assisted-ccsdt-cbs-ef.md).

## Decision

Introduce opt-in synchronous native companion execution, without changing the
legacy result/descriptor layouts or ordinary singlepoint work. MP2 can return
the stationary HF contribution from its canonical reference. CCSD(T) can
return both HF and MP2 from its canonical reference and exact interaction owner.
The Python CBS coordinator partitions by immutable basis identity, selecting
CCSD(T) as the correction group's primary method and MP2 for other groups.

CC's complete E+F finishes before its companions. Retire CC-only amplitudes,
denominators and redundant transformed blocks; preserve an explicitly admitted
six-block ERI symmetry cover (OVOV, OOVV, OVOO, OVVV, OOOO, VVVV), the physical reference,
the exact source and primary output arrays. MP2 energy uses the existing
generated CPU/CUDA energy kernels with borrowed ordered OVOV tiles. Its initial
force energy adjoint borrows the same block. Remaining MP2 response consumers
read compatible blocks through the native provider's validated eightfold-symmetry
borrowed routes, including mixed occupied/virtual slots and zero-padded tails.
All cached capacities are charged to each phase. A resource-only fallback drops
the complete optional cache and restores the qualified bounded native provider.
This does not introduce a host
energy oracle or a different correlation Hamiltonian.

HF forces use occupied diagonal stationary RHF Lagrangian weights and the same
bounded analytic derivative consumer as MP2. There is no second SCF or Z solve.
CUDA derivative consumers remain CUDA; rank-two/rank-four cotangent construction
and orchestration retain their existing host work. This is shared-reference
execution, not a claim that every buffer remains exclusively on the GPU.

## Invariants

- Companions are opt-in; ordinary MP2, RI-MP2 and CC endpoints preserve their work.
- Every component in a group uses the same canonical RHF frame and occupations.
- All simultaneously retained MO, output, source and executable-plan payloads
  are reserved before a companion phase. Planned and measured peaks are distinct.
- A memory-only fallback can retire optional MO/source/executable storage and
  reconstruct bounded tiles. Numerical, convergence and stationarity failures
  cannot enter that fallback.
- Group output buffers cannot alias and must request consistent force modes.
  Publish scientific outputs together only after every requested component
  succeeds. Rejecting or failing a replay revokes companion diagnostics.
- Different bases never share incompatible orbitals/integrals. Method-specific
  Lambda/adjoint and orbital-response solves remain separate.

## Rejected alternatives

Retaining multiple independent calculator caches still repeats same-basis SCF
and does not establish shared reference identity. Reconstructing MP2 correlation
from CC amplitudes would change the mathematical method. An environment flag
that always computes HF forces would regress ordinary MP2 consumers. A global
reference cache would require geometry/basis/control identities and cross-call
resource lifetimes absent from the synchronous native contract.

## Evidence and qualification

The pinned independent CBS fixtures check every constituent E+F as well as the
composite and independently reconverged directional finite differences. Native
ABI tests exercise rejected output descriptors, revoked diagnostics, unchanged
scientific buffers and subsequent recovery on the same prepared owner.

`CBSResult.execution_groups` exposes non-overlapping group ownership/timings;
component timings refer to their group and must not be summed. Native work
traces report companion energy source reads/transform FMAs, OVOV reuse and
actual borrowed force block reads, so
saved transforms are distinguishable from merely reduced owner construction.
Endpoint qualification artifacts are source/library matched and must not be
replaced by isolated microbenchmark claims.

## Revisit when

Large-basis MP2 response dominates the full CBS endpoint, or a bounded common
multi-RHS RHF response contract can share operator setup across CC and MP2.
Sharing more compatible transformed blocks requires explicit lifetime and peak
reservations; keeping an entire dead CC problem is not a free cache.
