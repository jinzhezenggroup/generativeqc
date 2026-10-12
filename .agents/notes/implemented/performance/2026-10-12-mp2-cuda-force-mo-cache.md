# Decision: optional CUDA MP2 force primal reuse

Status: implemented
Date: 2026-10-12

## Problem

Sharing same-basis CBS references and the correction group's CC primal blocks
does not remove repeated transformations in the larger MP2-only force group.
The energy adjoint, orbital RHS, every response action and relaxed stationarity
replay consume compatible ERIs in the same canonical RHF frame. Repeated exact
AO-to-MO transforms dominate even when their source and contractions are CUDA.

This extends [the exact CUDA RHF source handoff](2026-10-12-mp2-cuda-reference-source-ef.md)
and uses the validated borrowed-block routes introduced by
[native CBS groups](../architecture/2026-10-12-cbs-native-reference-groups.md).

## Decision

For conventional CUDA E+F with a device-readable exact reference source,
optionally construct OVOV, OOVV, OVOO, OVVV, OOOO and VVVV once. The existing
provider's eightfold ERI symmetry routes cover all occupied/virtual orientations,
including mixed selectors and zero-padded tile tails. The force energy adjoint
borrows OVOV; the remaining force and response consumers borrow the six blocks.
CPU, energy-only and density-fitted routes do not construct this cache.

Charge retained vector capacities to every producer and complete force consumer.
Admit the optional cache only below one quarter of the remaining numeric phase
budget, leaving response and derivative headroom. Preflight the widest admitted
basis-axis CUDA tile with the full cache plus producer scratch. Construct blocks
sequentially while subtracting already retained capacities from producer budgets.
Keep the construction peak as a completed phase even after the cache is retired.
These are conservative planned peaks, not measured process/device allocation peaks.

A construction memory refusal drops the incomplete optional cache. A complete
force memory refusal drops the cache and retries the original bounded force path
using the same physical reference/source. Only `length_error` and `bad_alloc`
trigger these retries. Numerical, stationarity, convergence and backend failures
propagate. Source/executable memory fallback remains independently bounded by
the existing endpoint phase owner. No production reference package is introduced.

## Boundaries and observability

Cache transforms run on CUDA, but returned MO vectors and existing orchestration
retain host storage. VVVV storage scales quartically; quarter-budget admission
is not an arbitrary-basis scalability claim. Method-specific adjoint/response
algorithms and derivative contractions are unchanged. Different bases cannot
share their incompatible canonical frames.

`mp2_force_mo_cache_bytes` reports positive retained capacity on a completed cache
route and zero on bounded replay. `borrowed_mo_blocks` counts actual cache reads;
the `native_mo_blocks` journal retains actual AO reads and transformation summands,
including discarded attempts. Qualification compares whole oracle-accepted E+F
endpoints, not isolated kernels, and must include constrained-memory parity.

`GENERATIVEQC_MP2_FORCE_MO_CACHE=0` independently restores force reconstruction
for controlled qualification. `GENERATIVEQC_MP2_FORCE_SOURCE_TILING=0` also
disables cache construction so its shell-sized diagnostic contract stays intact.

## Companion lifetime correction

The physical reference's `numeric_capacity_bytes` records the completed SCF
and export phase peak, not just its detached matrix payload. Initially the HF
companion force admission charged that historical peak again while the outer
phase still reserved the executable, forcing a premature exact-source fallback
in the physical H2/TZ group. Charge actual reference vector capacities instead;
the source/outer phase owns the retained executable reservation. Preserve the
historical peak only in completed endpoint diagnostics. A native regression
inflates the historical peak above the force budget, checks actual live admission
and compares companion forces with independently executed ordinary HF forces.

## Qualification evidence

Source/library-matched Slurm E+F endpoint records and acceptance details are
retained in [the benchmark snapshot](../../../../benchmarks/results/ccsdt-cbs-native-reuse-20261012/README.md).
The same-library cache-disable comparison improves complete H2 DZ/TZ and water
algebra replays by 21.3x and 16.2x respectively, with actual source/read/transform
counts and independent per-component/composite oracle gates. These are qualified
small-domain observations, not arbitrary-basis speed guarantees.

Same-basis grouping itself improves the water endpoint but regresses H2 by 28%
against cache-enabled independent coordination. The generic stationary HF force
consumer remains more expensive than ordinary HF's specialized force route in
that case. Do not hide this cost behind reduced SCF/MO work counts; revisit its
qualified derivative backend or fusion before claiming universal grouping wins.
The disabled-source F-shell compatibility replay exceeds its 15-minute allocation;
its default cache-enabled endpoint is separately independently oracle-qualified.

## Revisit when

Larger virtual spaces make VVVV storage or construction dominate. Prefer a
bounded response-aware primal store or qualified common multi-RHS operator setup
over simply raising the cache budget or claiming all CBS derivative work is shared.
