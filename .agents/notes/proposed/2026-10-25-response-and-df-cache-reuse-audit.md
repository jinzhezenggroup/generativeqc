# Proposal: extend bounded source reuse to response and DF consumers

Status: proposed; source audit, implementation and endpoint qualification pending
Date: 2026-10-25

## Scope and source identity

This audit examines master `5bf5cf5007dbc7e9964e64d847ea90336fee1282` after
the LR host ERI cache in #2760 merged. The full-range host primitive is in
open PR #2762, inspected at `3cb888fa3f2991a1c663f1144c647801883f0c7d`.
The temporary-file extension is separate PR #2766. Those PRs' KS measurements
do not qualify RHF, orbital response, DF panels, or post-HF reuse.

The findings below identify executed source paths and resource/identity
contracts. They are not measurements of a new cache or claims of speedup.
The proposed production policy is automatic reuse within qualified admission
conditions, with explicit disable controls and bounded exact fallbacks.

## Priority and ownership

| Priority | Consumer | Confirmed gap | First bounded experiment |
| --- | --- | --- | --- |
| P0a | Exact RHF orbital/Z response | GPU values are optional; omitted values are recomputed on repeated signed-density actions | Add the full-range host primitive to the eager response owner |
| P0b | Ordinary RHF-SCF and public RHF/CPKS owners | RHF source leases have no host allowance; public response borrows another prepared Fock owner | Qualify each owner separately, including graph and borrowed-lifetime behavior |
| P1a | Source-backed streamed DF J/K | Raw A and fitted panels can be regenerated across consumers and iterations | Compare A backing with fitted B/E backing for the actual selected schedule |
| P1b | DF reference followed by MP2/CC | Same-model SCF and correlation can create separate source/metric owners | Borrow a validated immutable source/metric before extending factor reuse |
| P2 | Hessian and Lambda | Several relevant invariants already have bounded caches | Profile refusal/recomputation sites before adding another tier |

## P0: distinguish the response owners

[`RHFFrameResponseOptions::resident_jk_allowance`](../../../src/hf/rhf_frame_response.hpp)
already selects GPU ERI reuse automatically for relaxed, unscreened frames
with at least 64 AOs, subject to the complete numeric budget and an 8 GiB
ceiling. Explicit zero disables it. In
[`rhf_frame_response.cu`](../../../src/hf/rhf_frame_response.cu), `Owner`
constructs one immutable, unscreened Direct plan and optionally prepares
selected resident values. Repeated `potential` calls reuse it. This owner
uses ordinary stream submission rather than an RHF-SCF captured graph.

This is the most direct first integration point for #2762's strict full-range
host values. GPU-selected buckets should retain priority; host values and
both staging slots must fit the remaining **combined host/device numeric
budget**, available GPU memory and machine/job RAM headroom. A larger machine
RAM allowance cannot bypass `maximum_bytes`. Preserve the existing no-cache
path for small/short/screened cases unless separately qualified.

The final scalar residual uses the uncached action, and nuclear derivatives
have separate source work. Keep that independent recomputation: a cached
matvec must not validate itself, and zero-order ERIs cannot replace derivative
integrals. Preserve signed-density linearity and self-adjointness, compensated
FP64 contraction, physical stationarity and the exact residual gate. Optional
allocation refusal must release the entire optional tier before a bounded
retry; other numerical/device errors must propagate.

Two other owners need their own implementation and qualification:

- [`RhfResidentValues::prepare`](../../../src/scf/cuda/rhf_resident_values.cpp)
  forwards GPU policy to `DirectJkSourceLease` without a host allowance. Its
  owner has CUDA Graph borrowers. The existing host replay copies pageable
  data through pinned buffers and waits before slot reuse; simply passing a
  nonzero host ceiling into a captured schedule is not a valid integration.
  Compare an eager tiered path with the incumbent graph path, including lost
  graph benefit and graph-before-source retirement on failure.
- [`create_response_resident`](../../../src/api/c_api_fock.cpp) borrows the
  Direct plan from `PreparedFockPlan`; it does not create a private numerical
  ERI cache. `PreparedFockPlan` currently creates the ordinary Direct plan in
  [`fock_prepared.cpp`](../../../src/scf/fock_prepared.cpp). Public RHF/CPKS
  therefore cannot inherit a cache added only to `RHFFrameResponseOptions`.
  Put admission and lifetime in the prepared source owner, with borrower
  accounting, rather than mutating shared source storage from every solver.

The WB97M-V induced-Fock owner is another distinct consumer:
[`prepare_resident`](../../../src/dft/cuda_ks_wb97mv_response.cu) already
retains full-range/LR GPU values, but returns to recomputation when the full
required set exceeds its allowance. A DF primary keeps exact LR separate.
Its composite Hessian owner admits mandatory resources before optional
integrals. Preserve that ordering when testing partial GPU/RAM residency.

### P0 acceptance experiment

Use one source/binary and identical scientific inputs for Direct, GPU-only
partial storage, and GPU+RAM. Include a source that actually spills, a full-GPU
control, small/few-action controls, signed densities, and supported angular/
representation classes. Distinguish an imposed GPU ceiling from a naturally
capacity-limited case. A successful allocation is not evidence of consumption.

Report complete response and complete consuming force/Hessian wall times,
including setup, final uncached audit, derivatives and any retry. Count GMRES
iterations, provisional/exact actions, regenerated ERIs, cached reads, H2D/D2H
bytes and both numeric/resource peaks. Compare with an independent small
dense operator and molecular finite differences/reference forces. Exercise
changed geometry, source destruction, resource refusal and CUDA Graph refusal.
The proposed default remains contingent on those endpoint and lifecycle gates.

## P1a: cache the representation that removes the repeated work

The plan in
[`df_plan_internal.hpp`](../../../src/scf/cuda/df_plan_internal.hpp) already
supports resident A/B variants and a compatibility path with host-owned raw A.
`streamed` explicitly means the fitted tensor is regenerated; it does not mean
merely that a resident tensor is contracted in panels. Native generated-source
owners and compatibility host-tensor owners must be counted separately.

The residual regeneration sites are concrete:

- [`build_coulomb`](../../../src/scf/cuda/df_coulomb.cpp) visits raw source
  panels to form the auxiliary charge and again to project J. Its trace reports
  two raw passes, or one when K supplied the charge. The admitted Hermite
  producer is a separate vector path and must not be displaced solely to
  create a cache hit.
- [`build_exchange`](../../../src/scf/cuda/df_exchange.cpp) regenerates fitted
  row/column panels. For R row blocks and T output auxiliary blocks,
  [`df_streamed_k_panel`](../../../src/scf/df_streamed_k_policy.hpp) accounts
  for R*T raw-source passes. The diagonal already reuses its row panel.
  [`generate_metric_panel`](../../../src/scf/cuda/df_generated_tiles.hpp)
  regenerates A over the raw auxiliary axis and applies the metric transform.
- [`df_occupied_exchange.cpp`](../../../src/scf/cuda/df_occupied_exchange.cpp)
  has a different projected schedule with two retained device slots and
  explicit `shared_projected_regenerated_rows` work counters. Do not apply the
  dense-K R*T model to every occupied route. Its U=B*C projection depends on
  the current orbitals/density factor and cannot be retained across arbitrary
  SCF iterations under a geometry-only key.

For repeated K, retaining fitted B/E in RAM can avoid both recurrence and
whitening; retaining only A avoids recurrence but leaves metric work. For J
and source derivatives, raw A can have wider reuse. Compare the two using
complete endpoints and avoid duplicating both O(N^2 Q) tensors without a
measured benefit. Begin with the immutable generated-source plan's panel
provider, not separate caches in each J/K loop.

The auxiliary representation is a scientific lifetime contract. Current
full-rank routes may retain spectral `E=A Q lambda^-1/2`; other routes retain
symmetric B. Rank truncation, gauge, AO pair packing, metric cutoff, operator,
geometry and precision must match. Reconstructed A cannot recover discarded
metric directions. Signed response densities require the general linear
consumer, not a positive occupied-factor assumption.

[`PinnedResponsePanels`](../../../src/scf/cuda/df_gradient_bridge.cu) already
provides two event-protected pinned slots for gathering/uploading an existing
host raw tensor in a response path. It is a useful lifetime precedent, but is
not a persistent generated A/B backing cache for SCF. Duplicating that staging
alone will not eliminate native source regeneration.

Compare same-plan resident, streamed Direct and streamed RAM arms. Retain
source passes, actual row/auxiliary tile shapes, whitening GEMM work, cache
coverage and transfer bytes alongside full cold/warm/moved E+F. Include
general signed densities, occupied-factor K, J-only, spin, truncated metrics,
packed/dense layouts and budgets too small for either pinned slot.

## P1b: cross-stage sharing is conditional, but the gap is real

In [`rccsd_method.cpp`](../../../src/methods/rccsd_method.cpp), the full-DF
reference branch calls `DFReference::solve` with the correlation auxiliary
basis and retains that owner in `RccsdNativeState`. It then calls
`build_problem`, which calls `build_df_source_cuda` without borrowing the
retained DF plan. That builder creates another integral source and metric
plan in [`df_source_cuda.cu`](../../../src/cc/df_source_cuda.cu). The retained
SCF capacity is already charged, so duplicate owners can also reduce the
correlation workspace available. This is a concrete same-auxiliary candidate.

The ordinary exact-RHF reference followed by DF correlation has no SCF DF
factors to share. Separate generic SCF/correlation requests may use different
JK-fit and correlation-fit auxiliary bases even with the same orbital basis.
Neither should be counted as an avoidable same-model duplication.

[`DFReference`](../../../src/hf/df_reference.hpp) already shares its plan
between SCF and its molecular response. Extend an explicit immutable lease
from that owner, or extract a shared source/metric owner, rather than adding
a process-global cache. First reuse source metadata, the raw source and the
matching metric/eigensystem; then qualify resident A/B/E or final occupied
projection reuse where each consumer accepts that representation. Inspect
the separately constructed source in
[`ri_mp2_cuda.cu`](../../../src/posthf/ri_mp2_cuda.cu) as another consumer;
its low-memory virtual-block source passes are a different amplification site.

Orbital basis equality alone is insufficient. Require exact geometry, full
auxiliary basis and normalization, public AO layout, radial operator,
precision, metric threshold/rank/gauge, device and stream dependencies.
MO factors additionally bind the accepted orbital frame and occupation/frozen
core partition. Source-response cotangents currently authenticate a nonzero
`factor_basis_identity`; sharing must preserve that lineage through reverse
response, not manufacture a second identity for the same borrowed storage.

Measure the entire SCF+MP2/CC energy/force endpoint, separating reference,
AO source, metric, MO transformation, correlation and response phases. Count
source and factorization builds and charge shared storage once, while retaining
all genuinely separate scratch and host payloads. Verify same-model reuse,
mismatched auxiliary/rank/frame rejection, energy/forces and failure teardown.

## P2: existing invariants change what remains worth caching

- [`LambdaOptions`](../../../src/cc/lambda_response.hpp) already defaults
  `df_auxiliary_reduction` and `df_core_reuse` to true. The
  [`DF Lambda owner`](../../../src/cc/df_lambda_cuda.cu) prepares an immutable
  frontier once, executes only its dynamic part per action, and falls back
  when the complete optional frontier does not fit. Host retention is a
  possible fallback experiment, not a missing first-level cache. Its key must
  include the immutable Problem/T owner; Lambda seeds remain dynamic. Retain
  the independent expanded audit graph.
- The native
  [`WB97M-V Hessian owner`](../../../src/dft/cuda_ks_wb97mv_hessian.cu) already
  admits a bounded small CPKS operator matrix and frozen exact directional
  columns. [`cuda_cpks.cu`](../../../src/response/cuda_cpks.cu) authenticates
  the reference/provider and checks a fresh physical residual. Cache misses
  from the dimension-squared operator bound are where P0 ERI reuse may help.
  Building a dense operator for large spaces can cost more than matrix-free
  solves; do not raise its cap simply because RAM is available.
- The
  [`ExactGeometryResponseCache`](../../../python/generativeqc/_geometry_response_cache.py)
  stores successful density-only force components under an exact density key
  inside a geometry/operator/grid owner. It does not cover density-dependent
  iterates, weighted-density/Pulay terms or arbitrary Hessian directions.

Profile each remaining recurrence or invariant cut and distinguish immutable
geometry data from density-, amplitude-, seed- and direction-dependent data.
Promote a new tier only when it removes measured source work and improves
the complete consuming endpoint within the original numerical gates.

## RAM versus temporary storage

Start these experiments with RAM. The LR disk result in #2766 cannot establish
a full-range, DF-panel or Lambda crossover; that campaign's full-range exchange
control was slower with disk. Temporary-file backing remains a separate
per-consumer experiment after RAM capacity is exhausted. Keep local-filesystem
capacity limits, descriptor-owned cleanup and exact failure semantics if it is
tested; do not make every owner inherit automatic disk spill from LR.

## Audit checks

The existing host-only schedule, geometry-cache and mocked DF-owner lifetime
checks pass: **54 passed**, using this source revision. These exercise actual
compiled panel schedules, independent row-coverage/storage simulation, exact
density invalidation, resource refusal and owner teardown. They do not exercise
a new RAM tier or a real CUDA device.

The first broad invocation also selected
`test_generated_positive_budget_reduces_raw_passes_monotonically`, which could
not load a native library in this fresh audit worktree. The follow-up limits
the selection to the 54 host-only checks; that native-library-dependent case
is unqualified here, not a passed or removed test. Logs and verified compiler
launcher information are retained in ignored `.artifacts/cache-audit/`.

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_df_streamed_k_policy.py \
  tests/python/test_df_projected_exchange_schedule.py \
  tests/python/test_geometry_response_cache.py \
  tests/python/test_df_source_state_lifetime.py \
  -k 'not generated_positive_budget_reduces_raw_passes_monotonically'
```

Standalone C++ probes should resolve their compiler through a verified cache
launcher. All local Markdown targets and `git diff --check` also pass.

## References

- [Performance qualification policy](../../../docs/maintainer/performance_engineering.md)
- [Existing Hessian reuse](../implemented/performance/2026-10-17-bounded-wb97mv-hessian-batching.md)
- [Existing Lambda invariant frontier](../implemented/performance/2026-10-10-lambda-invariant-frontier.md)
- [Spectral DF representation](../implemented/performance/2026-10-23-resident-spectral-df-factors.md)
- [LR RAM cache PR](https://github.com/jinzhezenggroup/generativeqc/pull/2760)
- [Full-range RAM primitive PR](https://github.com/jinzhezenggroup/generativeqc/pull/2762)
- [Temporary disk spill PR](https://github.com/jinzhezenggroup/generativeqc/pull/2766)
