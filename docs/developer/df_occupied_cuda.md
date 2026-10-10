# Occupied-factor CUDA exchange and force response

This page is the entry point for the occupied RI-K work policy,
resident/executed exchange contraction, seed and final state selection,
and SCF diagnostics. The [CPU mathematical contraction](df_occupied_exchange.md)
owns the normalized RI-K equation; [strict final-state selection](df_final_state.md)
owns determinant and physical-Fock acceptance. For device response and
packed-source layout see [occupied force response](df_occupied_response.md)
and [packed DF storage](df_packed_storage.md).

## Exchange work model and admission

`GENERATIVEQC_DF_EXCHANGE=auto` (also the unset default) selects occupied RI-K
using the shared work policy in `src/scf/df_exchange_policy.hpp`. For `n`
orbital AOs, `a` auxiliary AOs and occupied rank `r`, dense exchange requires
`4*a*n^3` FLOPs and occupied projection plus a full Gram requires at most
`4*a*n^2*r`. Auto requires at least a twofold arithmetic reduction (`0 < r <=
floor(n/2)`). This margin is a workload heuristic, not a promise of a twofold
latency improvement on every device. It uses FP64 contractions and has no GPU
product-name, architecture, exact AO/rank or equal-basis gate.

Resident execution additionally requires a single RHF system, a non-streamed resident
plan, full AO rows, reserved factors, native BLAS index bounds and sufficient
actual projection capacity. Host-raw plans need full auxiliary scratch;
explicit packed-source plans need retained raw storage and enough rank capacity.
Other generated-source layouts, UHF, batches, zero/high rank and insufficient
storage retain their checked fallback. `dense` and `occupied` remain explicit
comparison overrides. All factor, density and final-state checks still apply.
The [selection decision](../../.agents/notes/implemented/performance/2026-09-18-general-occupied-df-policy.md)
records the work model, validation and performance limitations.

## Generated streamed occupied exchange

Generated streamed singleton RHF value execution has a separate compiler-owned
schedule in `generativeqc_compiler.method.df_exchange_schedule`. When the metric is
full rank and the existing four scratch buffers can reduce source work, project
raw AO rows into occupied space before metric whitening. Two buffers retain
projections; raw input and metric projection use the other two. In triangular
K, alternate the retained slots between output rows and visit columns in
descending order. This consumes the preceding row's projection before its slot
is overwritten. For `b` balanced blocks of width `h`, the generated AO-row count
is `n + h*(b-1)*max(0,b-2)/2`; one or two blocks need exactly one raw tensor pass.
The explicit full-matrix traversal retains its `n*b` row count. Admission compares
these counts against the dense fallback, without increasing buffer capacity.
The value schedule grants no final-state or force-response projection lease.

## CUDA exchange contraction and occupations

For one spin, `D = w C C^T` with canonical occupation w=2 (RHF) or w=1 (UHF).
On a full resident plan, project the existing pair-major tensor directly:
`U[mu,i,Q] = sum_nu B[mu,nu,Q] C[nu,i]`. One strided-batched GEMM builds U;
a Gram contraction over `(i,Q)` produces `K = w U U^T`. The triangular route
uses SYRK and mirrors its result; providers without SYRK use full GEMM.
Generated and bounded panels retain `U = C^T L_row^T` followed by
`K_row,column += w U_row^T U_column`. Host `OccupiedDensityFactor` snapshots
already contain `sqrt(w)*C` in row-major AO/occupied layout and use w=1 in the
CUDA product. Device SCF retains column-major C and applies w once at the
second product. Existing Fock consumers retain their RHF -1/2 and UHF -1
exchange prefactors. Empty spin rank produces zero K without a GEMM.

Dense resident K keeps square per-Q products but divides Q into half-capacity
panels. Gathered B and per-Q contributions occupy disjoint halves of one
buffer; the other holds the density projection. Raw A stays untouched. The
running auxiliary sum continues across panel boundaries in its original order;
odd auxiliary counts have a bounded tail. The `flat` ablation instead projects
`B_mu D` and contracts the combined `(nu,Q)` dimension, using one scratch
tensor. Both identities preserve nonsymmetric diagnostic B and D.
`GENERATIVEQC_DF_RESIDENT_EXCHANGE=legacy|full|flat|auto` selects original J/K,
resident full-Gram K, resident flattened dense K, or the default panel-dense
and triangular occupied route. The plan freezes this policy; changing it
rebuilds captured SCF work.

Packed resident plans use a compiler-generated triangular FP64 Gram when
`n >= 384`, the reduction `a*r >= 32768`, and at most 64 slices of 32768
reduction entries fit the existing `exchange_intermediate` capacity. Each
32-by-32 lower output tile produces disjoint partials; a second kernel sums
them in ascending slice order, applies the occupation weight once and writes
both triangles. Padded AO/reduction tails are explicit. This preserves U for
its final-response lease, adds no allocation or atomics, and supports capture
on the existing stream. Short, constrained, nonpacked and full-matrix routes
keep their BLAS provider. Errors propagate without retrying another algorithm.
Trace counters include partial bytes, slice count, reduction elements and
executed FLOPs including padded diagonal/edge arithmetic.

The earlier [split GEMM experiment](../../.agents/notes/rejected/2026-09-17-split-occupied-gram.md)
remains rejected: it computed both triangles and lost its kernel saving to
extra SCF iterations. The generated triangular provider and its complete
endpoint qualification are described in the
[96-atom optimization decision](../../.agents/notes/implemented/performance/2026-10-03-df-symmetric-occupied-products.md).

The same plan supports resident tensors, generated panels and compatibility
host-backed tiles. Full AO panels follow #282's capacity rebalance and reuse
the diagonal T before any column replacement. Tight row traversal retains
bounded column regeneration; its generated tile trace exposes that work.
The T, column and output matrices fit the original three/four tile buffers
because occupied rank never exceeds nbf. No additional three-center tensor is
allocated. Compatibility host uploads drain before overwriting pageable
staging; generated plans stay on their existing stream and permit capture.

The native fixed-density API checks the immutable factor's exact density
witness, spin, orbital/density generations and a process-unique physical plan
identity. Rebuilding geometry, basis or metric policy creates a new plan
identity. Matching dimensions or caller-provided generation labels alone
cannot authorize use. A missing or incompatible factor runs dense K for that
item while compatible neighbors retain factorized K. Factor upload borrows
the existing density-transpose staging; no allocation is needed for host B.

## SCF convergence and physical acceptance

The production CUDA direct and DIIS-enabled DF RHF/UHF loops share the FP64
stopping policy in `scf_convergence_policy.cuh`. They require a finite preceding
energy and all three tests:

- `abs(E - previous_E) < energy_tolerance + 16*epsilon*max(1, abs(E), abs(previous_E))`;
- `density_step_rms < density_tolerance`; and
- `max_abs(FDS-SDF) <= min(1e-8, density_tolerance)` for the current physical
  pre-DIIS Fock, including both spin channels for UHF.

The energy guard accounts for FP64 contraction/reduction granularity. It never
relaxes density, physical residual, final determinant or independent energy/force
validation. Reported energy changes remain raw absolute differences. Direct can
restore an energy baseline only for its exactly matched warm density/geometry;
qualified singleton occupied DF can likewise restore its validated density,
occupied factor and compatible energy baseline. Other DF seeds rebuild the
baseline and therefore require at least two iterations.
Equal acceptance does not imply equal iteration or Fock-build counts. Low-level
no-DIIS compatibility calls lack iterative residual storage and still require
strict final-state validation; host numerical recovery retains its stricter
unguarded energy comparison. Neither compatibility route licenses skipping
final validation. Coarse direct mixed-precision stages still require subsequent
FP64 target refinement before publication.

## Seed factorization and warm replay

Every device SCF invocation starts with one seed iteration. Imported and unmatched warm
densities have no trustworthy orbital factor, but a checked algebraic factor
can replace its dense K. `GENERATIVEQC_DF_SEED_EXCHANGE=dense|factor|auto` controls this
choice; `factor` enables guarded factorization and `auto` uses the same
resident capacity and occupied-work policy as SCF.

Factorization requires an occupied-SCF singleton RHF plan with existing factor
capacity and either a qualified resident layout or the streamed value schedule
above. The experimental
packed resident constructor below also supports the explicit `factor` override.
UHF, batch and other unqualified generated-source plans keep dense seeds.

The seed reuses the compact GPU eigensolver and transient Fock/eigenvalue
scratch to form `L = V sqrt(lambda)`, then builds occupied K with weight one.
Eigenvalues below `-1e-13` reject the seed. Values at most `1e-13` may be
discarded only if their combined Frobenius norm is at most `1e-12`; retained
rank cannot exceed the reserved occupied rank. A full `L L^T` reconstruction
must match both triangles of the input with maximum error at most `1e-12` and
RMS error at most `1e-13`. No canonical identity is assigned to this factor.
Numerical rejection returns to dense K; CUDA failures propagate. The ordinary
path adds two explicit stream synchronizations and downloads the spectrum,
solver status and two reconstruction scalars. It allocates no new persistent
device buffer. `GENERATIVEQC_DF_SEED_VERIFY=1` additionally compares candidate and
dense K for the identical density under max/RMS gates `1e-10`/`1e-11`, restores
candidate K, and records K/Fock errors in the progress journal. This intrusive
validation must be disabled for clean endpoint timing.

For DIIS-enabled singleton RHF, a completed strict endpoint may retain two
immutable host records: the latest returned density/frame and a matched frozen
input. Admission compares every density, Hcore, overlap and orthogonalizer entry,
occupation, nuclear energy and immutable plan/source identity. Only this exact
match bypasses repeated seed normalization. The occupied factor is restored
into existing storage and the first iteration still rebuilds J/K, solves the
Fock problem and applies every convergence gate. Strict final Fock, determinant
and force validation still run; one iteration is an outcome, never a forced
count or a zero-work replay.

The compatible energy baseline uses existing physical final J/K with the SCF
Fock-assembly and energy-reduction kernels. This performs no additional J/K
build. The record is staged before response scratch consumers and published
only after the complete endpoint succeeds. New solve attempts revoke published
readiness before input checks; stale tokens and corrected final frames cannot
publish an exact retained entry. Host retention and temporary snapshot storage
are bounded by 64 MiB, add no device allocation, and fall back on allocation
failure. UHF, batches, no-DIIS, unmatched geometry/data, and unqualified final
Fock owners retain the existing path. `GENERATIVEQC_DF_WARM_REUSE=0` (or benchmark
`--disable-warm-reuse`) provides the same-binary control. Work traces distinguish
warm-factor, algebraic-factor and dense seed iterations and include retention
transfers and the separate energy reduction.

## Final physical exchange

`GENERATIVEQC_DF_FINAL_EXCHANGE=dense|occupied|auto` independently controls final
physical Fock evaluation. Unset/`auto` selects occupied K whenever the same
resident, single-fitted-B or streamed work/capacity and provenance gates qualify it; `occupied`
keeps the explicit comparison override, and `dense` keeps the diagnostic dense
fallback. For singleton generated streamed or `packed-single` RHF, qualified exact final factors
use occupied K automatically. A bounded strict-finalization correction may
reconstruct an algebraic occupied factor after the owner/model/occupation and
generation-advance checks. Reconstruction must pass the same spectral,
discarded-norm and full-density maximum/RMS gates as algebraic seed exchange.
These private final projections are temporary and never
grants a force-response projection lease; failed qualification, insufficient
capacity or missing source storage retains the original bounded dense fallback.
The exact retained-factor route requires
an exact current final-state token, matching device generation/solver status,
and entry-for-entry equality of the supplied and retained densities. It uses
the full retained coefficients with RHF weight two. The strict physical Fock
and final-state gates still run. Other resident layouts keep dense K for
correction generations. Missing/stale provenance and failed reconstruction
retain the checked dense fallback. No previous physical Fock is reused.

The seed iteration stores
the exact C that constructs the next D before the convergence update. Each
spin owns `batch*nbf*max_occupied` values plus generation controls. Inactive
systems retain both density and factor; eigensolver scratch is never borrowed
as persistent C. Occupation/policy changes rebuild captured GEMM shapes.
The first seed iteration counts against the original iteration limit, even
when that limit is one. Generation checks run before occupied K and at final
readback; a stale generation rejects the device result and preserves the
caller's established numerical recovery. Force evaluation receives only the
validated converged density and keeps its full metric/center/Pulay response.

## Resources and observability

Native and common resource ledgers reserve two full AO matrices for spin
factors plus generation flags only for explicit occupied selection or a known
RHF occupation accepted by the shared work policy. Unknown references, UHF,
zero/high rank and ineligible generated layouts keep dense reservation.
The method passes this occupation through the shape planner and native plan
constructor; the versioned Python query accepts `rhf_occupied` explicitly.
If optional factors would force a host-raw plan into streaming or make the
budget infeasible, auto retains the original dense plan. Packed plans can also
drop the optional SCF charge while retaining their explicitly requested U
capacity. Runtime allocation uses the actual occupied ranks. Dense mode retains
its previous minimum-budget and residency boundaries. A native plan freezes this reservation
at creation and rejects occupied SCF before allocation if it reserved only dense
storage. Ordinary prepared batches rebuild the value/SCF plan on policy changes,
retaining their geometry response cache. Batches with a global `ResourceBudget`
freeze `GENERATIVEQC_DF_EXCHANGE` in the resource identity: changing it after preparation
requires preparing a new batch and is rejected before native execution. The
fixed-density factor API needs no additional allocation and still borrows the
existing tiles independently of the SCF reservation.

`ri_k_occupied` traces report factor bytes, rank, projection/exchange products,
GEMM dimensions, intermediate bytes and panel hits. Captured records describe graph
construction; `occupied_scf_provenance` separately reports executed iteration
counts, dense seeding and final generation validation. Uninstrumented complete
endpoints remain the performance selection gate. The
[density exchange seed note](../../.agents/notes/implemented/performance/2026-09-16-density-exchange-seed.md)
records the factorization, final-state audit and qualification rationale.

## Resident raw ownership and response storage

The exact retained raw tensor and borrowed response scratch ownership
are specified in [occupied force response](df_occupied_response.md#resident-raw-ownership-and-response-storage).
Value-plan residency does **not** authorize an unrelated response
consumer to reuse raw pointers or omit metric directions.

## Exact occupied force response

The mathematical weighted projection, full/truncated metric scope,
generation checks, packed derivative pairs and bounded resources are
specified in [occupied force response](df_occupied_response.md#exact-occupied-force-response).
See [DF derivatives](df_derivatives.md) for the generated scientific
weight-consumer contract.

## Compact SCF DIIS and solver timing

`GENERATIVEQC_DF_DIIS_DOTS=auto` forms deterministic partial residual dots in blocks
of 4096 elements, then reduces the partials in the existing small DIIS solve.
The completed residual-product temporary supplies the partial storage. No
allocation, atomic dot accumulation, or host synchronization is added.
Physical slots are validated against the chronological circular history,
including a short window after dependent-history retirement. Inactive systems
and unpopulated entries remain untouched. Small reservations fall back to a
warp reduction; `serial` retains the original dot order for comparison.

Normalization, pivot threshold, chronological retirement, Fock construction
and physical convergence gates are unchanged. Floating-point reduction order
can change the iteration branch, so endpoint evidence must retain both the
starting checkpoint identity and actual update counts. The diagnostic policy
is frozen with captured work and all response/exchange controls participate
in global resource-plan identity.

Compact eigensolves keep the existing cuSOLVER provider and retained host and
device workspaces. `compact_diis`, `diis_residual_products`,
`diis_history_update` and `compact_eigensolve_provider` expose their separate
GPU/host scopes. A provider host call can wait for preceding queued DIIS work;
subtracting its GPU events from host duration does not measure CPU eigensolver
work. Component tracing introduces fences and must be qualified with separate
host-only or Nsight timelines and clean complete endpoints.

The [resident DF dataflow note](../../.agents/notes/implemented/performance/2026-09-16-resident-df-dataflow.md)
records the algebra, rejected variants, numerical gates and retained evidence
for these paths.

## Experimental native packed values

The packed raw/fitted owner, capacity admission and exact fallback
semantics live in [packed DF value storage](df_packed_storage.md).
The explicit packed constructor and diagnostic controls are not
unrestricted public capability claims.

```{toctree}
:hidden:
:maxdepth: 1

df_occupied_response
df_packed_storage
```
