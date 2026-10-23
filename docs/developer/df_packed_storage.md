# Packed and fitted-only DF value storage

This page describes the opt-in and resource-qualified
`packed`/`packed-single` DF value representations used by
[occupied CUDA exchange](df_occupied_cuda.md) and the corresponding
[force-response consumer](df_occupied_response.md).
A representation change does not alter metric/electronic method
identity or independently promote analytic forces.

## Packed value storage admission and execution

The explicit `create_cuda_density_fitting_jk_plan_from_source` overload accepts
`DfValueStorageOptions{DfPairStorage::SymmetricLower, rank_capacity}`. This route
requires a retained physical integral source and complete AO rows. Physical CUDA
SCF and composed Fock preparation accept the diagnostic selector
`GENERATIVEQC_DF_VALUE_STORAGE=auto|dense|packed|packed-single`. In HF SCF, unset/`auto`
keeps a fully resident dense owner when it fits; for a generated singleton RHF
source it promotes to `packed-single` only when dense would stream and the single
fitted owner fits the same value allowance. UHF, multi-item batches,
general-density/composed Fock and arbitrary public-tensor callers remain dense
unless packing is requested explicitly. The explicit native constructor does not
consult this ambient selector.

The physical selector runs before full host raw construction, including requests
with a zero value budget. Prepared metadata, device plans, composed Fock variants
and resource identities distinguish the representations. Changing the selector
invalidates ordinary cached preparation and is rejected by an admitted global
resource plan. Composed fixed-density Fock reserves no complete occupied U and
uses the exact bounded compatibility route.

Packed plans own separate immutable raw A and whitened B arrays in unit-weight
`[mu*(mu+1)/2+nu,Q]` order for `mu>=nu`. Raw generation writes lower rows directly;
native metric setup and one all-Q whitening preserve discarded raw directions.
J uses diagonal density entries once and off-diagonal `D_mn+D_nm`. Occupied K
projects directly into the existing full U layout when its rank fits the
reservation, then uses the existing Gram. Larger ranks and arbitrary densities
use exact bounded expansion. The common planner charges both immutable owners,
one `max(n*rank_capacity*a,n*n*q)` scratch buffer and two `n*n*q` buffers, plus
the existing source, metric, library and SCF reservations.

For a fresh single-system resident packed plan, automatic value mapping reuses
GPU-prepared normalized Hermite coefficients when they fit the unspent value
budget (at most 256 MiB). The existing fitted/occupied owners and panel sizes
stay fixed. Each raw output sums every ordered primitive product in FP64;
screening, metric rank and factor-ordered whitening are unchanged. Auxiliary g
and mixed Cartesian/spherical bases are supported. Explicit value mappings,
batches and capacity refusals retain the bounded generic source. The immutable
owner is charged to source storage and retires with the physical source;
geometry or basis changes require fresh preparation. `raw_hermite_values` in
the CUDA trace reports the actual generated output count.

The `packed-single` representation retains only one fitted tensor in the same
lower-pair order. It generates raw A once in bounded panels during setup, but
does not retain those panels. If every system's metric is full rank, it retains
`E = A Q diag(lambda^-1/2)` in metric eigen-coordinates. Setup projects raw A
before scaling, preserving weak-direction cancellation, and omits the large
rotation back to the original auxiliary basis. J/K contract the complete
auxiliary axis and are invariant under this orthogonal change. Force views and
final occupied projections carry the same immutable gauge and metric identity.
Rank-truncated or mixed-rank batches retain symmetric `B = E Q^T` throughout.
The `packed` and dense representations also retain symmetric B.

J/K, including qualified final physical Fock validation, reuse the fitted
tensor without claiming a raw-A owner. Validated canonical or
strictly reconstructed corrected occupied factors can use a separately budgeted
force response with `GENERATIVEQC_DF_RESPONSE_SPACE=occupied`.
`GENERATIVEQC_DF_OCCUPIED_RESPONSE_SOURCE=fitted` projects retained B in bounded
auxiliary panels when the source/metric identity matches and the metric is
full rank. The `raw` control instead regenerates raw A from the matching source;
`GENERATIVEQC_DF_SOURCE_PROJECTION=batched` batches its occupied projection.
Automatic response keeps its bounded general-density route. A constrained value allowance may
drop optional automatic occupied K scratch without dropping B; explicit
occupied requests still require their full reservation. The global small-HF
resource inventory currently supports `packed` but not `packed-single`.
The [single-owner qualification note](../../.agents/notes/implemented/performance/2026-09-25-single-fitted-df-owner.md)
records the numerical gates and complete-endpoint evidence.

For the qualified full-rank fitted occupied consumer,
`GENERATIVEQC_DF_OCCUPIED_METRIC=auto|retained-root|spectral` controls the second
metric transformation. `auto` and `retained-root` apply the matching second
root to the projected factors `S = C^T B C`: `U = S R`. For symmetric B,
`R = M^-1/2`; for spectral E, `R = diag(lambda^-1/2) Q^T`. The latter is
prepared in the otherwise unused full-rank inverse-matrix response workspace,
without a new allocation or a second retained AO tensor. Pure-J potentials,
occupied projections and bounded general-density panels all use the matching
map to physical auxiliary indices. This uses one GEMM into the existing disjoint
retained staging interval, with no final factor copy. `spectral` retains the
divide-before-rotate diagnostic; it requires two GEMMs for symmetric B and one
when E is already projected. Raw and rank-truncated
consumers retain their original spectral response, including discarded-direction
derivatives; they cannot borrow this root shortcut. The choice adds no device
allocation and grants no raw final-projection lease. Trace counters report the
actual root GEMMs, FLOPs, copy bytes and scratch allowance.

An exact final-K projection from a physical packed AO source additionally proves
`S_Q = C^T B_Q C` symmetric. The response stores its `r*(r+1)/2` independent
occupied pairs, with diagonals first, and applies the second metric root to
that smaller extent. The Coulomb potential still reads the diagonal trace.
Two lower-triangle SYRK products form the metric adjoint: diagonal occupied
pairs have weight one and off-diagonal pairs weight two. Their `beta=1`
updates preserve the existing Coulomb contribution before mirroring. Providers
without SYRK keep two full GEMM products over those same weighted pairs, with
full-product FLOPs reported; NVIDIA execution retains the two SYRK calls.

Only singleton RHF with its exact final-state lease, retained full-rank metric
root and packed physical source can take this route. Dense/nonsymmetric
fixtures, spectral and truncated-metric controls, UHF, corrected factors
without a lease and insufficient resident storage preserve their original
paths. Off-diagonal projection entries are averaged to remove FP64 reduction
asymmetry. After the compact Gram finishes, expansion uses the dead exchange
interval and restores the original disjoint staging layout; bounded derivative
panels and resource reservations remain unchanged. No in-place expansion is
permitted. Counters expose compact root elements, actual contraction FLOPs and
the expansion copy separately.

`benchmarks/compare_df_direct_endpoint.py` qualifies complete energy/force
endpoints for direct RHF and explicitly selected DF on identical geometry,
orbital basis, convergence thresholds and host thread counts. Each method has
its own independent GPU4PySCF oracle. Cold timing includes preparation; frozen
post-cold and post-move replays retain iterations and the API's optional Fock
count (`null` when unavailable); DF traces retain executed J/K work. `--interleave`
alternates dense-final/spectral, occupied-final/spectral and occupied-final/root
DF controls on the same density. Diagnostic traces are separate from clean
timings. These controls are same-binary ablations, not historical-build results.

`density_fitting_tile_plan(..., generated_source=True, pair_storage="packed")`
queries these capacities without allocating a tensor or creating a CUDA context.
Its `occupied` argument is the complete-U reservation and may be zero for a
bounded-only packed plan. The private `generativeqc_resource_df_packed_tiles_v1` ABI
reports both distinct factor owners and unequal scratch capacities through the
Python descriptor; existing dense v1/v2 queries retain their original ABI.
The complete Python HF candidate inventory exposes only `cuda-df-packed` for an
explicit packed request, charging both immutable owners and the actual scratch
capacities. Its existing limit of 16 orbital AOs and 128 auxiliary AOs still
applies; the standalone shape query is not subject to this inventory limit.

Force response uses a distinct `CudaDfPackedRawTensorView` with the plan's
metric/owner identity. Canonical factors may borrow the three actual scratch
capacities for occupied response. Missing/stale factors or insufficient
rank-squared storage use the bounded raw loader. Neither route regenerates raw
integrals or constructs a persistent full raw tensor. `GENERATIVEQC_DF_RAW_REUSE=off`
instead selects bounded source regeneration for diagnosis. Explicit seed/final
occupied overrides admit this resident source; automatic selection uses the
same resident work policy and checks the selected rank against both logical
rank capacity and actual retained projection storage. Other generated-source
exclusions remain. A final U lease is
published only if its full projection was retained;
the full-rank restriction and single-consumer invalidation still apply.

Explicit packed selection is retained for the measured warm/constrained domains;
unset and `auto` remain dense because bounded response and geometry rebuild can
regress substantially. The 384/1856 unequal case remains numerically unqualified.
The [retention decision](../../.agents/notes/implemented/performance/2026-09-17-packed-df-retention.md)
records domain evidence, validation and conditions for revisiting selection.

## Streamed pure-J contractions

Prepared restricted pure-J streamed sources automatically select native
Hermite charge/projection when the complete owner fits. Unset or
`GENERATIVEQC_DF_MD_COULOMB=1` enables this policy; `0` restores the previous
raw-value path and other values are rejected. The control is part of prepared-source
identity. Admission requires one streamed system, a full-rank metric, orbital
angular momentum through f, auxiliary angular momentum through g, and enough
value capacity for the complete optional owner. Native topology counts reserve
its exact device footprint before the planner sizes fallback panels. Device
and host storage remain included in the enclosing diagnostics.

An explicit `GENERATIVEQC_DF_MD_CHARGE_SPLITS=1..64` selects contiguous source
partitions for charge and its raised response. Unset retains one partition.
The count is frozen when the owner is created, including captured replay.
Partials reuse dead orbital primal-potential storage and reduce in ascending
partition order; each angular/derivative class independently falls back when
its complete partials do not fit. No allocation is added. Partitioning changes
FP64 summation order and remains an explicit tuning control, with numerical
and endpoint evidence in the
[decision note](../../.agents/notes/proposed/2026-10-23-df-md-charge-source-partitions.md).

The owner replaces raw three-center value regeneration with two vector
contractions and the existing metric solve. It remains unscreened; retained
integral sources remain available. Pure-J force response borrows the same
stream-bound owner to recompute auxiliary charge from its exact current density,
then applies the unchanged metric solve. A strict direct Hermite contraction
adds the three-center atom derivatives without emitting public-AO weight panels;
the original metric derivative remains in the response owner. Raised potential
storage is included in the value reservation. It does not reuse an earlier SCF
potential. Explicit panel/slice diagnostic controls retain
their existing paths. Unsupported domains and
optional capacity refusals use the existing bounded path. Device failures
propagate. The owner and captured graphs share the value plan's stream and
must retire before that stream is destroyed.

`tests/python/test_df_md_plan_cuda.py` exercises independent metric/J parity,
budget fallback and changed-input replay against the compiled core. Run it in
a finite Slurm GPU allocation with `GENERATIVEQC_TEST_DF_MD_PLAN=1` and an
explicit `GENERATIVEQC_LIBRARY`. Adding `GENERATIVEQC_TEST_DF_MD_FORCE=1`
checks complete Coulomb gradients against independent displaced libcint
energies, zero/changed density, borrowed-owner identity and failure propagation.
`tests/python/test_df_md_endpoint_cuda.py` adds independent complete SCF/force,
frozen replay and changed-geometry checks for the automatic and rollback paths.
Resident fitted tensors retain their existing selection. See the
[default decision](../../.agents/notes/implemented/performance/2026-10-22-default-df-hermite-and-xc-proposals.md)
for qualification and fallback rationale.
