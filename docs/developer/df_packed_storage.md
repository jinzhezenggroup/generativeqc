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

The explicit `packed-single` experiment retains only fitted B in the same
lower-pair order. It generates raw A once in bounded panels during setup, but
does not retain those panels. J/K, including qualified final physical Fock
validation, reuses B without claiming a raw-A owner. Validated canonical or
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
metric transformation. `auto` and `retained-root` apply the plan's immutable
symmetric inverse root `X` directly to the projected factors `S = C^T B C`:
`U = S X`. This uses one GEMM into the existing disjoint retained staging
interval, with no final factor copy. `spectral` retains the two-GEMM
eigenvector/scale route for independent comparisons. Raw and rank-truncated
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
