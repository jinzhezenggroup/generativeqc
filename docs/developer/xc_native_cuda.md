# Native semilocal XC device boundary

`dft/cuda_xc.hpp` supplies the ordinary-stream fixed-density LDA/PBE RKS/UKS
component for #162. It consumes current device density matrices and returns
device XC energy/electron totals, potential matrices and a numerical-error
flag. `dft/cuda_ks.hpp` composes it into native ordinary-stream SCF; the
registered method adapter exposes CPU/CUDA single-system and native prepared
ragged energy execution. The public #203 method resource plan accounts for
preparation and execution with one persistent ledger; see
[resource planning](../maintainer/resource_planning.md).

## Ownership and data movement

`cuda_xc_layout` returns an exact explicit device-memory request. The method
ResourcePlan supplies that arena; XC performs no device allocation and accepts
no separate user memory budget. `CudaXcPlan` borrows the arena and stream, which
must outlive it. Its destructor drains the stream before the caller releases
either resource. Concurrent calculations use independent plans and arenas.

The molecular quadrature remains an explicit immutable input with host exports
for derivative/reference consumers. Native CUDA preparation also retains the
exact generated points and weights on their source device. Device-fused XC
validates that lease against its current device and point count, keeps the
lease's lifetime token, and borrows its pointers directly. Only packed basis
data is uploaded during XC setup; points and weights are not re-uploaded and do
not occupy a duplicate region in the XC arena. Host-unfused execution retains
the existing host-only behavior. Quadrature preparation and its single host
export remain part of complete calculation cost.

Every `enqueue` consumes a strictly newer density generation, invalidates the
old result view, clears its outputs/error and rebuilds the full XC contribution.
It does not allocate, transfer host data or synchronize. Device density producers
must enqueue on the same stream, or establish an explicit event dependency.
`read_scalars` downloads only energy, both spin populations and the numerical
status. `download_potential` is an explicit user/reference output operation.
Transfer counters distinguish setup H2D from these requested D2H operations
and count their synchronization boundaries. The surrounding method must also
account for its density input, one-electron setup, J provider and final outputs.

The geometry, basis, grid and functional are immutable within a plan. Changes
require a new plan. The currently supported compositions are exactly
LDA_X+LDA_C_PW and PBE_X+PBE_C, with the
[versioned SCF point-domain policy](xc_scf_domain.md). No meta-GGA or force
capability is implied by these interfaces.

## Numerical path

Strict-FP64 integration retains three published totals and three private
rounding-correction words for XC energy and the two spin populations. One full
32-thread warp reduces each channel with lane-local compensated sums and a
fixed tree that merges both words. Corrections survive point-batch boundaries;
only the final fold publishes the existing three-double result. All three low
words are cleared with the totals at each enqueue and charged by the exact
arena request (24 extra bytes), with no new full-grid storage or device
allocation. Dense, compact, indexed and retained fallback consumers share this
reduction. Existing FP32 contraction schedules keep their original accumulation.
The point algebra, Vxc matrix and SCF convergence thresholds are unchanged;
this is an accuracy improvement, not a bitwise-reproducible SCF contract.

### Budget-Fock matrix products

The shared physical density/potential consumer can select FP32 inner products
independently of point algebra and total summation. Application Fast-Fock
admission prepares this capability before evaluation; ordinary strict and
response consumers retain their strict defaults. Selection uses executable
layout/ingredient capabilities, not functional names. Existing indexed,
compact and ragged schedules are retained.

Density AO columns use a per-column binary scale before their FP32 dot
products, restoring that scale into FP64 physical storage afterward. This
preserves representable tiny tails without a density floor or a hidden strict
recomputation. Potential panels also retain FP64 storage; their tile-local
products and sums narrow, while inter-tile scatter and compensated point totals
remain FP64. The precision census therefore reports separate matrix-product
reduction signatures rather than inheriting the point evaluator's dtype.

The existing one-sided diagonal schedule is available to both scalar types.
Its FP32-only operand guard prevents half-panel doubling overflow and otherwise
retains the original two-legged FP32 contraction; the strict schedule and its
guard are unchanged. Invalid arithmetic remains an explicit numerical failure,
not a clipped density or a successful Fast result after strict recovery.

### Experimental automatic AOT plumbing

`GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON` builds static ordinary E/vxc
entries from the common structural automatic registry. It is off by default;
the disabled registry returns no automatic entries. Enabling this internal
build option **does not admit public automatic CUDA KS or force methods**.
Unsupported products continue to fail explicitly rather than stage CPU science.

For bounded packages, set `GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS` to a
semicolon-separated list of exact structural registration names, for example
`"LDA_C_VWN_4;GGA_X_APBE;MGGA_X_LTA"`. Both CUDA and the automatic AOT option
must be enabled; a selected package cannot use the CPU-only compiler fallback.
An empty list preserves the full-catalog opt-in behavior, while the AOT option
remains off by default. Selected entries retain their original codes,
mathematical/source identities and ordinary/batched providers. Omitted entries
have no executable dispatch case and return a null static descriptor.
Unknown/unrepresentable names fail before generation writes outputs; duplicate
or reordered requests do not change structural shard order. Regeneration
rewrites all shards, removing omitted dispatch cases from shrinking packages.
Selection controls installed artifact content, not scientific admission,
public capability promotion, executable alias reuse or a complete persistent
binary-cache dependency closure.

The compiler's `xc/automatic_point.py` owns canonical CPU/CUDA point science and
work policy. `dft/xc_point_cuda.py` owns their common physical-spin traversal and
coefficient ABI. The static `CudaXcPointProgram` descriptor binds library-owned
callbacks, semantic/source identities, ingredient mask and feature width.
`CudaXcPlan::point_program()` exposes that immutable execution identity without
turning it into an installed artifact/cache key or a resource admission claim.
Automatic entries reuse the existing AO, density-feature and potential kernels
with strict FP64 and the bounded one-tile fallback. They do not inherit curated
mixed precision, response, local-AO selection, batching or graph qualification.

The default-on `GENERATIVEQC_CUDA_XC_EQUAL_SPIN` selector also applies to every
structurally registered automatic provider (`0` restores the old path).
Its RKS-only sibling shares scalar work after polarized differentiation,
leaving independent sigma-ab and all work-domain floors/clamps unchanged.
UKS and the explicitly disabled arm retain the original body; both arms keep the existing 128-thread automatic schedule and
bounded traversal. The descriptor's exact source identity covers both bodies
and the selector. This does not add response support, installed variants,
public backend admission, or resource/performance qualification. See
[equal-spin qualification](../maintainer/xc_equal_spin_qualification.md).

Build `generativeqc_dft_cuda_tests` with the option enabled. Inside a finite
Slurm GPU allocation, set `GENERATIVEQC_AUTOMATIC_CUDA_XC_EXECUTABLE` to that
binary and run `tests/python/test_libxc_automatic_cuda_xc.py`. The independent
PySCF 2.14.0 / Libxc 7.0.0 gate checks both physical spin layouts, partial tiles,
fixed-density XC energy/potential, replay, density directions and changed
geometry. This is not an SCF, analytic-force or installed-bundle qualification.

### Curated execution

The native generated grid translation unit reuses the existing through-f AO
kernel and generated D/C ingredient bilinears. LDA retains one AO/value feature
per spin; PBE retains value and three ordinary spatial derivatives. Neither
requests tau, D times derivative-AO panels, or higher AO jets.

The sequence is AO -> D times AO -> density/gradient -> shared point energy
and Cartesian potential coefficients -> weighted E/V reductions. The AO
traversal, dense D*AO/density-feature contractions, resident point-domain XC
coefficient algebra, symmetric-potential assembly and scalar-total reductions
are emitted by `generativeqc_compiler.dft.ao_cuda`; the resident header owns density
validation and launch/runtime scheduling only. The emitted point evaluator
differentiates the same stable energy used by the CPU
consumer; no separate singular sigma chain rule or CPU XC call is inserted.
Matrix assembly applies weights once, retains both differentiated AO legs, and
does not double the scalar term. All symmetric matrix cross terms are retained.

The compiler emits five bounded point consumers: physical LDA/PBE/r²SCAN and
signed LDA/PBE response. A plan resolves its immutable `(functional, response)`
key to an emitted launcher during preparation. Spin layout remains an argument;
CUDA KS intersects its shared `ExecutionPrecisionSchedule` with the selected
XC layout's capabilities before each iteration. A local AO layout keeps density
contraction strict FP64 while independently qualified Direct J may retain lower
precision. The filtered schedule preserves region names, surviving arithmetic
directives and qualification metadata. Strict refinement restores every region
to FP64; the requested schedule remains reusable for subsequent iterations.

### Occupied-frame density features

For physical four-jet density/gradient/tau evaluation, the native KS owner can
lend the AO coefficient frame that generated its accepted density. The lease
binds the density pointer, coefficient storage, AO/spin shape, occupations and
generation. It expires before residual/DIIS work reuses the coefficient scratch;
only another accepted structured proposal can publish a new lease. Imported,
held or mixed densities must not inherit an unrelated proposal's frame.
Response, nonlocal, chunk and lower-precision consumers retain their matrix
route. The [user selector](../user/ks_options.md#cuda-occupied-frame-xc-features)
requests reuse but cannot bypass these checks.

The compiler projects the original four AO jets through occupied coefficients,
then uses the existing feature bilinears. Reuse is admitted independently per
tile only when occupied rank is smaller than the active AO width. Projection
shares the bounded matrix-product panels; potential construction retains the
original AO jets and differentiated legs. No additional full-grid or quartic
storage is introduced. RKS applies its occupation factor exactly once; UKS
retains separate, possibly unequal or empty, occupied spin spans.

Floating-point storage is part of the contract. A nonzero coefficient outside
the conservative IEEE exponent interval `2^-256..2^256`, or a nonfinite projected
panel, declines occupied arithmetic. A warp vote in the existing feature kernel
recomputes the affected point from the still-live original stored density and
AO jets using the shared `symmetric_density_element` graph and feature
bilinears. This register-only fallback must not overwrite compact occupied
panels, whose strides differ from dense AO panels. It adds no allocation,
kernel launch, host synchronization or CTA barrier. A temporary refusal
sentinel is not a numerical error; invalid matrix results retain the original
finite-error behavior. Matrix fallback must not receive RKS occupation scaling
a second time. See the [representability rationale and independent edge cases](../../.agents/notes/implemented/numerics/2026-10-17-occupied-xc-representability.md).

### Matrix density providers

The density portfolio also contains a strict-FP64 materialized matrix-panel
candidate. It emits `0.5*D + 0.5*D.T` from the shared scalar graph once per
physical evaluation, then reuses that factor across all point/jet panels.
The shared tensor provider owns GEMM, finite publication and resource lifetime.
Qualification requires a separate 96 MiB provider allowance, exact spin-matrix
cache bytes, and 16 KiB host reservation. A budget alone does not select it;
generated execution remains the production incumbent. Mixed arithmetic and
signed response retain their existing bindings. The explicit native
`--density-provider` and `--density-provider-benchmark` commands exercise the
candidate and its complete fixed-density XC endpoints.

An independent indexed candidate gathers the same symmetric density graph over
each nonempty local AO map, then contracts compact `[spin, active, active]`
factors with the existing mapped point/jet panel. It preserves the local domain
and reuses one global-capacity cache and handle; it allocates and selects nothing
per tile. Empty maps retain the generated zero-domain operation. AO discovery
invalidates an earlier dense provider, so callers must prepare again after
discovery to admit the indexed recipe. Dense qualification cannot promote the
indexed candidate, whose gather costs need separate endpoint evidence. The
native `--indexed-density-provider` gate covers scalar oracles, changed captured
inputs, explicit and discovered maps, independent CPU E/V and resource fallback.
`--indexed-density-benchmark ORIGINAL_INPUT MOVED_INPUT` compares generated and
indexed execution on complete water-cluster grids. It reports map discovery,
provider setup, cold/warm XC endpoints and gather/work counts separately; its
positive diagnostic density and fixed-density scope do not qualify SCF/forces.

Native KS callers can supply `CudaXcPreparationBudget` from their enclosing
resource plan. Both host and device reservations must be admitted before
optional density preparation. KS reports the exact explicit cache under XC
device storage, the library allowance under provider storage, and binding
metadata under retained host storage. The matrix cache also participates in
the shared numeric allocation ledger, including budget failure and teardown.
Ordinary public callers retain zero optional reservations. Native
`generativeqc_ks_cuda_tests --density-provider` exercises complete PBE0 cold,
warm and changed-geometry solves with independent CPU energy/density gates.

AO precision does not change the FP64 point algebra. The selected consumer calls
the same canonical point implementation with constant functional/consumer facts,
so CUDA compilation can remove unrelated algebra before register allocation.
Physical PBE uses a compiler-selected 32-thread point block; the other consumers
retain 128 threads. The point tile and arena remain unchanged. Native code binds
its buffers to the retained launcher; graph replay retains that entry.
Unsupported functional/response pairs fail during preparation. See the
[consumer specialization decision](../../.agents/notes/implemented/performance/2026-09-23-xc-point-consumers.md).

### Bounded independent point submissions

Ordinary native KS defaults to bounded batching of independent XC point domains:
it requests up to 32 original tiles within an **additional** per-owner device
allowance of 32 MiB per density-product jet: 32 MiB for a one-jet contraction
and 128 MiB for a four-jet contraction. This is a resource-shape policy, not a
functional-name selector. The admitted count may be smaller. `GENERATIVEQC_CUDA_XC_BATCH_TILES=N`
overrides the request; zero or one explicitly retains the one-tile executor.
`GENERATIVEQC_CUDA_XC_BATCH_BYTES` overrides the additional allocation cap; zero
also retains the incumbent; an explicit byte cap is never scaled by the jet count.
If the automatic larger allowance cannot retain a batch, preparation retries the
original 32-MiB allowance before falling back to one tile. This scheduling policy does not prune or specialize
the canonical scientific point source.

Public `resource_budget` and `resource_plan` execution retains one-tile XC under
the active native device ledger. Its current inventory reserves incumbent storage,
later fleet owners and force workspace, with no separate optional-panel allowance.
Unused ledger capacity therefore cannot fund point batches. This also applies to
an explicitly supplied `ResourceBudget()` with unlimited user caps and to explicit
batch environment overrides. Ordinary calls without a public resource ledger
retain the default bounded batching above; budgets and their estimates are not
silently enlarged.

The compiler's `xc_point_batch_cuda.py` prepares a bounded residency plan from
the original tile domains and selected AO counts. Admission reduces the requested
batch size until complete retained panels fit; tiny domains, response, mixed
arithmetic, insufficient allowance, and allocation rejection retain a bounded
fallback, ultimately the one-tile executor. No device model or molecule-size whitelist is used.
Prepare maps and arithmetic bindings first. Preparation is forbidden during
capture or after evaluation; a retained batch also prevents later map discovery
or a mixed-density rebind. Changed geometry creates and qualifies a new owner.

Each group retains compact AO panels, but reuses the existing density-product
and potential scratch. Features, coefficients and three-channel point totals
have bounded tile-local channel-major slots, with a compact final partial tile.
A single point launch spans all group points using the canonical FP64 consumer
and the existing 32-thread physical-PBE or 128-thread other-consumer blocks.
Empty/noncontiguous/full AO maps remain independent; no union restores dense AO
work. No AO, jet, density or contraction arithmetic is repeated, and no point
gathers, descriptor transfers or extra stream fences are added to evaluation.
Vxc contractions/scatters and scalar reductions execute in the original tile
order on the original stream, rather than using concurrent matrix atomics.

For an admitted batch of `N` tiles, point submissions per Fock change from
`ceil(npoint/tile_points)` to `ceil(npoint/(N*tile_points))`; AO, density and Vxc
submission counts do not decrease. Record that distinction when interpreting
profiles. Retained/peak device storage includes the original arena **plus** the
optional AO and feature-slot allocation reported by `point_batch_plan()` and
charged to the numeric resource ledger/KS XC resource diagnostic. Preparation
includes planner traversal and allocation; there is no per-evaluation allocation
or retained host descriptor vector. Memory boundedness alone is not evidence
of endpoint profitability.

`generativeqc_dft_cuda_tests --point-batches` compares independent CPU E/V
references, both spins, ragged/empty/high-occupancy maps, tails, scaled PBE,
bitwise ordered accumulation, captured changed densities and bounded-memory
fallback. Run real-device gates only under a finite Slurm allocation. For
fixed-density XC work/memory census, `--point-batch-benchmark ORIGINAL MOVED`
on the same native executable reports preparation and six interleaved E/V
samples; those diagnostic densities do not qualify complete SCF/force timing.
For complete interleaved warm and moved-warm PBE0 E+F populations, use
`python -m benchmarks.pbe0_xc_tile_pairs --point-batch-tiles 32` with the usual
basis/reference/output arguments. This keeps both arms' SCF/force tiles at 256
and reapplies the arm's policy when geometry rebuilds the owner. Both comparison
arms explicitly set their policy, so a changed default cannot contaminate the
one-tile baseline. Compiled-resource evidence includes both the batched point
kernel and its retained fallback for multi-tile automatic execution. The
[default decision](../../.agents/notes/implemented/performance/2026-10-07-xc-point-batch-default.md)
records complete cold/warm/moved E+F evidence and remaining diagnostic limits;
the [capacity decision](../../.agents/notes/implemented/performance/2026-10-15-xc-point-capacity-by-work-shape.md)
qualifies the per-density-jet allowance and bounded retry. Source-specialization/composed
ablations belong to the independent #2072 arm.

**Resource-guarded compact contractions (default).**
Ordinary native KS additionally requests batched mapped
density products, density features, weighted AO panels and local Vxc
contractions. The compiler keeps each original active-AO map throughout; it
does not union maps or expand intermediate AO/density tensors to the full
basis. Only the final public potential is full-basis. A compact group needs at
least two tiles, with each nonempty tile having 32–1024 active AOs and at least
32 points. Empty tiles remain legal. Other original groups keep point-only
batching; one large or tiny AO domain does not disable independent small groups.
Response, mixed arithmetic and optional library-provider
execution retain their existing paths. `GENERATIVEQC_CUDA_XC_COMPACT_BATCH=0`
disables compact contractions while preserving point batching. The small-domain
bound is a resource/launch constraint, not a universal profitability
threshold. The automatic compiled-resource envelope includes every compact
stage and its retained fallback; explicit incumbent-only evidence cannot qualify
automatic execution.

The candidate reuses the incumbent FP64 bodies but masks density output-tail
lanes and non-authoritative/padded Vxc pairs while all lanes continue to reach
shared-memory barriers. Authoritative reduction order, coefficients, weights
and selected scientific summands are unchanged. Large groups use unique owners
of global AO row strips for the upper triangle and its symmetric mirrors;
each matrix element retains the original tile and addition order without
floating-point atomics. Small groups use one CTA per spin for
ordered scatter; consecutive equal AO maps keep the running potential in
registers and write once per map group, preserving every per-tile addition.
Different overlapping maps are separated by a block barrier, never atomics.
For `G` compact groups, density/features each submit `G` kernels and weighted
panels/local Vxc/scatter each submit `G`. The incumbent submits density and
Vxc once per nonempty tile, weighted panels once per tiled-contraction tile,
and features once per tile, including empty tiles. AO collocation still submits
once per nonempty original tile. Noncompact groups
retain the original density/feature/Vxc submission counts; the native work census
reports actual `compact_groups`, `compact_tiles` and `compact_nonempty_tiles`.

`GENERATIVEQC_CUDA_XC_RAGGED_BATCH=0` restores the original 32–128-AO admission.
Unset or a nonzero value permits larger groups. Density/panel/potential launches
use each group's actual maximum extent and include every cross-block term.
The existing additional byte cap includes retained panels, local potentials,
features and descriptors; shrinking groups and bounded fallbacks remain active.
Preparation trace `cuda_xc_batch_plan` reports `compact_large_tiles` separately
from execution counts. The [default decision](../../.agents/notes/implemented/performance/2026-10-12-ragged-xc-default.md)
retains complete-endpoint evidence and its cold/moved versus warm limitations.

Retained work panels, compact local matrices and immutable device descriptors
share the same **additional** XC byte cap as point batching. Descriptors are
uploaded once during preparation; transient host descriptor staging is bounded
by that cap and drained before destruction. Evaluation and capture allocate
nothing and upload no descriptors. Unsupported shapes or insufficient compact
workspace preserve point-only batching; an optional allocation failure retains
the one-tile executor. Public resource-ledger execution still admits no optional
batch allocation. Report actual `point_batch_plan().compact` and bytes, not just
the requested environment switch.

`--point-batches` also covers the compact candidate with independent CPU E/V,
empty/overlapping maps, partial tiles, both spin layouts, density replacement,
canaries and graph replay. `--compact-batch-benchmark ORIGINAL MOVED` reports
fixed-density preparation, actual contraction/scatter submissions and retained
storage. For complete E+F pairs, add `--compact-xc-batches` to the existing
`benchmarks.pbe0_xc_tile_pairs --point-batch-tiles 32` command. The benchmark
resets and restores this control separately for both arms, including moved-owner
rebuilds. Both compact benchmark arms use the same point-batch request so only
the compact contraction policy differs. Complete E+F qualification includes
12-atom small-AO and 48/96-atom larger domains. Numerical, sanitizer, resource
and complete-endpoint gates remain required when extending the admitted domain;
host simulation or reduced launch counts do not establish speedup. See the
[compact-batch decision](../../.agents/notes/implemented/performance/2026-10-08-xc-compact-contraction-batches.md).
The [default decision](../../.agents/notes/implemented/performance/2026-10-08-xc-compact-batch-default.md)
records complete cold/warm/moved E+F evidence and the larger-domain noise boundary.

The resident contraction block is currently compiler-emitted maintained CUDA
text, not a complete typed grid/XC IR lowering. The native header's runtime-only
ownership classification does not remove this remaining scientific-text owner.
See the [ownership decision](../../.agents/notes/implemented/architecture/2026-09-20-resident-xc-emitted-text.md)
for the preserved native/JIT boundary and the condition for replacing it.

RKS input is the total density. The point layer receives half in each spin and
the returned single potential uses the corresponding total-density chain rule.
UKS input is `[alpha,beta,AO,AO]` with independent spin densities and potentials.
E/V outputs remain in public normalized Cartesian or real spherical AO order.

The current dense reduction is a correctness baseline. It has no point-by-AO-
by-AO tensor, but it does not yet use the promoted local-dense or GEMM potential
contractions. The ownership ledger counts this numerical glue conservatively;
neither a scientific-code retirement nor a performance advantage is claimed.

## Direct J connection

`scf/cuda_direct_jk_device.hpp` exposes the #202 provider's ordinary stream and
an enqueue operation over caller-owned device density/J/K arrays. It reuses
the existing operator validation and contracted-ERI kernel, with device finite
checks and no success-path staging or synchronization. This lets a method use
one stream for matrix production, J and XC without inheriting the HF graph
layout. `CudaKsPlan` uses this seam directly, with an exact state arena charged
through the #203 device ledger. It reuses the existing matrix, DIIS, eigensolver
and density kernels, keeps current physical Fock and proposal state separate,
and reads one scalar diagnostic record per iteration. Convergence returns the
current evaluated density; only converged states replace the resident warm
cache. An energy-only public call omits final matrix export. Full #203 public
planning remains separate integration work.

Strict-FP64 restricted resident VV10 solves use bounded native initialization
before the ordinary physical target tail. The initializer first solves a
semilocal/range-exchange proposal problem, then refreshes complete VV10 anchors
between fixed-potential inner solves. Transported densities start with a target
anchor. Provisional and full-target iterations share the caller's main SCF
iteration budget; initialization gets no separate allowance. The existing ECP
physical-final-closure policy retains its separate bound of four correction
iterations. Only consecutive complete target evaluations passing the unchanged
energy, density and physical residual gates can publish a successful state or
forces. An already stationary target density is held for a further
physical energy confirmation, instead of applying another sub-tolerance update.

Resident warm replays and unsupported spin, precision, incremental-J/K or
budget cases retain their existing solve. A caller-selected device chunk also
keeps its existing solver. Set `GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED=0` to
select the ordinary full-target fallback explicitly. Provisional history scopes
and actual VV10 pair-work receipts distinguish initialization from accepted
physical work; initialization is never a cached convergence result.

Strict-FP64 direct restricted VV10 initialization automatically admits full
screened s/p/d range-ERI storage on grids of at least 65536 points, subject to
the existing reuse threshold, a 4 GiB value ceiling and complete host/device
admission. Cold and transported-density initialization can prepare a lease;
a retained warm call only reuses its audited immutable-geometry lease. It never
builds a new quartic source for one Fock. `GENERATIVEQC_KS_RANGE_RESIDENT_VALUES=0`
selects ordinary Direct evaluation; explicit `1` retains the separately selected
experimental capacity domain. The existing bucket mask and byte ceiling remain
explicit controls. Capacity refusal, unsupported angular classes, fitted,
lower-precision, incremental and externally budgeted device-ledger paths retain
their existing providers.

The lease binds geometry, radial operator and omega. It supplies only matching
raw range K; primary full J/K and derivative owners remain independent.
Successful solves rebase warm energy metadata with both incumbent exchange
terms before optional values are retired. This preserves the physical replay
gates and records the additional complete Fock and both exchange applications.

## Prepared ragged execution

`KsPreparedBatch` owns one compatible KS plan per input item. CUDA execution
submits every active item stream before reading any iteration's scalar record.
Converged and failed items leave the active schedule independently. CPU items
use the same scientific KS routines serially. Results and diagnostic queries
retain input order; this schedule does not inherit an HF graph layout.

An omitted coordinate entry selects the original prepared geometry. A changed
geometry reconstructs that item's basis/grid/J/KS owner and starts fresh DIIS.
Before releasing a CUDA owner, its last successful density is downloaded only
if a compatible seed has not already been exported. The old owner is released
before the new one is allocated, avoiding two complete live device plans for
one slot. The seed is normalized in the target overlap, separately for each
spin. A rejected or nonconverged warm solve gets one cold retry. Malformed
geometry and failed solves preserve the previous successful seed.

Normal energy-only replays leave warm densities resident. Explicit snapshot
export and geometry rebuilds are distinct transfer boundaries. The legacy
`*_hf_warm_state` buffer APIs also represent KS densities using the same total
RKS or alpha/beta UKS convention; they export scientific seeds, never cached
convergence. Imports validate the source metric and all supplied items before
changing any seed. Missing import entries preserve neighbors. Frozen warm
updates keep the same seed even across successful or changed-geometry solves;
clearing seeds while frozen prevents later solves from creating replacements.

The additive `generativeqc_batch_get_scf_diagnostic` query returns density-update and
physical-commutator RMS values without changing the legacy result array stride.
Unavailable and failed items report absence, including after a rejected replay.
Python batch calls default to the method's supported observables, so KS defaults
to energy and rejects forces. HF retains its energy-plus-force default.

## SCF local AO selection

Geometry-bound AO discovery is requested automatically for device-fused XC
layouts whose compiler-emitted point program and physical FP64 layout support
local AO selection. The XC owner determines legality; SCF does not add a
functional-name, spin, exact-exchange, range-separated, fitted-provider or
nonlocal-correlation whitelist. The enclosing KS composition must still be
supported by its own owners. Response, already-local and unsupported point
program layouts do not admit discovery.
`GENERATIVEQC_CUDA_KS_ACTIVE_AO=0` explicitly disables selection for debugging;
`=1` explicitly requests it and rejects layouts that cannot select local maps.
Invalid switch values are rejected. An automatic request keeps dense execution
when the layout or execution schedule does not support selection.

Public `resource_budget` and `resource_plan` execution retains dense AO work
under the active native device ledger: its inventory reserves dense XC, later
fleet owners and force workspace, with no separate optional-map allowance.
Unused ledger capacity cannot fund retained maps. This includes an explicitly
supplied unlimited `ResourceBudget()` and `GENERATIVEQC_CUDA_KS_ACTIVE_AO=1`;
the explicit request still rejects an incompatible layout. Unbudgeted selection
and `=0` are unchanged. Plan decisions report this budget fallback; the AO work
diagnostic preserves `requested` separately from `selected` and reports dense
work with no discovery when a requested map is not admitted.

Selected physical layouts propagate back from the XC owner. The shared
iteration precision schedule intersects that layout's arithmetic capabilities:
local density contraction remains FP64 while independently qualified Coulomb J
may retain its lower-precision directive. A whole-schedule FP64 veto would
incorrectly couple these independent operations.

Execution capability is not a profitability certificate or a numerical
qualification of the AO cutoff for every complete KS composition. Automatic
selection follows the shared capability policy; performance in the additional
domains is unmeasured. Use the complete-endpoint qualification described in
[performance engineering](../maintainer/performance_engineering.md) for claims
and further tuning. Frozen PBE0/WB97M-V receipts establish only their recorded
scientific, source, precision and device scope; they are not current-head or
generic-family performance evidence.

The prepared XC owner discovers AO value/first-derivative support at cutoff
1e-16. Existing compiler-generated contractions gather local density entries
and scatter the potential into the global matrix. Maps persist within that
geometry/grid owner; rebuilding coordinates or the grid requires new discovery.
Host/device resource admission can still retain dense execution, so read
`generativeqc_batch_get_ks_ao_selection_diagnostic_v1` to determine whether
selection actually occurred and inspect actual AO work and XC build counts.
This policy does not select local force AO maps or change Becke response.

`GENERATIVEQC_CUDA_KS_AO_DISCOVERY=host|device` selects the discovery schedule
at geometry preparation; `host` remains the default. The experimental `device`
schedule evaluates the same generated AO jets and `abs(jet)>1e-16` predicate,
then compacts sorted IDs on the device. It downloads CSR offsets and the error
status once per bounded chunk (at most 128 tiles). It reuses dead XC scratch
and the already admitted worst-case map storage; it adds no device allocation.
Both schedules publish only after validation and drain outstanding borrowers
on exceptions. Resource refusal retains the existing dense route.

The optional `GENERATIVEQC_DF_PROGRESS_TRACE` preparation trace reports
`cuda_xc_ao_discovery` with observed tiles, chunks, synchronization count,
transfer bytes, AO jet work and active counts. These are discovery-only counts;
frozen-geometry XC calls reuse the map. The public v1 AO-work ABI is unchanged.
Use separate clean complete endpoints for timing, including geometry preparation.

The targeted `generativeqc_dft_cuda_tests --pbe0-local-ao` gate compares
scaled-PBE E/V with independent CPU integration and checks empty maps and
bounded admission. It does not establish complete PBE0 SCF/force accuracy or
performance. Qualify cold, warm and changed-geometry energy/force endpoints
before promotion, counting discovery in setup rather than amortizing it away.

## Validation

`generativeqc_dft_cuda_tests` checks the actual device-buffer pipeline against CPU
full-matrix integration, the independent #214 H2 fixture, spin-resolved finite
differences, empty spin/vacuum tails, partial tiles and Cartesian/spherical f
shells. It checks a density changed by a device kernel without re-upload,
generation rejection, same-shape stale grid rejection, independent-stream
failure isolation/recovery and exact arena bounds.

All real-GPU invocations use finite Slurm allocations. Fixed-density results,
point-domain checks, full SCF, replay, changed geometry and batching must remain
distinct evidence until the corresponding native consumers are verified.

`generativeqc_ks_cuda_tests` independently rebuilds returned SCF densities with the
CPU providers, exercises resident replay and changed-geometry normalization,
and rejects stale grids and failed warm-state replacement. The registered
C API tests cover both spins/functionals on the actual CUDA backend and reject
forces. `tests/python/test_dft_scf.py` compares CPU/CUDA stable small endpoints
against two independently converged PySCF guesses on the identical grid.
`tests/python/test_dft_batch.py` separately exercises public ragged replay,
geometry rebuilds, per-item failures, frozen/cleared/imported seeds, ABI
diagnostics and method-specific derivative requirements. The combined batch
and independent SCF modules pass all 45 cases on the CPU/CUDA build with a
Slurm-allocated RTX 5090. Resource budget boundaries and ledger lifetime are
separately covered by `tests/python/test_ks_resources.py`; the full workload
evidence required by #162 remains a separate acceptance gate.
The five CUDA batch cases also pass Compute Sanitizer memcheck with full leak
checking: zero errors and zero bytes leaked.

An admitted radial source survives fixed-geometry warm replay by default.
`GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY=0` retires it before the next solve.
Changing range selection, bucket mask or byte ceiling also retires the lease.
A force-memory eviction restores the incumbent provider; separate energy
baselines preserve the original convergence checks in both cases. Every call
still evaluates the physical Fock, convergence gates, derivatives and force
assembly.

`GENERATIVEQC_KS_INTEGRAL_RESPONSE_REUSE=1` is a default-off Direct RSH force
experiment. A token-authenticated host snapshot may reuse only the two-electron
response when every density bit agrees within the same immutable prepared owner.
It adds no host density export. The one-electron and Pulay responses still run
with the current resident D/W. Optional key/result storage is capped at 64 MiB
and one quarter of the caller allowance; budget refusal, allocation failure and
explicit disable preserve the ordinary derivative path. This control is not
qualified for default selection.

`GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE=1` separately enables exact-density
retention of completed semilocal/nonlocal and nuclear geometry components in
the composite prepared owner. Current native integral and Pulay sources still
execute. Geometry/operator/grid changes, failures and explicit disable retire
the key; active-AO experiments and explicit stationary execution controls bypass
retention. FP32 semilocal geometry requests also bypass retention so Fast
precision receipts describe fresh work. Capability fallback and numerical retry
still compute strict geometry; later strict requests may reuse that result.
Its host allowance includes
comparison scratch, is capped at 64 MiB and one quarter of the host budget, and
uses only remaining admitted capacity. Hits report zero executed geometry and
pair work. This control is also default-off pending endpoint qualification.

The explicit `GENERATIVEQC_VV10_PAIR_SCHEDULE=parallel-wide` experiment uses
eight rows per lane and retains row inputs for admitted VV10 domains with at
least 65,536 active points. Smaller domains use the existing four-row schedule.
Both branches select using the device count without a host synchronization;
numerical admission, output ownership, ordered fallback and FP64 roots are
unchanged. This experiment requires separate complete-endpoint qualification.

The automatic VV10 pair schedule uses the rotating-partner packet kernel for
admitted molecular domains with at least 65,536 grid points on any CUDA target
supported by the build. There is no sm_120 whitelist: the generated schedule uses
FP64 atomics and warp shuffles, not architecture-specific instructions. Smaller
grids and refused numerical/output leases retain the ordered
fallback. `GENERATIVEQC_VV10_PAIR_SCHEDULE=parallel-packet` explicitly selects the
packet route, including small qualification fixtures; `parallel` restores the
previous four-row bilateral schedule and `ordered` forces the ordered fallback.
Unset and empty selectors both use automatic selection.
Each warp loads 32 adjacent compact partners and rotates their
inputs and bilateral FP64 partials through all lanes. After 32 steps each
partner returns to its original lane for publication. Four rows per lane keep
the 128-row tile size and scalar pair work unchanged; no extra workspace, host
count read, allocation or synchronization is introduced. Padded lanes still
participate in every shuffle. Numerical/output-lease admission, positive-zero
response, exact-component reuse and ordered fallback retain their contracts.
The changed reduction order retains independent numerical and complete-endpoint
qualification requirements. This is a scheduling change, not a scientific
approximation or a claim that all large molecular populations complete within
their benchmark limits. See
`.agents/notes/implemented/performance/2026-10-15-vv10-packet-default.md` for the
promotion decision, measured scope and unresolved 96-atom endpoint boundary, and
`.agents/notes/implemented/performance/2026-10-15-vv10-packet-portable-default.md`
for architecture-independent selection. Hardware admission is not a claim of
measured speedup on every supported architecture.
