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
it requests up to 32 original tiles within a 32-MiB **additional** per-owner device
allowance. The admitted count may be smaller. `GENERATIVEQC_CUDA_XC_BATCH_TILES=N`
overrides the request; zero or one explicitly retains the one-tile executor.
`GENERATIVEQC_CUDA_XC_BATCH_BYTES` overrides the additional allocation cap; zero
also retains the incumbent. This scheduling policy does not prune or specialize
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
arithmetic, insufficient allowance, and device/global-ledger allocation rejection
retain the one-tile executor. No device model or molecule-size whitelist is used.
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
source-specialization/composed ablations belong to the independent #2072 arm.

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
