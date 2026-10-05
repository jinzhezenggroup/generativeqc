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
generated execution remains the production incumbent. Mixed arithmetic, local
maps and signed response retain their existing bindings. The explicit native
`--density-provider` and `--density-provider-benchmark` commands exercise the
candidate and its complete fixed-density XC endpoints.

AO precision does not change the FP64 point algebra. The selected consumer calls
the same canonical point implementation with constant functional/consumer facts,
so CUDA compilation can remove unrelated algebra before register allocation.
Physical PBE uses a compiler-selected 32-thread point block; the other consumers
retain 128 threads. The point tile and arena remain unchanged. Native code binds
its buffers to the retained launcher; graph replay retains that entry.
Unsupported functional/response pairs fail during preparation. See the
[consumer specialization decision](../../.agents/notes/implemented/performance/2026-09-23-xc-point-consumers.md).

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
