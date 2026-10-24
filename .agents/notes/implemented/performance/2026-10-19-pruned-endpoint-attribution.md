# Investigation: water-96 pruned endpoint performance attribution

Status: implemented diagnostic; attribution completed, optimization qualification separate
Date: 2026-10-19

## Scope

The user asked why GPU4PySCF is faster than GenerativeQC in the newly qualified
LDA/PBE pruning measurements. Explaining that STO-3G/semilocal DFT is cheaper
than the historical README hybrid calculation does not answer this question:
both measured engines use the same water-96 geometry and 224-AO STO-3G basis.
This investigation changes no production numerical defaults or runtime paths.

## Confirmed endpoint facts

The retained jobs 10323/10324 contain six alternating complete-endpoint samples
after warmup. These are fresh owners and no converged-density reuse, but warm
process/artifact caches. Preparation, converged SCF, analytic forces, cleanup,
GC and device synchronization are all timed.

| Method | Native median | GPU4 median | Native/GPU4 time | Native points | GPU4 points |
| --- | ---: | ---: | ---: | ---: | ---: |
| LDA | 21.044082 s | 6.591725 s | 3.1925 | 2,467,200 | 1,078,528 |
| PBE | 14.995148 s | 7.813512 s | 1.9191 | 1,579,008 | 1,078,528 |

Native therefore uses 2.2876 times as many points for LDA and 1.4640 times for
PBE, despite both candidates being pruned. The independent production grids
pass common accuracy gates; they are not identical quadratures. This is a
confirmed work difference, not a measured allocation of the wall-time gap.
Do not divide total endpoint time by point count and call the result kernel
efficiency: preparation, forces, active AO counts, and SCF counts also matter.

Native performs 15 iterations / 15 reported Fock builds for both methods in
every sample. GPU4 performs 13 cycles / 14 get_veff calls for LDA and 15 cycles /
16 get_veff calls for PBE. Fewer iterations cannot explain the PBE advantage.
Stopping policies differ: native uses energy/density tolerances 1e-12/1e-10;
GPU4 uses conv_tol=1e-11 and conv_tol_grad=1e-8. Accuracy qualification, not an
assertion of equivalent stopping norms, is the shared acceptance boundary.

Both drivers are direct RKS, without density_fit(). The GPU4 driver explicitly
suppresses dm_last/vhf_last via configure_reference_full_fock, and requests
gradient.grid_response=True. It is not omitting forces or grid response, nor
using incremental Fock updates. Its serialized warm_start.used=true is a
reporting default of gpu_convergence_payload, not evidence of cross-sample
density reuse: reference_endpoint constructs a new RKS and calls kernel()
without a supplied density on every invocation.

## Code-confirmed performance lead, not yet a timed attribution

The measured source enables local AO selection for physical LDA/PBE and uses
host-tile discovery unless GENERATIVEQC_CUDA_KS_AO_DISCOVERY=device is selected.
The host route evaluates a dense AO tile, downloads selection flags, synchronizes
the stream, and uploads compact indices for each 256-point tile. It repeats
this setup for each fresh owner. The prescribed grids have 9,638 LDA tiles
and 6,168 PBE tiles; these are tile counts, not measured kernel-launch counts.

The existing device-chunk route applies the same sampled predicate and packs
up to 128 tiles before each host boundary: 76 or 49 chunks respectively.
This motivates a controlled scheduling ablation, not automatic promotion or
a claim that setup explains a particular fraction of the observed gap.

Do not incorrectly blame the public resource-ledger fallback for these
measurements. An active ledger closes optional AO maps/point batching, but
the measured Calculator supplies neither resource_budget nor resource_plan;
the ordinary endpoint is not executing inside such a ledger. Likewise, its
256-point logical tiles do not prove one launch per tile: optional point
batching exists and the actual prepared plan must be observed.

## Initial scheduling and provenance

Slurm job **10393** runs through srun on node1/main with one RTX 5090, one CPU,
8 GiB host memory, a finite 12-minute runtime and a 2026-10-19 20:00:00 local
deadline. At the last check it was pending, with a scheduler estimate of
17:22:00, not an execution result. All permitted GPUs on n1/n2/n4/n5 were busy;
n3 is under maintenance and is not used. A preceding immediate allocation
attempt, 10392, exited because GPUs were busy; it is not left pending.

The ignored diagnostic scripts are retained locally at
`.artifacts/pruning-gpu-20261019/attribute_pruning.py` and
`.artifacts/pruning-gpu-20261019/run-attribution.sh`. Copies and eventual raw
results reside under
`/home/jzzeng/generativeqc-qualification/pruning-20261019-codex/endpoint/`
on n1. Output names are `attribution-10393.json` and associated per-method
native CUDA/host JSONL traces, with scheduler logging in `attribution-slurm.log`.

The job deliberately reuses the original measured pre-merge source overlay
and library, not current HEAD. Its library SHA256 is
`92e4508516078043a72030fcd8dbf8ed6bbe556b8eece29a0e1b3511c4fac447`.
The diagnostic records script/library hashes, versions, device visibility and
execution overrides, and reuses the existing verified sccache/stationary cache.

The protocol first measures unchanged clean controls. Separate intrusive runs
record coarse preparation/execute/force/cleanup host scopes, native CUDA-event
SCF J/XC regions, force routing and AO-selection work. Explicit added fences
mean these totals cannot replace the clean production timing cohort. Finally,
untraced device-discovery runs test the scheduling lead without changing the
grid, sampled AO predicate, stopping controls or force request. Energy and
maximum-force differences from clean controls must each remain below 1e-9.

The pending state above records the initial scheduling observation, not the
final outcome. The job subsequently completed; its results follow.

## Completed measurements

Job 10393 completed on the retained pre-merge capsule. Untraced fresh-owner
control samples are 21.066460 / 21.061583 seconds for native LDA and
15.208113 / 15.227489 seconds for native PBE. The corresponding GPU4 controls
are 6.570771 / 6.544735 and 7.901843 / 7.863635 seconds.

Intrusive host scopes put native preparation at 2.33134 seconds for LDA and
1.69320 seconds for PBE; SCF plus analytic forces take 18.65489 and 13.45117
seconds. AO discovery itself takes only 0.40747 / 0.35417 seconds. Device
discovery clean samples are 21.007415 / 20.863308 and 15.121403 / 15.102502
seconds: approximately 1.0061x / 1.0070x endpoint ratios. Energies agree and
maximum force differences are about 1e-13. This rejects AO discovery as the
main optimization opportunity; the default is unchanged.

The native compiled force controller is eligible and actually used. LDA's
separately traced 15 XC applications total 5.28948 seconds, whereas its 15
exact-J applications total only 0.09261 seconds. One-electron and direct
force intervals are 0.29650 and 0.34348 seconds. These intrusive scopes are
not additive fractions of the untraced wall population.

Nsight Systems job 10395 captures a fresh PBE endpoint after a complete
warmup. `geometry_cooperative_kernel<false,double>` accounts for 14.8239
seconds across 6,168 calls, 49.7% of the intrusive kernel-time total;
partition kernels account for about 1.8406 seconds (6.2%). Profiler overhead
is substantial: 49.7% is not a clean endpoint wall-time fraction and the
14.8239 seconds cannot be subtracted from the clean endpoint timing.

The measured native compiled controller configures cooperative geometry but omits
the existing optional phased Becke schedule used by the Python/composite
owners. Reusing that algorithm, with explicit concurrent cached-tier byte
admission and the existing bounded fallback, is the next controlled
optimization. Qualification is recorded separately rather than inferred
from this profiler observation.

That optimization is now implemented and qualified in
[native phased Becke admission](2026-10-19-native-pruned-phased-becke.md).
Its subsequent functional-independent scope and source-matched requalification
are recorded in [generic phased admission](2026-10-19-functional-independent-phased-becke.md).

Raw outputs remain ignored under
`.artifacts/pruning-gpu-20261019/attribution-10393.json` and the original n1
capsule's `endpoint/nsys-pruned-pbe-10395.{nsys-rep,sqlite,json}` and
`endpoint/nsys-pruned-pbe-10395-stats.csv`.

## Revisit when

Read the completed reports before repeating profiling. Attribute preparation,
SCF J/XC and force costs separately; quantify instrumentation overhead and
check work/results before drawing conclusions. A same-grid fixed-density XC
counterfactual may be useful if grid size and implementation costs remain
confounded. Any production scheduling change still needs independent numerical,
resource/fallback and complete-endpoint qualification.
