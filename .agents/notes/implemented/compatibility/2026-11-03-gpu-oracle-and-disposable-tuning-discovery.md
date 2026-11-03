# Decision: independent GPU DFT oracle and disposable tuning discovery

Status: implemented; device-specific performance promotion remains separately gated
Date: 2026-11-03

## Problem

The first bounded master-based A100 J/K search exhausted its entire 1,800-second
budget on a single-threaded CPU PySCF reference before measuring any candidate.
The oracle is necessary for scientific acceptance, but CPU SCF/gradient execution
is not a production Direct requirement or the only independent implementation.
See the [original proposal](../../proposed/2026-11-03-a100-native-jk-autotune-fp64-baseline.md).

An initial explicit GPU-oracle attempt exposed a second, independent issue:
native device discovery calls `cudaSetDevice`, which can retain a CUDA context
in the tuning coordinator. On an exclusive-process GPU, the subsequent oracle
or native worker cannot establish its own context. Restoring an ordinal is not
the same as releasing the context. The initial failure is not a candidate
performance measurement.

## Decision

Offer `--reference-engine gpu4pyscf` for explicit DFT tuning. Both independent
SCF and analytic gradient use GPU4PySCF on the requested CUDA-visible ordinal;
GPU package import also occurs within that device scope. Host PySCF molecule and
basis construction remains setup. The original CPU oracle stays the default,
and missing GPU dependencies, nonconvergence or CPU SCF/gradient owners fail
closed without a CPU-solver fallback. Record producer engine and versions.

Run tuning discovery, incumbent library selection and recipe replay identity
verification in disposable subprocesses. Only those children call the native
GPU probe, and their exit releases their contexts before the next GPU worker.
The parent can load the context-free native ABI but does not perform discovery
on it. Production Calculator profile selection is unchanged.

Explicit SHA256-pinned independent-reference reuse is also available through
`--reference-file` and `--reference-sha256`. The reference document binds exact
Bohr coordinates, named and resolved reference Gaussian contents, the immutable
native basis catalog, functional components, representation, charge/spin,
batch, quadrature family/level/pruning and analytic grid response. No device or
native schedule enters this scientific key. The caller pins the original
producer provenance; this is not permission to manufacture a reference document
or to import historical timings as fresh performance evidence. Every retained
observation is checked independently at the unchanged numerical floors.

## Rejected alternatives and invariants

- Do not drop the independent oracle or relax numerical gates to obtain timings.
- Do not patch the oracle at runtime or relabel old observations as a fresh run.
- Do not use a process-global `cudaDeviceReset`: another caller might own a
  live context. Disposable children release only their own resources.
- Do not change GPU compute mode, hide scheduler visibility, or bypass LSF/Slurm.
- Keep scientific inputs, strict SCF controls, analytic grid response, complete
  native E+F timing, observed work, allocation gates and fresh holdout intact.
- A GPU oracle and coordinator fix are not a faster Direct algorithm and do not
  qualify an A100 default or transfer a recipe to RTX 5090.

## Evidence

Host tests execute synthetic engine dispatch, RKS/UKS and functional selection,
device scoping, batch/force layout, convergence/dependency/CPU-owner refusal,
subprocess failures and visibility preservation. Search/replay tests retain all
raw numerical, performance, work, resource and corruption gates. These tests
are executable interface contracts, not GPU performance evidence.

The frozen master-based experiment and source-matched scheduler receipts are
retained under `.artifacts/direct-jk-gpu-reference-20261103/`. Terminal outcomes
and source identities are appended after inspection; no default is promoted
while the search is incomplete.

### Reference preparation and initial outcomes

CPU build 225210 succeeds. GPU bootstrap 225211 fails in GPU4PySCF import because
the coordinator still owns the exclusive-process GPU. No candidate is measured.
After disposable discovery is implemented, CPU build 225212 succeeds and GPU
job 225213 reaches independent GPU SCF, proving the context conflict is resolved.
The strict GPU oracle does not converge in its 150-cycle limit and is refused
before gradients/candidates. Do not misreport that as an accurate reference,
a native performance loss, or justification for relaxing convergence criteria.
Together these GPU preflights take 100 seconds (5 + 95).

Instead, the next frozen attempt reuses all four original converged independent
GPU4PySCF observations from LSF 224875, not a new oracle solve. Original JSON
SHA256 is `499e48cfa604222c53eb37b83e03408fcec0b07d9bbd6554387f84362fb3b80d`;
producer revision is `cd2b4514b`, PySCF 2.14.0 and GPU4PySCF 1.9.0. Its original
runner hash (`8231b80c16936d25e6598ebb1b409258ee45d155321708f3c92bad7fc91e0d2d`)
is independently matched to Git. The original and current native basis catalogs
are byte-identical (`1bbb5611e5580d033e56eea7e24cf17ef85e58d0d6d67aa25e3732690822d6fe`).
All 96 Bohr coordinates, charge/spin, spherical def2-SVP, standard full-range
PBE0 components and the complete Treutler/NWChem level-3 grid/response settings
are checked. Resolved PySCF Gaussian contents enter the reuse identity. PBE0
alias versus explicit component semantics are checked using hybrid coefficients
and libxc values/first derivatives; their parsed functional IDs need not be
identical because libxc also exposes a combined hybrid functional.

The original producer uses its documented defaults (energy tolerance 1e-9 and
resolved orbital-gradient tolerance sqrt(1e-9)), **not** the new strict fresh
oracle settings. Preserve these original settings and convergence observations;
do not relabel them as a newly strict solve. The unchanged independent numerical
gates still compare every fresh native sample with every one of the four
original observations. Native schedule parity remains the separate, tighter
gate, and no old oracle timing is used in speedup calculations.

The pinned reuse document SHA256 is
`65e52997c8ea468d8721dc8159c6ad37a469fec9e21ee2860e8288d982cc3600`.
The final frozen master-plus-tuner-corrections source identity is
`bf3d17bf82c8069d3c4e13449862bdb4e3495a1f49a424257b6654f1282c061d`.
CPU build 225229 and GPU search 225230 retain the original six-candidate/five-pair
scope but reduce the remaining search budget to 1,700 seconds: no silent budget
increase after the 100 seconds of reference preflight. This is the first attempt
that can proceed to native candidate timing. It does not qualify the integrated
wider-force-queue source merely because that branch carries these tuner changes.

### Host validation

The focused tuning/profile/CLI/reference suite passes **156 tests**, with seven
explicit environment/GPU skips and three native J/K GPU cases deselected. Ruff
checks and format checks pass for all twelve changed Python files. The existing
local SM120 build is rebuilt with verified sccache 0.16.0, including the native
source-identity header, so the host ABI/source check uses a current image rather
than an old binary. Its integrated source identity is
`3091ad10150dc1b45ab50e23b6e7d243570ad058842e7adf99eb62e1234d481f`.
No physical RTX 5090 execution is claimed by this host build. All six tuner
runtime modules are byte-identical to the frozen A100 experiment modules;
the integrated wider-force-queue CUDA source is still a different native build.

Parallel PR changes through `57d8af1c4` are subsequently preserved with a
fast-forward, without overwriting their evidence cleanup or upstream DF/CI work.
The post-integration host suite passes **155 tests**, with seven environment/GPU
skips and four deselections. The additional deselection is the source-bound
native-image check: the earlier 156-pass image is now explicitly pre-integration.
A full SM120 rebuild expands to 293 tasks after those unrelated native changes
and is stopped during close-out; no stale image is relabeled as the new source,
and the identity check itself is not weakened. These are host contracts, not a
new GPU qualification of the combined branch.

### Completed bounded result

Search 225230 runs from 23:02:12 to 23:30:38 +08:00 and exits normally. The
1,700-second policy-search budget expires while attempting the second policy;
the remaining few seconds are startup/report/receipt overhead. There are 14
completed native workers, each retaining its observed cold/warm diagnostics.
No CPU reference SCF or gradient is solved in this search.

One policy completes the required five interleaved pairs:
`GENERATIVEQC_DIRECT_K_TASK_SCHEDULE=fill`. Complete frozen-density warm E+F,
including host return, has paired medians **9.503142 s incumbent / 9.494493 s
candidate**. Speedup is **1.000911x**, about **0.09%**; the bootstrap lower 95%
bound is **0.996797x**. This fails the unchanged 1.02x/statistical performance
gate and is refused. It does not meaningfully close the A100 Direct gap.

Observed native work per warm batch item remains 1,078,336 grid points, one Fock
build, one Coulomb application and one exchange application, with complete
census and the same resolved XC schedule. Scientific scalar results need not
be bitwise equal for the work census to match. All samples pass independent
numerical gates against **all four** original observations: maximum energy
error per atom 3.28745e-12 Hartree, maximum force error 4.09572e-7 Hartree/Bohr,
force RMS 1.10020e-7 Hartree/Bohr. Measured resource gates also pass; each
candidate device peak is 623,529,284 bytes. No kernel-resource estimate
substitutes for the native allocation ledger.

The `primitive` K schedule has only three completed workers before a timeout,
not five paired repeats. The other four policies never receive a completed
comparison. **Do not describe all six policies as measured losses.** There is
no qualified winner, no holdout promotion, `qualified_recipe=null` and
`installed=null`. A100 is not installed as a high-FP64 family default, and RTX
5090 defaults remain unchanged. No additional GPU measurements are submitted.

The actual A100 main-library SHA256 is
`03ea2e15595cf2c6fa6c01c785b8ee51f2c855eae8a5738f87b32cbb0732ab34`.
Terminal receipt digests are:

- Build 225229: `5e8e163e1f86292ff1fb15c25ac4b94b5c14bc8bbf1edd0255eabfb4c72447bf`.
- Search 225230: `26b3aec67b0163a419f6d25f52f3492205d0b2630ebded24d42f9e44a6d25636`.
- Read-only CPU inspection 225232:
  `f5da779e8bb2deee3aea35d0dc11609d95a9c65c15930b4c89e0503d2a71c939`.

The retained `v3/audit.py` recomputes every completed raw acceptance gate,
checks build/report/library identities, and verifies that each retained
independent energy/force/settings record is unchanged from the original JSON.
Owned transfer/cache processes are stopped; shared VPN/SSH infrastructure and
compiler caches are retained.

Reference reuse fixes the original CPU-oracle blockade, but the next practical
budget bottleneck is repeated native cold convergence/preparation/warmup in
every isolated replay (17 cold SCF iterations in the observed workers). A
future large-workload tuner should explicitly budget these costs, reserve a
fresh holdout, and consider exact validated frozen-density handoff. Do not
repeat the same six-policy/30-minute setup assuming warm endpoint time alone
predicts search cost, or silently expand the user's budget.

## Revisit when

Independent-reference reuse is still useful for repeated workloads, but must
validate exact physical-model identity and retain original producer provenance.
For high scalar-FP64 defaults, first obtain a passing local policy recipe and
then qualify the intended family/workload coverage. FP64 support alone is not
a performance class, and A100 branding is not a dispatch predicate.
