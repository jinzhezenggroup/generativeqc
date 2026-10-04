# Decision: use the idle CUDA KS eigensolver for checkpoint admission

Status: implementation candidate; device qualification and endpoint comparison pending
Date: 2026-10-04

## Problem and measured scope

The private GPU LDA preliminary-density experiment charges source export and
target import to complete WB97M-V cold startup. At 48 atoms, its original
source export/import costs about 12.7 seconds. A separate finite n5 Slurm 1420
component audit distinguishes the ABI calls: 24-atom export query/copy take
0.000138/0.000103 seconds, while three imports take 1.231363/1.225954/1.225731
seconds. At 48 atoms, query/copy take 0.000495/0.000400 seconds and imports take
15.684234/15.631609/15.632248 seconds. Imported densities remain bitwise equal
to the converged source. The audit does not execute a WB97M-V target endpoint.

Slurm 1421 wraps the unchanged restore ABI in the existing host observer. Each
48-atom import makes two actual `reference_eigensolve` calls with reason
`seed_validation`. Their combined time is 15.346126–15.363690 seconds out of
15.625583–15.642699 seconds of native restore. This identifies scalar reference
diagonalization as the dominant import cost, rather than density transfer.
It does not measure an optimized endpoint or a general checkpoint workload.

The audit uses qualified integration source 678f7eb88, native identity
`d198acd713d7c4218bb4cacd9acbd7eb646d5fc2b89c77d04bdb47901c74ad36` and library
`fe826ad7eb9293686d5bb3b5e97456f3a9c884a13256603e7573ea3a1703f25b`.
The implementation is separately based on master d9431c913. Evidence, scripts,
ccache receipts and the diagnostic wrapper remain in the integration checkout's
ignored `.artifacts/seed-transfer-audit-20261004/`. The first audit attempt,
Slurm 1419, fails during library loading because importing CuPy first binds an
older private cuSOLVER. The corrected driver matches the comparator's
native-first load order; that failure is retained, not treated as qualification.

## Decision and invariants

Allow the existing shared warm-density guard to receive a borrowed eigen
operation. An idle prepared CUDA KS owner supplies its ordinary eigensolver;
CPU callers and owners without a prepared CUDA plan keep the existing route.
Source overlap is still rebuilt from the checkpoint's coordinates, not silently
substituted with the target metric. Electron/spin trace, shape, finiteness,
Hermiticity, singular-overlap and ensemble-occupation gates remain unchanged.
The common guard retains its existing reference fallback for legacy inputs
which satisfy its 1e-7 asymmetry contract but not the GPU frame's 1e-12 rule.
Runtime/provider failures propagate instead of selecting a numerical retry.

The callback borrows the already charged Fock/residual DIIS-history matrices
and the iteration solver's temporary eigenvalues/status while no iteration is
active or pending. Every future `begin()` discards that history. It does not
borrow `tmp1/tmp2`, which may hold a published stationary D/W lease. Final
coefficients/eigenvalues/status, density, warm state and generation tokens remain
untouched, including on rejection. Graph capture is explicitly rejected.
No device allocation or extra solver owner is introduced. Synchronous downloads
drain before host input/frame/status storage can expire.

Column-major device eigenvectors are converted to the common row-major frame.
The shared independent eigenframe residual/metric validator checks every frame
before the existing ensemble guard consumes it. Native code binds the provider
and storage; it does not reimplement an occupation formula or repair a density.
The optional host trace reports actual `cuda_ks_seed_eigen` executions.

## Required qualification

The new independent-metric tests cover native-small/provider solver sizes,
restricted/unrestricted analytic ensemble densities, charge-preserving invalid
occupations, atomic multi-item rejection, continued last-good warm execution,
exact density preservation and legacy near-symmetric admission. Device runs,
live stationary-weight lifetime checks, existing checkpoint/resource regressions
and complete source-plus-WB97M-V E/F comparisons are still required. This slice
does not expose a public CUDA preliminary-SCF API, lower resource bounds or
claim that the remaining source/target capacity contract is solved.

## Initial device qualification and retained baseline failure

Release/sm_120 build source 8829743e1 has 444 verified ccache compiler
commands, native identity
`bf918c0f028de30a00ffec7223f39b91f688ec476027e73c8797a862423bd168`
and library
`5b216500f300c5251bebb006bccb3bd05323c4365d2cf46054c2a6b2cad3d907`.
Finite n5 Slurm 1422 passes both standalone ordinary-eigen and shared seed-gate
executables. The initial Python run exposes fixture errors: the named PySCF
STO-3G table differs from the pinned BSE data at the unchanged electron-count
gate; a neighbor comparison incorrectly crosses a warm SCF step; and older
SCF/snapshot tests rely on a now-obsolete energy-only CUDA default. The tests
now use exact bundled coefficients with independent PySCF integrals, compare
preservation to the immediate pre-import neighbor, and explicitly request the
energy-only behavior they test. No scientific gate is relaxed.

Slurm 1425 reruns both native executables and reports 29 Python passes,
including all four new independent ensemble/live-stationary-lease cases, with
one existing PBE-UKS OH changed-geometry discrepancy. A separate finite n5
baseline run reproduces the same discrepancy with reviewed #1801 library
`dc3fca86b9c44013268e6883f521b515a4a2afcf53ec8ea675d7825a0d2a62c1`
(native identity `4eab0c4fa2bd533a216349604959dcbb334cfa57de014e2b914369c93b7605b3`):
changed/fresh energies are -75.58331065095038 / -75.58331066533137 Eh,
versus candidate -75.58331065095040 / -75.58331066533140 Eh. The 1e-9 gate
remains failing; this is not reported as a full passing regression. That case
does not import a checkpoint and never invokes the new admission provider.
The failure and initial attempts remain under ignored
`.artifacts/qualification-20261004/`, including the test patch and hashes.

The four new tests are scientific admission/lifetime qualification, not a
complete WB97M-V cold speed measurement. Latest-master composition and complete
preliminary-source plus target E/F controls remain separate required evidence.

## Latest-master composition

Merge 622af6871 includes master b909e14c1. Its frozen Release/sm_120 build
verifies 453 ccache compiler commands, native identity
`20dac455927d54d4361dd08c163a3ca3c92641ef08f69e5a48fa0a0a068841f1` and library
`06c296721006fbdd79378e96dcca5f8880b119ded66e94290b11187b01e0e3f7`.
Finite n5 Slurm 1427 passes all three native executables: ordinary stream eigen,
shared proposal/seed guards and the now-linkable full CUDA DFT regression.
Python again reports 29 passes, including all four new admission/live-lease
cases, and the one unchanged baseline PBE-UKS OH failure above (no skips).
The raw allocation exits 1. The scoped receipt preserves that outcome and
explicitly says `whole_suite_passed: false`; it admits only the new checkpoint
path, not a passing full Python regression.

The complete private LDA-source plus WB97M-V E/F control compares this candidate
with actual master b909e14c1 on one allocated RTX 5090. The baseline reuses the
qualified #1801 binary only after recomputing and matching every native input
of the actual master snapshot. Both sides use identical source/target settings,
charge source creation/solve/export/import/destruction to complete cold, retain
three warm samples and require every E/F pair against GPU4PySCF. Slurm 1430 is
queued with a finite two-hour limit for 3- and 48-atom controls. No result or
endpoint speed claim is inferred from submission. Source archives, binary and
driver hashes, the exact retained baseline failure and all raw outputs remain
in ignored `.artifacts/masterb909-20261004/` and
`.artifacts/cold-admission-20261004/`.

## Master 73755342d integration

The next union takes #1800's CPU KS final-maximum closure from master
73755342d. It adds a shared optional convergence-eligibility predicate and
changes CPU KS closure; it does not change the CUDA checkpoint callback or its
source-metric/occupation contracts. This union receives a new source-bound build
and qualification. The b909 complete cold controls keep their original archives
and binaries; source convergence changes are never retroactively substituted
into their timings. The inherited GPU PBE-UKS changed-geometry discrepancy is
not assumed fixed by a CPU-only change.

## Complete candidate endpoint checkpoint

Slurm 1430 subsequently completes both 3-atom variants and the 48-atom
candidate, each passing all five independent cold/priming/warm energy/force
pairs. The 48-atom baseline is still running at this checkpoint. These records
retain the b909/622af6871 source and library identities above. The allocation
actually provides one CPU and 5 GB host memory, with eight configured BLAS/OMP
threads for both variants; keep these results separate from earlier eight-CPU
campaigns and do not infer a differential speedup before the baseline completes.

The candidate's 48-atom complete cold is 897.029193 s versus its paired
GPU4PySCF 494.848258 s, with 15/16 SCF iterations. Warm medians are
151.512390 / 103.706417 s, each using one iteration. This ordinary-master
checkpoint branch does not contain the composed WB97M-V integral/AO/index
optimization; these timings are not points on the published integration curve.
Maximum errors are 1.1824e-11 Eh / 6.211e-10 Eh/Bohr, within the unchanged
1e-8 / 1e-7 gates, and every reference XC component reports on-GPU execution.

The LDA source converges in 25 iterations. Its entire lifecycle costs
36.728837 s, including 36.362070 s preparation/solve, 0.001107 s density export
and 0.362802 s checkpoint import; all source costs are charged to complete cold.
The target uses the imported density without fallback. These completed candidate
measurements establish scientific endpoint behavior and the actual import cost;
they do not establish a controlled baseline-versus-candidate endpoint benefit.
The locally rerun independent verifier, raw samples and hashes remain in
`.artifacts/cold-admission-20261004/results/matched48-candidate-verified.json`.

## Composite-force caller master update

The branch also takes master f88f42b4f, which adds #1778's opt-in local-AO
consumer to composite forces. The checkpoint provider and its admission gates
remain unchanged, but the runtime Python inputs change, so the completed
master-737 library is retained as an intermediate build and is not relabeled.
A new source-bound ccache build and native/admission qualification cover this
union. The running b909 baseline/candidate endpoint comparison remains frozen.

Frozen 11d793fac completes the new build with 453 verified ccache compiler
commands, source identity
`dcfdc7f0687b950381ea5dca9c3f47601a521cfdb43fca3756cef7cad93f94eb`
and library SHA-256
`1d5d7ac28cd214bc6ae4e86fb9682657dca793b5b398f6b8d7b6af0312bdd955`.
All 37 device-free resident-caller tests and applicable hooks pass. Finite n2
RTX PRO 6000 Slurm 2194 passes all three native executables and all four new
admission/live-lease cases. The Python selection again has 29 passes, no skips,
and the same single PBE-UKS changed/fresh failure: -75.58331065095037 versus
-75.58331066533135 Eh at the unchanged 1e-9 gate. The prior n5 baseline control
remains separately identified; no new PRO 6000 baseline rerun is implied.

The raw job exits 1 and the source-bound scoped receipt explicitly retains
`whole_suite_passed: false`. Before execution, the runner verifies every deployed
source/test file against the archived commit, the native manifest, library,
native executables and qualification-script hashes. The completed evidence is
under ignored `.artifacts/masterf88-20261004/`; it establishes current-composition
admission qualification, not a passing whole Python suite or an endpoint speedup.

## Completed frozen baseline comparison

Slurm 1430 has now completed both 48-atom variants successfully. The original
independent verifier accepts all five cold/priming/warm E/F pairs for each
variant, with every reference XC component on GPU. The actual master-b909
baseline has complete cold 911.838916 s (52.944932 s preparation and
858.893984 s first E/F execution), versus 897.029193 s for candidate
622af6871 (37.789973 s preparation and 859.239220 s first E/F execution).
Both target solves take 15 iterations. Baseline/candidate warm medians are
151.485277 / 151.512390 s; no warm improvement is claimed.

The source lifecycle is 51.875090 / 36.728837 s. Construction, preparation and
solve remain one timer, 36.301326 / 36.362070 s, with 25 source iterations.
Source Fock-build counts remain unavailable. Export takes 0.001129 / 0.001107 s;
import takes 15.569886 / 0.362802 s. Destruction and other source bookkeeping
are included in the lifecycle total but not separately timed. These values
support a measured import reduction and an observed 1.624% complete-cold
reduction, not a 43-fold endpoint improvement. The ordered single-cold controls
still need reverse-order repetition before claiming a stable general gain.
The original one-CPU/5-GB allocation and eight configured BLAS/OMP threads remain
part of this evidence; do not combine it with the earlier eight-CPU controls.

The 3-atom import does not improve (0.001499 / 0.001984 s), while complete cold
is 10.791685 / 10.755986 s. Keep the small negative component result. All four
3/48-atom reports, all-repeat errors, original verifier receipts and Slurm exit
status are retained under `.artifacts/cold-admission-20261004/`. This completed
checkpoint supersedes the pending baseline statements above without changing
their source/library identities or substituting them into the README curve.

## Latest-master qualification boundary

The resumed work integrates master fc7d5e2d8, including the private VV10 phase
storage change and the offline CUDA timing model. A new ccache build and
source-bound device qualification will cover this composition. The completed
b909 timings remain historical controls and are not relabeled as latest-master
performance. The public CUDA preliminary-source capacity/lifecycle API remains
a separate task; neither this merge nor GPU admission removes its requirements.
