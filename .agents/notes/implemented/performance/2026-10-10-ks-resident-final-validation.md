# Decision: reuse resident KS matrices for strict final-state products

Status: implemented opt-in; default reference route unchanged, no broad performance promotion
Date: 2026-10-10

The original default-off decision is superseded by the
[bounded PBE0 automatic-selection decision](2026-10-11-default-pbe0-resident-final-validation.md).
The source-matched observations and publication scope below remain historical.

## Problem and evidence

The source-level GPU4PySCF comparison inspects upstream commit
`c1a6e371d1a46afc9932c618108a4bc70e895edf`, including `grad/rks.py::energy_ee`.
That owner consumes density/orbital data without executing our live native
snapshot proof. The comparison motivates reducing redundant proof work, not
removing our provenance or final-state gates. The tracked evidence summary
below preserves the algorithm comparison without depending on an unpublished
proposed note. A retained host observation attributes
approximately 0.914 s to native final validation in an earlier 96-atom endpoint,
versus approximately 22 ms for its NumPy SPD check in CPU replay. Native
validation is a substantial complete-owner consumer. This decision does not
assert the cause of the earlier transient
96-atom regression, remove its failed samples, or promote larger XC admission.

## Implemented opt-in dataflow

Under explicit `GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION=1`, CUDA KS can
compute final-state product evidence from the same D/F/C/epsilon it just
exported. The private callback authenticates the live owner/generation token,
the provider's exact S/H inputs and the exact exported arrays. It uses the
existing matrix-library handle and shared final-validation reduction kernels.
Only the existing shared validator accepts or rejects the evidence.

Four phase-local iteration matrices supply FC, SC, Gram/canonical, reconstructed
D, DS/DSD and both FDS/SDF products in a serialized schedule. A bounded partial
packet array is charged by the shared KS arena layout used for allocation and
resource planning. Its maximum is 385 packets (less than 50 KiB); no separate
four-matrix or nine-matrix allocation, second cuBLAS owner, dense host upload,
floating atomic or response-cache lease is introduced.

For a canonical nonempty spin channel, the schedule has nine full-AO products
and one occupied-rank density reconstruction. The reference route has eleven
full-AO products and the occupied reconstruction: it recomputes FC for
canonicality and computes `(FD)S` separately from its already-needed DS.
Reusing FC and evaluating `F(DS)` removes those two full-AO products, without
pretending that both approximately symmetric commutator branches are identical.
The algebra remains cubic in AO size; this is reduced redundant work and
resident library execution, not a changed asymptotic integral algorithm.
The legacy exported D/F/C/epsilon bytes and host orbital-energy W remain in the
complete endpoint contract. The new proof adds one bounded result packet per
spin and two drains; it does not eliminate the existing dense host export.

The adapter borrows tmp1/tmp2/residual/effective only after the final frame has
been exported and its solver status checked, with no published stationary-weight
lease occupying that scratch. W and, for UKS, total
D are restaged only after successful shared validation and an explicit drain.
The fitted projection's separate proposal slot and retained warm orbitals remain
untouched. KS host W retains its original orbital-energy expression; the HF
physical-Fock weighted callback is deliberately not substituted.

## Layout and numerical boundary

The real captured 96-atom metric and Fock are not bitwise symmetric: maximum
asymmetries are approximately 4.235e-22 and 3.309e-15. Therefore resident buffers
cannot be interpreted by silently equating A with its transpose. KS exports
operator buffers as row-major, while exporting C requires a transpose from its
column-major resident frame. An explicit internal transposed-operator flag drives
both GEMM operands and scalar-reduction indexing; C remains column-major.
The existing DF consumers retain the default column-major interpretation.

All physical commutator, density drift, electron trace, idempotency, eigen/metric,
absolute canonicality and KS component-energy gates remain unchanged. Product
reductions keep the existing compensated signed sums and hypot norms. The
independent oracle remains outside production. Device products returning invalid
evidence fail validation; a runtime/provider exception is not an invitation to
retry secretly on the reference path.

## Admission and fallback

The default remains off pending qualification. Missing admitted partial storage
or a matrix-library handle uses the existing reference validation route.
Small-AO owners therefore retain their bounded fallback. Invalid selector values
are errors. CUDA graph capture is rejected, all submitted work drains on exceptional
exits, scratch/input overlap is rejected, and failures revoke final-state/force
eligibility. The reservation is visible even when the optional library route is
unavailable; there is no uncharged lazy allocation at snapshot export.

A published stationary W/total-D binding is valid while its token is current.
Consequently an already-ready weight lease also blocks scratch admission: a
repeated read at the same generation uses reference products without writing
the borrowed buffers. Restaging equal values would not make concurrent writes
safe, and token equality alone does not grant permission to recycle storage.

## Qualification boundary

The build candidate is isolated from the rejected ragged-XC experiment: start
from `9c77456aea4b54b2c14080331e9bf1522b86091f`, copy only validation changes,
and reverse the two incidental frozen V3 hunks in cuda_ks/CMake test registration.
No previously measured source or artifact is overwritten. Retain a complete
candidate source manifest, verified compiler-cache version/statistics and actual
native/AOT hashes before real-device execution.

Native qualification must cover independent analytic nonidentity-metric rotations,
both spin domains including empty beta, row/column operator orientation, stale
generation/solver failure, unsorted/overflow products, scratch canaries/aliasing,
capture rejection and the shared KS rejection policy. Shared DF tests protect
the unchanged default layout. Full original/moved PBE0 E+F at both 48 and 96
atoms, independent references, actual dispatch/work counts and all raw timing
samples are still necessary before a performance phase PR or default promotion.

Evidence root: `.artifacts/ks-validation-stage/`.

## Terminal qualification receipts and retained failures

V4 CPU-only Slurm build job 7204 completes with exit 0 and passes the host KS
contract tests. Its isolated patch SHA256 is
`f56e715fab8d61ae9eda75154cd71cdf87693bc40d5e49387570a91796ea0121`;
the full-source archive SHA256 is
`015e567e8430053b0784f807ea9155ffe12b40e53ce1b8ee9057b4ea309b9fcc`.
The resulting native library SHA256 is
`b9983da20db38d3345726b7727c698a5cd9ab47d2dc126d4b6b3eada1ad61594`.
The packaged stationary force library stays byte-identical to earlier variants,
SHA256 `969d92da3008b710c287c673d778194485443ba2517f5b0b334d99bdf89192b5`.
Verified ccache 4.5.1, before/after statistics and actual launcher commands are
retained. The shared cache's global counter deltas include concurrent work and
are not this build's private hit rate.

V3 native job 7196 passes the independent resident analytic suite and the shared
DF validation suite, but the broader CUDA KS suite fails its existing PBE0
density-provider admission-route assertion at seven AOs. Focused job 7203 passes
memcheck with zero errors and racecheck with zero hazards/errors/warnings. The
same candidate binary with the new selector disabled also fails that identical
admission assertion. Seven-AO owners have no admitted partial reservation, so
the new device proof cannot run there. This is a retained disabled-selector
control, not a pristine-HEAD binary proof and not an all-green KS suite.

V4 job 7208 again passes the focused resident and shared DF suites. Its adapter
and reduction sources are byte-identical to the V3 sanitized sources; the
changed production behavior is the published-weight scratch-admission guard.
Do not repeat the unchanged sanitizer matrix solely for a source-directory or
linked-library hash change. Exercise that guard through repeated snapshot
exports at the complete owner instead.

The first endpoint harness attempt, also job 7208, fails construction for both
48 and 96 atoms because its constructor wrapper reads `export_work` before the
original lazy `decode()` populates that attribute. Both failed JSON records and
logs remain under `ks-validation-endpoint-v4`. This is an instrumentation defect,
not numerical or performance evidence. The next harness passively observes the
return of the original decoder, retaining the constructor arguments for an
untimed repeated-export/lease probe. Production source and artifacts are not
rebuilt or changed to fix the observer.

Observed master at this boundary is
`7685c251ded6c986b3cb63336e3c1cd31c0cd739`. Its new signed rank-k capability
and mixed-f force-class partition do not change this final-validation adapter;
the retained H/O def2-SVP endpoint has no f shells. Master advances trigger
dependency inspection, not a blanket numerical or GPU requalification. Any
submission-source reconciliation still needs relevant endpoint evidence rather
than transferring old snapshot timings to current master by assertion.

## Complete endpoint result, source V4 and observer V5

Finite Slurm job 7209 completes with exit 0 at
`2026-10-10T22:34:25+08:00`, preserving assigned visibility 0. It runs the
unchanged V4 production source and binaries. Five interleaved baseline/candidate
pairs are retained for each original/moved geometry at both 48 and 96 atoms.
There is no new GPU4PySCF timing contest: its retained independent E/F reference
is the numerical oracle for our two validation schedules.

| Atoms / AOs | Geometry | Reference proof median (s) | Resident proof median (s) | Reduction | Robust gate |
| --- | --- | ---: | ---: | ---: | --- |
| 48 / 384 | warm | 4.881321423 | 4.822977915 | 1.195% | below 2%; not significant |
| 48 / 384 | moved-warm | 4.920020342 | 4.820484616 | 2.023% | pass |
| 96 / 768 | warm | 14.691069994 | 13.755493056 | 6.368% | pass |
| 96 / 768 | moved-warm | 14.699058909 | 13.759123527 | 6.395% | pass |

These are complete synchronized E+F host-return times, not validation-only or
kernel-duration sums. All raw samples, including the slower 48-atom baseline
sample, remain in the assessment. The original 2% / twice-relative-MAD gate is
unchanged. Do not rerun the 48-atom warm cohort until it happens to qualify or
call the entire four-workload performance portfolio a pass. The 96-atom gain is
real in this frozen source; it is not a current-master performance claim or
authorization for broad default promotion. Default remains off.

The local independent analyzer recomputes all 40 timed E/F comparisons from
the reference arrays, checks actual checkpoint-file and numeric-blob hashes,
and confirms the original assessments from all raw timings. Maximum energy
and force errors are respectively 6.139090e-12 Eh / 2.467537e-11 Eh/Bohr at
48 atoms and 8.640200e-12 Eh / 3.755424e-11 Eh/Bohr at 96 atoms. The unchanged
acceptance gates remain 1e-8 Eh / 1e-7 Eh/Bohr.

Every timed call retains one SCF iteration and one Fock build. Observed SCF AO,
force AO, grid/point and force semantic counts match at each geometry across
both arms and every sample. These are the available semantic counters, not an
invented census of screened J/K quartets: capacity counts are not executed
quartet counts. Packaged stationary AOT and native-build grid admission are
authenticated by the original owner harness; the AOT force binary is unchanged.
The native library hash matches the V4 build receipt in both arms.

Each admitted candidate export adds exactly 112 diagnostic bytes and uses
three drains versus the reference's one. Four untimed repeated exports per
case authenticate the same owner/epoch/density/orbital identity and retain
legacy bytes, one read and one drain, proving the published-weight admission
fallback is exercised. All eight per-case frozen warm-density/coordinate
checkpoint files survive byte-hash and before/after blob-hash verification.

Receipts are under `.artifacts/ks-validation-stage/evidence/endpoint-v5/`;
`independent-analysis.json` records independent gates and scoped work parity.
The failed observer V4 records, V2 unsigned-fixture analytic failure and V3
broader KS admission assertion/control remain separate retained evidence, not
silently discarded qualification attempts.

During this cohort master advances to
`ec04f1301e516d7f33031f43ae28d609b85fa0a3` with shared CPU JIT artifact
provenance. Dependency inspection finds no change to the relevant CUDA KS,
final-state adapter, grid or force consumers. This observation does not trigger
additional GPU tests. Submission-source reconciliation and its relevant
validation remain outstanding before a phase PR.

## Submission-source ownership alignment

The phase submission reconciles the existing feature history with master
`c820ad08de1ceb33bc19c79115e0af2d52b01784` in an isolated detached worktree.
The original rejected ragged-XC working tree remains untouched. Its production
reconstruction patch, before this rationale/evidence packaging, has SHA256
`d44b766d753414cf7c2fdc1895b6dbac999901ebf1ada7069971c0c1eb5ee5ad`.

The measured V4 adapter submitted cuBLAS directly, which would introduce new
method-local vendor ownership rather than reuse the retained tensor boundary.
The submission therefore routes the identical dimension/rank/transposes,
leading dimensions and unit/zero scales through the shared already-admitted
`tensor::cuda::square_panel_product` primitive. The existing KS matrix-library
owner still supplies its handle, stream, modes and workspace. No new tensor
context, dense allocation, provider discovery or scientific formula is added.
The square-padded reduction rank preserves occupied work without repacking.
Vendor-boundary checks pass without classifying a new SCF migration exception.

This ownership change requires fresh focused adapter/DF/safety and complete
endpoint evidence for the reconciled source; old V4 timings are not transferred
by assertion. Unrelated CC, CPU JIT and evidence-storage master changes do not
justify another broad GPU qualification matrix.

## Submission-source qualification

Finite build job 7218 and GPU job 7222 qualify the reconciled submission source.
GPU job 7222 preserves assigned visibility 3 on node1 and completes at
`2026-10-10T23:18:50+08:00`; the later independent download/analysis completes
at `2026-10-10T23:35:23+08:00`. Host KS, independent resident analytic products
and shared DF contracts pass, with memcheck zero errors and racecheck zero
hazards/errors/warnings. The submission native SHA256 is
`98cd00a8db9a19895f90b2a6e3a87efe2dd0da3d31a3bbdde4a43a566c3ae0e6`.
The stationary AOT binary remains byte-identical to the prototype.

| Atoms / AOs | Geometry | Reference median (s) | Resident median (s) | Reduction | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| 48 / 384 | warm | 4.876250710 | 4.740550406 | 2.783% | pass |
| 48 / 384 | moved-warm | 4.890742093 | 4.791956812 | 2.020% | pass |
| 96 / 768 | warm | 14.595477730 | 13.687022794 | 6.224% | pass |
| 96 / 768 | moved-warm | 14.526292738 | 13.641281232 | 6.092% | pass |

Both complete cohorts retain five interleaved pairs per geometry and all slow
samples. The historical 48-atom warm result remains below gate. Cohorts are not
pooled, and the wrapper alone is not asserted to cause their difference. The
submission's 40 timed records pass independent E/F gates 1e-8 / 1e-7, scoped
semantic-work parity, one-iteration/one-Fock replay and authenticated frozen
checkpoints/weight-lease fallback. Maximum submission E/F errors are
6.366463e-12 / 2.467937e-11 at 48 atoms and 9.094947e-12 / 2.961187e-11 at
96 atoms. Admitted exports still add exactly 112 bytes and two drains.

The broader KS suite retains `PBE0 density provider admission route` for both
selectors in this same submission binary. Its seven-AO owner cannot admit the
device proof. This disabled-selector control is not a pristine-master binary
comparison, and no all-green broader KS claim is made. Default remains off.

The compact source-matched publication is
`benchmarks/results/ks-resident-final-validation-20261010/`. It preserves all
80 timed old/new records, independent analyses, references, failed attempts,
source patches and finite scheduler/cache receipts. Actual ignored NPZ files
were verified independently before packaging; digest receipts do not replace
those files for a new checkpoint analysis.

At the October 11 publication boundary master is
`4ac8d4517f3323c5b7ba567270bdb2bd3323cde7`. Its newer HF workspace/device
queries and inactive-output matrix-library correction touch neighboring owner
interfaces. They are not included in the measured source and do not acquire
its evidence by assertion. Any integration must retain those corrections;
this observation does not justify an unrelated full GPU qualification matrix.

Publication fits the unchanged 64-MiB aggregate and 2-MiB change-review budgets
by losslessly compacting nine existing density-candidate/psss sample JSON
members with the shared compaction tool (70-KiB per-invocation threshold).
Decoded arrays, sample order and scientific decisions remain unchanged; only
storage paths, attachment hashes and the large-evidence review inventory move.
