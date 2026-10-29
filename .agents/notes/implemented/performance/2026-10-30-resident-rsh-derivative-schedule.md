# Decision: retain specialized RSH derivatives across resident source transport

Status: implemented
Date: 2026-10-30

## Problem

The resident Direct RSH force adapter always selected the generic fused radial
traversal. The existing host adapter already decomposed the canonical omega=0.3
case into specialized full-range J/K and long-range exchange. Moving the result
transport to the stationary device owner had lost that scheduling decision.
The stationary receiver also lacked a full-range K destination for RSH profiles.

On an A100, a 12-atom WB97M-V diagnostic attributed 2.486485 seconds to
`direct_rsh_force_resident`, despite one warm SCF iteration. Clean complete warm
energy-plus-force calls took 4.053219 seconds. This was a performance defect in
the resident path, with no numerical discrepancy in the independently checked
endpoints. The diagnostic CUDA-event ledger is retained; Nsight's importer failed
with an event-order error, so it does not establish a kernel census.

## Decision and invariants

For omega=0.3 and a nonzero LR correction with a finite SR/correction ratio,
publish `FullRangeJk` followed by `RangeCorrection`. Both providers use the
existing prepared owner and stream. Views expire when their callbacks return:
the receiver must enqueue the full-range copies before the next traversal
reuses provider storage. No host force array, extra integral owner, new scalar
algebra, GPU-model allowlist, or inherited tuning claim is introduced.

The RSH stationary receiver temporarily puts full K in the short-range slot.
Its existing correction transport subtracts the scaled LR contribution from
that slot and fills the long-range slot. It marks only the short-range source
as present until the correction arrives, preserving incomplete/duplicate source
rejection. Force-to-gradient signs and the model's SR/LR coefficients remain
owned by the same transport.

Other omega values, a zero correction, and nonfinite recomposition ratios retain
the complete fused provider. A failed provider propagates failure; it cannot
publish a successful complete endpoint with only the first source. The omega
boundary describes the existing qualified radial specialization, not a hardware
restriction. Broader admission requires separate radial qualification.

## Rejected alternatives

- Returning to host force arrays would restore unnecessary transfers and weaken
  callback-scoped source ownership.
- Reimplementing LR mathematics or tuning the generic fused traversal would
  duplicate already qualified providers before recovering the lost schedule.
- Promoting partial, unchecked large-system timings would hide missing work or
  oracle gates. Such attempts remain diagnostic records only.

## Qualification

The ECNU campaign uses finite LSF allocation 216451 on gpu22, queue `gpu_chem`,
eight CPU slots and one exclusive-process A100 PCIe 40GB. CUDA visibility is
preserved. Native builds use CUDA 12.9.86, GCC 14.4, Release SM80 cubins and the
portable profile with the existing verified ccache. The reference uses
GPU4PySCF 1.7.6, PySCF 2.13.1 and CuPy 14.1.1 without cuTENSOR on the same
allocated GPU. The task-private glibc/runtime environment does not replace
system libraries. Its native test executable needs a task-local `-rpath-link`
to CUDA libraries and is invoked with the private loader; this is not a
production CMake change.

The charged UKS water check uses frozen spherical def2-TZVP through f,
charge +1, multiplicity 2 and 24x8x16 quadrature. Original and moved geometries
pass independent reference gates: maximum energy error 7.96e-13 Eh and maximum
force error 1.65e-10 Eh/Bohr. Reconverged directional differences at steps 1e-3
and 3e-4 Bohr have errors 1.54e-8 and 1.54e-9 Eh/Bohr (gate 2e-6).
The strict reference needed 186 SCF cycles on the original geometry. Its first
two-geometry run timed out after preserving that complete reference; a separate
bounded run completed only the missing geometry, recording the original receipt
hash. No native density seeds the reference. Earlier task-script type errors
and the incomplete timeout receipt remain in the raw evidence.

The endpoint driver retains cold setup plus energy/analytic forces, three warm
calls, a moved geometry and three moved-warm calls. Standard cases use frozen
spherical def2-SVP; r2SCAN-3c retains prescribed def2-mTZVPP, D4 and gCP. PBE
uses its maintained pruned grid; the other RKS cases use 48x16x32 quadrature.
WB97M-V retains LR exchange and VV10. Gates are 1e-8 Eh for energy and 1e-7
Eh/Bohr for maximum force error against independent references.

Matched grids diagnose implementation agreement. They do not establish
grid-converged production accuracy or an overall engine ranking. Reference SCF
counts differ and remain visible; comparisons must not equate an iteration
with an identical amount of scientific work.

Clean WB97M-V before/after complete endpoint times (seconds) are:

| Atoms | Phase | Previous resident path | Specialized resident path |
| --- | --- | ---: | ---: |
| 3 | cold | 3.037302 | 2.920843 |
| 3 | warm median | 0.427299 | 0.290524 |
| 3 | moved | 1.716922 | 1.596461 |
| 3 | moved-warm median | 0.431217 | 0.291712 |
| 12 | cold | 13.066112 | 11.102799 |
| 12 | warm median | 4.053219 | 2.095206 |
| 12 | moved | 10.253613 | 8.316820 |
| 12 | moved-warm median | 3.991309 | 2.104016 |
| 48 | cold | 163.587556 | 96.850466 |
| 48 | warm median | 92.033405 | 26.019429 |
| 48 | moved | 152.463118 | 86.204287 |
| 48 | moved-warm median | 92.191169 | 26.104878 |

All 24 fixed records pass. Native SCF counts are unchanged between these arms;
each warm call uses one Fock. Raw samples, AO work and nonlocal-work receipts
are retained. These are one fresh process per arm with three warm repeats,
not a broad hardware or startup-variance study.
The 48-atom cold calls both use 28 iterations/29 Focks, and moved calls use
19 iterations/20 Focks. All eight records in its complete baseline pass;
the earlier interrupted, unchecked attempt does not supply these timings.
A second fresh fixed 48-atom process gives cold 97.6273 seconds, warm median
26.0665 seconds, moved 86.3067 seconds and moved-warm median 26.0734 seconds.
All eight gates and the same iteration/Fock counts pass again. Its samples
remain a separate repeat rather than replacing the first fixed population.

Paired CUDA-event diagnostics without Nsight attribute 2.4404 seconds to the
old 12-atom resident RSH derivative operation. The fixed path uses 0.1535
seconds for full J/K and 0.3077 seconds for isolated LR. Both diagnostic
cold/first-warm records pass the independent gates. The complete instrumented
warm calls take 4.1615 and 2.1781 seconds; clean endpoint evidence above supplies
the speedup claim. Initial profile attempts requested one repeat against a
three-repeat reference and were correctly rejected for protocol mismatch;
the corrected diagnostics retain the exact reference protocol and deliberately
stop after its first warm return.

For r2SCAN-3c48, a diagnostic warm endpoint takes 8.2565 seconds. Its resident
full-range derivative operation takes 3.1159 seconds; within that operation,
the low/order-6 region takes 2.8993 seconds, order-7 0.1050 and order-8 0.1111.
The SCF iteration takes 0.9390 seconds, including 0.8692 seconds for XC and
0.0593 seconds for pure J. Nested inclusive regions must not be summed twice.
These intervals identify substantial work outside the repaired RSH schedule;
they do not prove another numerical bug or fully attribute the remaining wall
time. In particular, the uninstrumented stationary XC/correction work cannot
be assigned to a specific kernel from this ledger alone.

The restored portable baseline also passes complete 3/12/48/96-atom matrices
for PBE, B3LYP, r2SCAN, r2SCAN-3c and M06-2X. The final RSH library replays
all six non-RSH functionals at 48 atoms, including PBE0, plus PBE0 at 96 atoms.
These 56 final-library non-RSH records pass. Omitted unchanged-path repeats
have explicit `not_run` receipts; the final library is not claimed to have
replayed every baseline size. At 48 atoms the complete warm medians are:

| Functional | Final native (s) | GPU4PySCF (s) | Native iterations | Reference iterations |
| --- | ---: | ---: | --- | --- |
| PBE | 1.8890 | 2.2570 | 1/1/1 | 4/4/4 |
| PBE0 | 3.3606 | 3.6377 | 1/1/1 | 3/3/3 |
| B3LYP | 3.4282 | 3.6776 | 1/1/1 | 3/3/3 |
| r2SCAN | 3.0231 | 4.8430 | 1/1/1 | 4/4/4 |
| r2SCAN-3c | 8.2631 | 6.3915 | 1/1/1 | 4/4/5 |
| M06-2X | 3.7231 | 3.6417 | 1/1/1 | 1/1/1 |
| WB97M-V | 26.0194 | 25.8394 | 1/1/1 | 1/1/1 |

The remaining r2SCAN-3c and M06-2X warm gaps are retained, not described as
resolved by the RSH fix. Reference Fock counts and all samples accompany the
iteration counts; a reference warm iteration may perform two potential builds.
The 96-atom WB97M-V reference completed its cold geometry in 1103.298 seconds
and 49 iterations, then exceeded the 1200-second process bound during its first
warm call. Its saved cold energy/forces are usable as a geometry oracle, but it
does not qualify a complete cold/warm/moved reference population.
A separate bounded, instrumented native run uses only that complete cold
geometry oracle. Its cold/first-warm records both pass: 365.9791/92.2220 seconds,
28/1 iterations and 29/1 Focks, maximum energy error 2.64e-11 Eh and maximum
force error 2.35e-11 Eh/Bohr. This diagnostic contains no moved geometry and
does not replace the missing complete 96-atom RSH population. Its warm SCF
iteration takes 6.0892 seconds, full J/K derivatives 6.1133 and isolated LR
7.3785; these separate top-level intervals do not account for the full
stationary XC/nonlocal force work.

## Cross-device comparison scope

The retained 2026-10-26 README RTX 5090 cohort is not a matched hardware control.
It uses source `46b916ac`, level-3 pruned Treutler/Lebedev grids, looser native
and reference convergence, one math thread, GPU4PySCF 1.8.1/PySCF 2.14.0 and
CuPy 13.6.0. It also explicitly sets XC spatial order and validation GEMM
controls. The ECNU campaign uses the current source, eight math threads,
GPU4PySCF 1.7.6/PySCF 2.13.1 and CuPy 14.1.1, and full-density reference Fock
rebuilds. For 48-atom WB97M-V, native XC/nonlocal grids contain 539168 versus
1179648 points. Grid prescription and active nonlocal work matter in addition
to the point-count ratio; neither the old seconds nor nominal hardware peak
rates establish a regression between devices.
The new paired source builds retain `GENERATIVEQC_PROFILE=off`, CUDA 12.9.86,
ordinary Release compilation, the portable profile and the same CuPy/NumPy
versions. No SM120 measured tuning claim is applied to A100. An existing
SM120-only MD-J `launch_bounds` compiler hint does not gate kernel admission
on other architectures and is not the measured resident RSH dispatch defect.

The matching SM120 binary was compiled with GCC 12.3 and CUDA 12.9.86, then
transferred to the permitted n4 host. Its seven library hashes and complete
production source inventory were checked there before execution. Slurm job
1362 uses one RTX 5090, eight CPU cores, 32 GiB host memory and a 40-minute
bound, preserving `CUDA_VISIBLE_DEVICES`. The local maintenance-node request
was canceled before execution; no GPU test bypassed its scheduler. A missing
Python runtime alias initially prevented interpreter startup on n4; that failed
setup remains separate from the measured population.

Completed controls use the same protocol, source identity and portable profile:

| Case | A100 warm median (s) | RTX 5090 warm median (s) | 5090 / A100 |
| --- | ---: | ---: | ---: |
| WB97M-V, 12 atoms | 2.095206 | 5.222887 | 2.49 |
| r2SCAN-3c, 48 atoms | 8.263127 | 13.885904 | 1.68 |
| PBE0, 96 atoms | 10.984577 | 20.224107 | 1.84 |
| WB97M-V, 48 atoms | 26.019429 | 75.203750 | 2.89 |

All 32 energy/force records in the four completed 5090 controls pass. Every
compared warm call uses one iteration/one Fock, with equal retained AO,
Coulomb/exchange and nonlocal work. The 5090 job completes with exit code zero.
PBE0-96 cold convergence takes 18 iterations on A100 and 17 on 5090; cold
seconds therefore also include that observed work difference. Host CPU,
compiler, OS, driver and GPU resources differ, so these are complete endpoint
observations under matched scientific conditions, not isolated GPU peak-rate
measurements or an all-workload hardware ranking. The old figure alone does
not establish an A100 regression, and remaining gaps against GPU4PySCF are a
separate comparison.

The separate 5090 cold/first-warm component diagnostics also pass their gates.
Its WB97M-V12 full-J/K and isolated-LR derivative operations take 0.1430 and
0.1713 seconds, and r2SCAN-3c48 full-range derivatives take 2.5278 seconds.
Those individual stages are faster than on A100 even though the complete
endpoints are slower. Their nested CUDA-event intervals must not be combined
with host wall intervals or treated as complete endpoint timing. In particular,
this evidence does not assign all remaining stationary XC/nonlocal force time
to a single kernel, nor establish that one GPU wins every component.

The compiled host adapter fixture checks RKS/UKS coefficient reconstruction,
zero SR, noncanonical omega, zero/subnormal correction, invalid scientific
identity, provider failures and destructive callback-scoped storage reuse
(42 cases). It is a transport test, not an integral oracle. The actual A100
native binding executable passes all eight built profiles, including RKS/UKS
RSH source recomposition, destructive provider scratch reuse, incomplete and
duplicate correction rejection, arena admission and lifetime checks.

The final source passes 1,000 host checks across the resident adapter, native
stationary binding, shared stationary kernels and Direct RSH callback contracts.
Selected pre-commit checks pass. The baseline used for the RSH timing comparison
already includes the restored portable J/K work, isolating the resident RSH
change from that earlier optimization.

## Provenance and follow-up

Ignored raw evidence is under `.artifacts/ecnu-a100-functionals-20261029/`.
The baseline production inventory is
`b2ae0550b2e95eff39850166b2ad4a2e8e01ba8ba6e4f9df5142ee7605420d83`;
the corrected production inventory is
`e2f28f36363baaa5da97abe7062a32b1ea4d853f5d39b8ff5cecb580986fd53a`.
Binary hashes, source inventories, dependency hashes, inputs, scheduler and
build receipts accompany the endpoints. Failed environment setup and interrupted
unchecked measurements remain separate from qualified complete cohorts.
The downloaded final A100 archive `functionals-evidence.tar.gz` is 2398033
bytes with SHA256
`6f33e2fb04aa9fd7cbd0240817bc0270910194edc75bf3cacd45558a6e9b0d88`.
Independent inspection finds 27 complete reference populations (216 records),
23 complete portable-baseline populations (184), 10 complete fixed populations
(80), and one separate fixed repeat (8), with every complete record passing.
The partial 96-atom reference, diagnostic populations, UKS qualification and
failed setup/profile attempts remain separately labeled.
The combined A100/5090 bundle `a100-5090-final-evidence.tar.gz` is 2840753
bytes, SHA256
`57bd70c0690e3490ada8868122209134d1cfb259770a3c431e634103127b88f3`.
Its 75 members include the A100 archive, complete 5090 populations and
diagnostics, matched-work comparison script, build/source/library inventories,
cache/build logs, scheduler completion, host checks, CI resource evidence and
retained setup failures. Every archived member was independently hash-checked.
Native libraries remain in the ignored local build; no release or external
backup was published. The source CI at `6a73a4c0b` is fully green; the subsequent
evidence-only commit changes this note, not the measured production inventory.

PR #2880 was merged during qualification, but its master tree omitted the
final portable generated-J/K/cold-reuse changes from the earlier tested branch.
The follow-up restores those unchanged, previously qualified paths as well as
this RSH fix. See [the portable J/K rationale](2026-10-29-portable-rys-jk.md)
for that separate implementation and its complete numerical/performance evidence.

The restored portable manifest also requires deliberate refresh of the existing
SM120 stable-shard resource ledger. [CI job 123602639166](https://github.com/jinzhezenggroup/generativeqc/actions/runs/40917805079/job/123602639166)
compiled the qualified shard with its pinned CUDA 12.9 production toolchain and
flags: all 102 kernels retain identical REG/STACK/SHARED/LOCAL, and every other
identity field matches. Its sole failure was the full manifest hash, because the
portable profile changed. Regenerating the ledger with the validator's
`measured_ledger` from this fresh measurement changes only `manifest_sha256`;
no resource bound or integrity gate is relaxed. Report SHA256:
`6a86abc60fe812acda0edb314d14c4abf780eab3d4d407142228c7ab26847c08`.
The CI report, compile/cache receipts and cuobjdump output are retained locally;
all 268 resource-ledger validator tests pass.

Revisit the canonical boundary when another radial schedule has independent
force qualification. Revisit the transport if provider streams, source lifetimes
or stationary slot ownership change. Any later optimization must retain complete
energy-plus-force gates and disclose SCF, J/K and nonlocal work.
