# Decision: qualify local SCF AO maps for exact-direct RKS-PBE0

Status: implemented; standalone complete24/96 qualified, qualified local-AO default promoted in follow-up
Date: 2026-10-04

## Problem and decision

Native SCF already has compiler-owned local density and potential contractions,
geometry-bound AO discovery, resource accounting and a bounded dense fallback.
Its ordinary admission admitted WB97M-V only. PBE0 still contracted every grid
block against every AO, even when its basis tails were negligible there.

Extend local-AO admission to all-electron, exact-direct, restricted PBE0 in device-fused FP64 execution. The initial qualification used `GENERATIVEQC_CUDA_KS_ACTIVE_AO=1`; after the completed endpoint campaign, qualified local-AO selection is promoted to the ordinary default. The predicate checks the PBE
semilocal family, exchange/correlation scales 0.75/1, full-range exchange
coefficient -0.125, and absence of DF, ECP, range and nonlocal corrections.
WB97M-V keeps the same admission predicate and now follows the same automatic qualified default. `GENERATIVEQC_CUDA_KS_ACTIVE_AO=0` remains an explicit debugging opt-out. This is not admission for arbitrary PBE hybrids or unrestricted PBE0 SCF.

The existing owner discovers value/first-derivative AO maps at cutoff 1e-16,
once per prepared geometry/grid, and reuses them for XC builds. It gathers
the density submatrix and scatters local potential contributions through the
existing generated kernels. No XC algebra, precision, force AO maps, Becke
response, J/K screening or SCF convergence policy is changed here. Existing
host/device budget declines retain dense execution; selection is observable
through the existing AO diagnostic, rather than presumed from the switch.

Matrix work changes from `G M^2` to `sum_b G_b m_b^2`, with one-time
`O(G M)` discovery and geometry-bound CSR storage. These are work proxies,
not FLOP counts or a claim of a different asymptotic rate for every molecule.
Discovery and map storage must be counted again after a geometry rebuild.

## Evidence and scope

The motivating experiment is **not a build of this admission patch**. It used
the frozen #1841 library at `5de22fd9a377f43536a31866456f08c2187b9752`, source
identity `18f397771932744c7c438e2469241c024b09deb9d44a3a81167d294b6cd7c602`,
library SHA256 `ec184b64996c856932fe1d9401cda4c38b1a42fcdd20e625aa41a8bbcc09e81c`.
The probe called `CudaXcPlan` directly, bypassing SCF admission. n1 RTX 5090,
finite Slurm job 5745, 96 atoms, 768 spherical AOs, def2-SVP, 2,359,296 grid
points, 256-point tiles, scaled-PBE semilocal E/V only, fixed resident density.

Three alternating pairs after warmups gave dense/local medians
10.212886419 / 2.300512154 seconds. These include XC submission and E/V
readback, not geometry setup, SCF, J/K or forces. Energy disagreement was zero;
maximum potential disagreement was 4.440892098500626e-16. Eight independent
small CPU integration gates covered Cartesian/spherical, RKS/UKS component
execution, cutoff 1e-16 and entirely empty maps. UKS component evidence does
not qualify an unrestricted SCF trajectory.

Actual selected/dense matrix work proxies were 75,674,112,000 /
1,391,569,403,904. There were 9,216 tiles, 1,460,424 total active columns,
768 empty tiles and a maximum of 536 active AOs. Selected setup cost
1.899793115 seconds, including 1.898161623 seconds of discovery. The selected
plan reserved 69,340,448 device bytes and charged 56,699,912 host peak bytes;
these are whole-plan/discovery bounds, not additional bytes versus dense.

Raw receipts remain under
`/data/jzzeng/qc-pbe0-force-compact-pages-20261004/.artifacts/xc-local-5745/`
on n1 and its retained local copy. None of these timings is latest-master
complete E/F evidence. The production admission patch starts at master
`daa932853e37b742008b54cc6e348484f9ad7034`.

## Acceptance and rejected alternatives

The host capsule compiles the actual admission block with synthetic owner
facts; it tests policy only, not CUDA concurrency or chemistry. The native
`--pbe0-local-ao` cases independently integrate full-AO scaled PBE on CPU,
including off-diagonal positive densities, through-f AO tails, empty maps,
host-budget decline and output canaries. New-source GPU qualification, then
same-source/GPU cold, warm and changed-geometry complete E/F comparisons, are
required. Every call must satisfy the README energy/force gates (1e-8 Eh and
1e-7 Eh/Bohr); record retries, final builds and selection diagnostics, not just
the final successful iteration count. Discovery belongs in cold/moved timing.

Do not promote the switch from component timing. Do not multiply this result
by #1830 or #1833 speedups: those modify force execution, and their composition
requires fresh complete-endpoint evidence. Do not use this work to declare the
generic force consumer solved; #1841's compact schedule regressed despite a
smaller linked stack. It remains default-off while separate recurrence work
addresses that larger hotspot.

## New-source component qualification

The standalone native XC executable built from `c99e524b3` passed both
`--pbe0-local-ao` and the adjacent `--ao-discovery` gates on n1 RTX 5090 in
finite Slurm job 5749. The former includes the positive off-diagonal density
cases added by this patch. Compute Sanitizer memcheck and synccheck repeated
that gate with zero errors. Executable SHA256:
`c54f4cdb7a6ccf1dde4d3b5aaf5ac83f419f8b2621796196717070ca93ddf670`.
Raw receipts are `.artifacts/component-5749/` in the local/remote
`qc-pbe0-scf-local-ao-20261004` checkout. Documentation-only follow-ups do not
change the production source identity
`7f0770695009fee99ed7a4b08f55f1367eb52d81fbc204bf065aa47b26511358`.
This standalone executable does not exercise `CudaKsPlan` admission or a
complete SCF trajectory. Those still require the production-library endpoint
campaign; no new complete endpoint speedup is claimed here.

## Production-library qualification progress

The ccache-verified Release/sm_120 build completed on n5 using CPU compilation
only. Library SHA256 is
`d95626509e4eb44756947138df156d91217e0c7a3cb794662527ebcda007666d`;
the production identity remains `7f0770695009fee99ed7a4b08f55f1367eb52d81fbc204bf065aa47b26511358`.
The actual production library passes native KS plus the component/sanitizer
gates on n1. Slurm5750's first full24 force attempt failed because the benchmark
environment omitted its `ptxas` wrapper, not because of a reported numerical
gate. Its raw failure is retained. The retry adds executable/version/hash
preflights for all three compiler wrappers without altering production source.

Finite n1 Slurm5752 has completed fresh independent GPU4PySCF and native
local-OFF/ON full24 cold, warm, moved and moved-warm endpoints. Both native
policies pass every same-geometry reference pairing, with maximum energy
error 5.003e-12 Eh and force error 2.466e-11 Eh/Bohr. The reference reports CUDA
LibXC. These 24-atom rows have only one replay, not five-repeat medians:

| Phase | Dense SCF (s) | Local SCF (s) | SCF iterations OFF / ON |
| --- | ---: | ---: | ---: |
| Cold | 271.528225 | 267.985695 | 21 / 21 |
| Warm | 4.992399 | 4.827158 | 1 / 1 |
| Moved | 26.856837 | 25.454883 | 12 / 12 |
| Moved-warm | 4.997141 | 4.825056 | 1 / 1 |

Separate stationary caches but a retained shared ccache and ordered runs do
not establish a causal cold speedup. All rows report no warm-start fallback;
the endpoint's native Fock-build counter is absent, not zero. Actual original
geometry SCF selection uses 2304 tiles, 284 empty, 271332 total active columns,
and matrix-work proxy 9,896,403,968 versus dense 21,743,271,936. Discovery takes
0.233330 s; moved geometry rebuilds it in 0.213669 s. The selected owner reserves
5,850,464 device bytes and charges 3,558,152 host peak bytes.

The adjacent suite on the real checkout reports 226 passed, five failed and
one skipped. All five failures reproduce against the older frozen library in
Slurm5751: four custom-grid force cases request unsupported partition iteration
2, and one failed-preparation test expects an unraised resource exception.
The earlier sixth failure was missing Git metadata in the archived checkout
and passes after restoring a real checkout. Do not report the adjacent suite
as green or fix these unrelated failures in this patch.

Raw endpoints and validation are retained in `.artifacts/qualification-5752/`;
the original failed run and old-library reproduction remain in
`.artifacts/qualification-5750/` and `.artifacts/baseline-adjacent-5751/`.
Complete96 five-repeat OFF/ON qualification is still running. Integration
branch `codex/pbe0-integrated-scf-ao` composes the existing #1830/#1833 paths and
this admission on master `8c1233a5c`; it has host qualification only so far.
Neither that composition nor these partial results justify default promotion.

## Completed standalone complete96 qualification

Slurm5752 completed successfully at 2026-10-04 13:16:05 UTC, with the same
frozen production identity and library described above. Both OFF/ON policies
complete cold, five original warm, moved, and five moved-warm calls. Each
native96 run passes all 72 same-geometry pairings against the fresh independent
GPU4PySCF reference. Across both native96 policies, maximum energy error is
1.078e-10 Eh and maximum force error is 3.290e-11 Eh/Bohr, within the unchanged
1e-8 / 1e-7 gates. No native row reports a warm-start fallback.

| Phase | Dense SCF (s) | Local SCF (s) | SCF iterations OFF / ON |
| --- | ---: | ---: | ---: |
| Cold | 736.894027 | 501.525637 | 29 / 28 |
| Warm, five-repeat median | 76.113717 | 68.064636 | all 1 / all 1 |
| Moved | 251.161355 | 172.323781 | 12 / 14 |
| Moved-warm, five-repeat median | 75.922818 | 67.891332 | all 1 / all 1 |

Warm and moved-warm elapsed time decrease by about 10.6% in these ordered
same-source/same-GPU comparisons. This is complete E/F, not the earlier 4.4x
isolated XC ratio. Cold includes different JIT/cache histories and SCF work;
do not attribute its full difference to AO contraction. Moved geometry takes
**two more SCF iterations** with local AO, although elapsed time improves. This
negative trajectory difference remains part of the result.

The fresh reference96 warm median is 9.929292 s, with iterations 5/1/1/7/1;
moved-warm uses 7/6/3/3/3 iterations. The local-SCF-only endpoint is still about
6.85x the reference warm median. This PR does not solve the dominant remaining
force cost or establish parity. Do not substitute a reference from another job.

Actual original SCF maps have 9216 tiles, 768 empty, 1,460,424 total active
columns and maximum 536. The per-XC matrix-work proxy is 75,674,112,000 versus
dense 1,391,569,403,904, not FLOPs. Discovery costs 1.945800 s; the owner
reserves 69,340,448 device bytes and charges 56,699,912 host peak bytes. The
complete raw receipt retains moved-map regeneration and every sample's XC
evaluation count. Native Fock-build counts and the standalone integral-stage
wall observer remain unavailable (`null`), not zero.

The compact machine-readable receipt is
`benchmarks/results/pbe0-scf-local-ao-20261004/summary.json`: every warm sample,
SCF trajectory, map/memory count, reference J/K count, raw/validation SHA256,
protocol, source and binary identity. Raw data remain in the retained local/n1
`.artifacts/qualification-5752/` directories. Independent adjacent-test failures
and the failed first benchmark environment attempt above are not discarded.

The current-master composition at `d40aeafee` is a separate campaign, Slurm5757,
and is not included in these numbers. Its build, component/sanitizer gates and
coarse48 combined/fallback preflights pass; complete96 is still running. The
user requested stopping after the current optimization and PR are closed out;
no new integral-consumer optimization is started during this qualification.

## Completed same-source composition on master 8c1233a5c

Finite n1 Slurm5757 completed at 2026-10-04 14:25:59 UTC. Frozen composition
`d40aeafee0bb1e73fc130a9a253efdbf9beb31e8` contains #1830, #1833 and this PR
on master `8c1233a5c6f5dd0ff7d528a8c0772b9da06bf60f`; it does not include
the slower compact-force #1841 or paused incremental #1803. Source identity
`60f730f179cef1c1649435f39915f85774f76ecc16dd0b5308d0cd11e42a60c8`, library
SHA256 `948c0d6892c8ad3ab05020b4e19b76f221a89d14b1e26ad83af715193015802d`.
All three policies and the fresh reference run on the same RTX 5090 assigned
as CUDA_VISIBLE_DEVICES=1. This is a different physical GPU/campaign from5752;
do not combine its reference or stage timings with the standalone experiment.

All entries are complete full-grid 96-atom/768-AO spherical def2-SVP PBE0 E/F.
Warm rows are five-repeat medians; cold/moved rows are single calls.

| Phase | Default (s) | Force-local/indexed/phased (s) | Plus SCF-local (s) | GPU4PySCF (s) |
| --- | ---: | ---: | ---: | ---: |
| Cold | 751.783607 | 708.875197 | 479.354171 | 88.744315 |
| Warm | 77.366631 | 40.953721 | 32.512149 | 17.167006 |
| Moved | 271.620869 | 259.362597 | 137.329585 | 85.542494 |
| Moved-warm | 77.317810 | 41.108577 | 32.546870 | 10.080260 |

The composed candidate takes 42.0% of default warm time (2.38x throughput).
This PR's incremental contribution on that already optimized force stack is
40.953721 to 32.512149 s, about 20.6% less time. It is not credited with the
other force PRs' improvement. These are ordered same-source comparisons, not
ABBA causal attribution, and cold retains compiler-cache history differences.

Native cold iterations are 29/28/28 and moved iterations are 13/14/13 in table
order; every native warm replay takes one iteration. Reference original warm
iterations are 4/5/5/4/5 and moved-warm 4/1/3/1/1. Preserve that work difference:
the candidate is still 1.89x slower on the original warm median and 3.23x on
moved-warm, not parity. The force-only moved-warm sample of 44.884 s is retained,
not removed as an outlier. Reference XC components all report CUDA LibXC.

Every policy passes all 72 same-geometry independent pairings (216 total), with
maximum energy/force error 1.074e-10 Eh / 2.478e-11 Eh/Bohr. Native component,
KS, memcheck and synccheck gates pass; coarse48 combined and zero-force-cache
fallback full endpoints also pass, but their timings are not README points.
The first integration launch5756 failed before GPU tests because launch raced
the transfer of its SHA receipt; terminal failure and raw logs are retained.
The retry first verifies completed remote binary/head/wrapper transfer, with
no production source change. All jobs for this qualification have finished.

Actual warm work confirms the selected paths:

- All policies consume 9216 geometry batches. Phased execution reports 9216
  phased batches and reduces pair-primal evaluations from 21,516,779,520 to
  10,758,389,760. This is not a reduction of the O(G A^2) exponent.
- Force AO matrix-work proxy is 81,057,099,776 versus dense1,391,569,403,904;
  all9216 map lookups hit on warm replay, with1,516,176 total active columns,
  maximum553, and12,129,408 retained map bytes. It is distinct from SCF maps.
- SCF-local reports1,460,424 active-column sum, maximum536, work75,674,112,000,
  device reserve69,340,448 B and charged host peak56,699,912 B. Warm XC count is1.
- Composed force additional-device bound is396,778,368 B under536,870,912 B;
  additional-host numeric bound is213,824,928 B under268,435,456 B. These are
  planner bounds, not measured whole-process peaks or additive owner totals.
- Integral-source and combined semilocal-geometry warm timers remain about
  14.35 s and8.81 s respectively. The latter includes AO/XC geometry and Becke,
  not a Becke-only timer. Missing Fock counts remain null; indexed page counts
  still lack an endpoint observer. No task or work proxy is converted to FLOPs.

Machine-readable evidence is
`benchmarks/results/pbe0-scf-local-ao-20261004/integrated-summary.json.gz`, binding
every sample and validation file by SHA256, with actual work and resource
observers. Raw receipts are retained under the integration checkout's
`.artifacts/endpoint-5757/` on both local storage and n1. The composition branch
is evidence for existing PRs, not another scientific optimization PR.

At the close of this qualification campaign, production defaults were still unchanged; no merge or release was authorized. The user-requested next action after publishing these results was to pause, not to start another integral or scheduling experiment.

## Late branch synchronization: current head is GPU_NOT_RUN

While the completed evidence was being published, another writer synchronized
the PR branch with master in merge `4ce9514582aa55598baebbeeca306d476cc555f7`.
This imports the shared precision schedule and JIT-root pruning changes. The
merge is preserved; the evidence commit is rebased on top without rewriting
the synchronized production changes. Its strict-FP64 admission now queries
`any_lower_precision()` rather than `any_mixed()`. The host admission capsule
is updated to match that API, without changing its accepted/rejected cases.

The completed GPU campaigns qualify frozen `c99e524b3` and `d40aeafee`, not
this later synchronized head. No new GPU job is started to qualify the latter;
the PR remains Draft with current-head GPU_NOT_RUN status. Host-only results
after the capsule adjustment must not be reported as GPU requalification.
That synchronized-head handoff retained those explicit limits and paused at the user's request, without merging, enabling defaults or claiming parity.

The synchronized-head host admission and component-precision census suites
pass all 58 cases after the fixture adjustment. The initial stale-fixture run
reported 2 passes and 56 setup compilation errors; that log is retained beside
the successful rerun in `.artifacts/synchronized-admission-host-tests*.log`.
This fixes only the mock API mismatch introduced by synchronization; it does
not change production numerics or replace the outstanding GPU qualification.


## Follow-up: automatic qualified admission

After reviewing the completed standalone and composed evidence, the user removed
the opt-in requirement. Qualified device-fused FP64 WB97M-V and all-electron
exact-direct RKS-PBE0 now request local AO maps automatically. Unsupported
compositions remain dense when the environment variable is unset; an explicit
`GENERATIVEQC_CUDA_KS_ACTIVE_AO=1` still fails closed outside the qualified
predicate. `=0` is retained only as a debugging/benchmark opt-out. Resource
budget or device-allocation misses still fall back to dense execution.

This policy promotion does not broaden the scientific predicate to UKS, DF,
ECP, range-separated, nonlocal-correlation, host-unfused or lower-precision
paths. The synchronized branch head still has GPU_NOT_RUN status after this
policy-only follow-up, so the PR remains Draft pending ordinary CI/current-head
GPU requalification; no merge or release is authorized.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Generic-owner reconciliation

The [generic default reconciliation](2026-10-04-generic-scf-local-ao-default-reconciliation.md)
supersedes this note's method/provider-specific admission predicate. All
measurements, source/binary receipts and negative findings above remain frozen
at their original scope; the broader automatic capability policy does not
convert them into generic-family or current-head qualification.
