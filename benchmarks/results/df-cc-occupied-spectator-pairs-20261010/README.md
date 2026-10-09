# Bounded occupied-spectator folding of the DF CCSD ladder

This bundle retains scoped numerical qualification and complete-endpoint
observations, not shared formal/global performance promotion. The frozen parent
is master `2daa0aceaf6d140389ac5a183ead8b0ac3d3cd4a` (merged #2173).

## Endpoint and observations

Fresh-process native molecular energy-only DF CCSD(T): ethane230, 230 spherical
AOs, 9 occupied / 221 virtual orbitals, 488 auxiliary functions, all electrons
active. FP64 pedantic matrix execution, DIIS history 8, Q tile limit 8,
**ordinary full DIIS storage** (`packed_diis=false`), 64-GiB correlation budget.
Both selections retain the same independently expanded matrix physical replay.
RHF, native DF source, CCSD (including admission/projection), physical replay,
standard FP64 (T), startup and owner/caller teardown are inside process wall.
No production oracle, force, Lambda or orbital response is requested.

Finite Slurm job 2805 runs ABBA on node2's RTX PRO 6000 Blackwell Workstation
GPU, sm_120 Release/portable CUDA AOT, CUDA 12.9.1 and one thread per numerical
library. CUDA reports the assigned GPU UUID directly; NVML numeric indices are
not assumed to match Slurm's visibility. Exact UUID, runtime/driver API versions,
input, independent oracle, source overlay and binary hashes are in provenance.

| Median | Baseline | Candidate |
| --- | ---: | ---: |
| Complete process wall, s | 187.495296 | 165.657467 |
| Native helper, s | 187.142498 | 165.320404 |
| Complete CCSD, s | 91.362928 | 69.422902 |
| Primary iteration timer, s | 81.548458 | 59.540565 |
| Independent expanded replay, s | 9.675681 | 9.673095 |
| Logical packing bytes | 6,263,154,422,624 | 3,961,001,679,968 |
| Complete CCSD contraction summands | 44,063,663,429,680 | 27,891,350,238,256 |
| Complete CCSD numeric capacity, bytes | 4,381,900,502 | 4,717,423,790 |

Process wall decreases **11.65%** (1.13183x), CCSD **24.01%**, and its primary
timer **26.99%**. Baseline wall samples are 187.323076 / 187.667517 s; candidate
samples are 165.580412 / 165.734522 s. No sample is trimmed. Two observations
per selection are insufficient for shared formal promotion or broad hardware,
method, force or molecule claims.

The capacity increase is deliberate and included: the owner retains the full
original arena for per-state refusal, admits any larger folded-action scratch,
and adds separate projected tau. The `ccsd_pair_capacity_bytes` diagnostic is
the projected-tau/metadata payload (17,582,784 bytes), **not** the total extra
shared arena reservation; the complete capacity delta is 335,523,288 bytes.
Descriptor reservations are separately reported. Packing/projection counters
are logical work/bytes, not physical DRAM measurements.

All four solves use 20 iterations and 38 primary/trial evaluations. Each
candidate admits all 38 states, refusing none, without packed DIIS. All original
Q slices, tiles, accumulations, setup transfers and independent replay work
match. Each admitted primary Q saves exactly 872,104,896 complete auxiliary
summands; `488*38` such Q actions save 16,172,313,191,424 summands. The driver
checks the actual executed contraction delta rather than inferring it from
storage. New work includes 38 per-state projections, 2,005,039,296 logical
projection bytes and 49,610,080 immutable factor inspections per solve.

## Numerical and fallback gates

All four observations pass independent PySCF 2.14.0 energy/(T) gates of
`1e-8` / `1e-10` Eh and physical residual norms at most `1e-10`. Maximum total
energy/(T) errors are `2.203e-12` / `8.449e-14` Eh; singles/doubles replay norms
are `5.316e-13` / `2.621e-13`. Paired and original total energies agree exactly
at the preserved output precision. There is no convergence/iteration shortcut.

- 26 host checks cover exact signed-rational reflection proofs, shared-axis
  negative majorants, independent determinant/NumPy physical residuals,
  integer-exponent metadata, grouped norms, nonfinite/overflow refusal,
  subnormal tolerance margins and exact rational constants.
- Slurm 2802 passes six projection and 36 native solver checks; projection
  memcheck reports zero errors. Original tau retention, canaries, diagonal
  virtual-order identity, subnormals and nonfinite metadata are checked.
- Slurm 2804 passes nine final frozen-library checks: actual contraction
  deltas, initial/per-state refusal, pair-first budget priority, uneven Q tails,
  unchanged replay, and no optional allocation/projection for a single occupied
  block. Host constructor failure injection and five provider-lifetime tests
  also pass, including pair-first OOM retry and retained peaks.
- Seven focused CPU native-action/generation checks pass. The eleven original
  conventional/DF/core/hoisted generated artifacts remain byte-identical.
  Compiler structure: 509 modules, zero dependency errors.

Initial supplied T2 must be bitwise simultaneously pair-symmetric. Every
amplitude state rebuilds projected tau from independently retained original
tau. The original ladder DAG supplies a positive real-arithmetic perturbation
majorant and its proven gain to the physical doubles residual. Checked integer
power envelopes cannot underflow to an exact-zero marker; actual stored-mean
distances use qualified outward CUDA subtraction. Admission reserves at most
residual tolerance/8 for this projection contribution. This is **not** a full
floating-point solve certificate: final independent physical replay and the
external numerical gates still cover contraction/reconstruction roundoff.
CuMetal refuses this optional path until directed-DP qualification exists.

Pair resources are admitted after the original tile and replay. Optional OOM
removes pairs before replay, tile or provider fallbacks. Bound refusal selects
the complete original Q action for that state, without projecting supplied
amplitudes or weakening physical sticky-error checks.

## Reproduction and retention

`measured-source.patch.gz` reconstructs the measured worktree from the pinned
parent. `build.sh.gz`, `rebuild.sh.gz`, `qualify.sh.gz`, `measure-gpu.sh.gz`,
`measure.py.gz` and `visible-device.cpp.gz` retain the exact construction,
scheduled checks, ABBA driver and CUDA identity probe. Source archives, binaries,
all raw logs and Slurm-attempt artifacts remain in ignored local/remote scratch;
they are not published as Releases or external assets. The small accepted
bundle contains all four samples, provenance, summary and validation records.

Detailed rationale is in the occupied-spectator-pairs Agent Note. This iteration
reduces semantic work; it does not revive the rejected amplitude-layout cache,
adopt fully packed amplitude/DIIS coordinates, or change response equations.
