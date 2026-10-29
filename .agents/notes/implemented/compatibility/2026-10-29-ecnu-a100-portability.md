# Decision: admit CUDA targets by capabilities and retain untuned provenance

Status: implemented
Date: 2026-10-29

## Problem

An ECNU phase-9 A100 deployment of master `dc526e408` exposed several assumptions
derived from the development SM120 device. Automatic shell-profile resolution
rejected an untuned SM80 target even though a portable implementation existed.
Experimental integral AD defaulted to compiling SM120. Cooperative/phased Becke
resource planning and native phased-force admission excluded other architectures
despite using ordinary CUDA arithmetic and synchronization.

The first optimized SM80 build also failed at native-direct device link with 101
`uses too much data for compiler-generated constants` diagnostics. A previous
SM89 campaign recorded the same toolchain workaround; this is not an A100-only
scientific limitation.

## Decision

- Automatic profile resolution prefers exact or explicitly compatible tuning,
  then uses the existing portable resolver. Preserve untuned metadata. Ambiguous
  profiles, incompatible explicit requests and stale generator ABIs still fail.
  Do not relabel SM120 schedules as measured on another target.
- Integral AD discovers the requested CUDA-visible ordinal's compute capability
  when no target is supplied. Explicit targets remain explicit, and CPU work
  does not query a GPU. Never rewrite scheduler device visibility.
- Admit cooperative and phased Becke work using molecular bounds, thread/shared
  memory limits, launch dimensions, retained centers, full point-lane coverage
  and byte budget. Keep small-system and constrained-resource fallbacks, optional
  allocation fallback, and propagation of actual driver/runtime failures.
- Disable ptxas optimizer constants only in the device-linked native-direct
  archive. The large shared recurrence graph otherwise exceeds the constant-bank
  budget. Independently compiled CUDA owners retain normal compiler policy.
- Let the existing complete-endpoint benchmark recognize finite LSF allocations
  and explicitly opt into measuring portable builds, retaining scheduler and
  untuned-build provenance.

## Rejected alternatives

Adding A100 to another model allowlist repeats the underlying portability issue.
Borrowing an SM120 tuned profile gives false provenance and unqualified resource
assumptions. Removing budgets or retrying CUDA driver failures as ordinary OOM
would weaken existing safety and error semantics. Disabling optimization globally
would change unrelated kernels and invalidate a useful performance comparison.

## Environment and reproduction

ECNU uses LSF queue `gpu_chem` on `gpu22`, A100 PCIe 40GB and driver 535.154.05.
Use an actual finite `bsub` allocation; do not fabricate Slurm environment fields.
The source build used CUDA 12.9.86, GCC 14.4, `80-real`, Release and verified ccache.
The numeric cubin avoids new-toolkit PTX JIT on the older driver. Stationary
companions were PBE RKS/UKS, PBE0 RKS and r2SCAN RKS; this is a focused source build,
not qualification of every method or a distribution wheel.

CentOS 7's system glibc 2.17 cannot load the required modern cuSOLVER. A private
conda sysroot supplied glibc 2.28, with interpreter/RPATH relocation restricted to
disposable environment executables. The system libraries were not modified.
Runtime NVCC also needs the selected host compiler through `NVCC_CCBIN`, since
CMake's host-compiler setting does not configure independent JIT invocations.
Home GPFS quota was exceeded, so deployment and caches used node-local `/tmp`;
that location and the VPN authentication are temporary, not a persistent install.

## Evidence and scope

Task-local raw evidence, exact build recipes, frozen basis/oracle inputs, patches
and environment receipts are retained in `.artifacts/ecnu-a100-20261029/` in the
qualification worktree. `base-revision.txt` identifies the full baseline revision.
The measured baseline has the same device-link repair as the candidate because
the original artifact could not link. This comparison isolates the scheduling
changes; it cannot establish the compiler flag's performance effect on other
architectures.

Public H2 RHF/STO-3G CUDA energy and forces passed an independent PySCF check.
PBE0/def2-SVP direct tests use 3/12/48/96-atom water clusters, full cold/warm/moved
energy-plus-force calls, three warm samples per geometry and two fresh processes
per arm in ABBA order. Independent GPU4PySCF receipts use identical orbital basis
and quadrature. Gates are 1e-8 Hartree in energy and 1e-7 Hartree/Bohr maximum force
error. Preserve SCF/Fock and available AO work; unavailable force counters remain
unknown. Matched-grid regression evidence does not certify grid-converged
production accuracy or an engine performance ranking.

All 128 direct-PBE0 ABBA records passed. Warm energy-plus-force medians (six
samples per arm and geometry, including the full host-visible endpoint) were:

| Atoms | Baseline seconds | Candidate seconds | Change |
| --- | ---: | ---: | ---: |
| 3 | 0.101511 | 0.101727 | +0.21% |
| 12 | 0.499322 | 0.498981 | -0.07% |
| 48 | 6.306732 | 5.242488 | -16.87% |
| 96 | 26.160282 | 18.835904 | -28.00% |

Moved-warm changes were +0.34%, -0.50%, -17.25% and -28.06%, respectively.
Warm Fock counts and retained AO work matched between arms. Cold medians changed
by -0.98%, +0.07%, -4.36% and -9.19%. One candidate 96-atom moved solve required
13 Fock builds versus 12 in the other processes; its timing is retained, and
that moved-solve comparison is not an isolated force-schedule claim.

HF and PBE0 density-fitted 3/12-atom endpoints also passed with the same frozen
orbital and auxiliary bases. An initial HF/DF moved-warm 3-ms difference did not
reproduce in four further ABBA rounds (16 fresh processes, 24 warm samples per
arm per geometry). These 9–14 ms samples have overlapping ranges; no stable
regression or HF/DF speedup is inferred. Across the main comparisons and this
recheck, all 320 independent-reference benchmark records passed; maximum errors
were 7.96e-12 Hartree and 9.28e-11 Hartree/Bohr. This excludes the separate H2
smoke and finite-difference tests from that record count.

Eight A100 tests passed for PBE RKS/UKS, PBE0 RKS and r2SCAN RKS at 48/96 atoms,
checking moving-grid force finite differences and constrained-budget fallback.
Two further A100 tests passed for automatically targeted RHF/UHF integral AD,
including independent PySCF energy/gradient checks, integral JVP/VJP, coupled
HVP, and retained-state lifetime. Host target/resource checks passed (208 passed,
one optional test skipped), as did seven actual SM80/SM120 CMake graph cases.
Ruff and whitespace checks passed. The production source inventory matched the
measured candidate binary before the required pre-commit formatter reflowed two
lines in the native admission condition. A separate final formatting receipt
records both identities and verifies identical whitespace-delimited tokens in
that sole production difference; no production logic changed after measurement.

Raw archive SHA256:
`d1acd972a197f040ec1e1f23d5c35e19b36b76f8577014529f98073814556a2c`.
The archive includes both source-build graphs, binary and companion hashes,
exact dependency listings, compiler-cache receipts, numerical samples and GPU
test logs. Later local changes relative to `candidate-measured.patch` add tests
and documentation plus the formatting described above. The original measured
source-identity receipt and raw archive retain their exact bytes.

## Revisit when

Add measured architecture-specific schedules only with independent numerical and
complete-endpoint evidence. Revisit the compiler flag when a supported toolchain
links the full recurrence graph without constant overflow and preserves endpoint
performance. Preserve capability checks and bounded fallbacks when changing the
phased implementation or expanding its molecular domain.
