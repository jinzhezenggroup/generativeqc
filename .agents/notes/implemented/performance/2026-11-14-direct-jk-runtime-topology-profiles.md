# Direct J/K runtime topology profiles

Status: implemented
Date: 2026-11-14
Issue: #597

## Problem

The master baseline `5e34e2769915ead77ffd1a93f9b9bf82c1c7aba8` already had
exact-stack FP64 matrix-profile admission and compiler-owned persistent queue
domains. However, the accepted profile could not select the resident-ERI AO
crossover or the force/Fock queue domains in the production prepared owner.
Moving a constant into generated output was not runtime profile activation.

## Decision

- Extend the existing canonical `HfMatrixProfile` to v2 with an explicit
  `DirectJkTopologyProfile`, rather than introduce another device-name selector
  or infer integral profitability from matrix-product rates. The record contains
  a residency crossover, resident byte ceiling and force/Fock queue domains.
- Require the same exact source/provider/library/build-mode/target identity and
  qualified AO/batch domain as matrix admission. Zero disables residency; the
  crossover cannot exceed the qualified maximum AO dimension. Queue domains
  stay between three and the compiler-emitted maximum. Pre-release v1 records
  fail closed and must be regenerated, rather than acquire invented defaults.
- Resolve the effective record in one shared workload-admission method. Store
  its entire identity and resolved topology in `HostBatch`; copy runtime domains
  into `DeviceBatch` before capture. Allocation and reset retain the bounded
  compiler-emitted queue superset. Reduced runtime domains select existing
  fixed-grid workers without altering low-order fused owners or equations.
- Retire a refused resident arena before repacking Cartesian transforms and
  trying the bounded direct owner. Each optional matrix/residency feature can
  be disabled only once. Compatible replay retains the refusal, while a new
  effective profile can request a new selection. An admitted but out-of-domain
  file is not a new effective profile. Reference-provider lifetime authority
  and device-inert resource-query contracts remain unchanged.

## Rejected alternatives

Neither an n^4 byte estimate nor calibrated GEMM throughput describes the actual
screened-quartet alternative, expected Fock reuse or geometry rebuild frequency.
Do not manufacture an automatic resident/direct cost-model winner from those
inputs. Explicit topology choices require complete external endpoint tuning
against that alternative, including setup and storage. This change ships no new
measured profile and preserves the earlier no-promotion conclusion.

Do not shrink queue allocations to a runtime count while their captured resets
still use compiled bounds. Keeping the bounded superset avoids a second optional
FP32 allocation/reset ABI and does not grow default device storage.

## Invariants

No-profile crossovers remain ERI=16 and matrix=17, with force/Fock domains 7/6.
Target-memory ceilings, worker legality, the unknown-target fallback and the
explicit 1024-task staging bound remain intact. Scientific screening, precision,
Hamiltonian, force equations and the energy-only resident guard are unchanged.
Production never falls back through a CPU/PySCF oracle.

## Evidence

The retained qualification bundle is
`benchmarks/results/issue597-direct-jk-policy-20261114/`. It contains the complete
population report, independent oracle generator/results, actual iteration/Fock
work, selected routes, numeric high-water marks, sampled process memory,
  source/library identities and finite Slurm allocation receipts. The report's
acceptance gates are energy 1e-8 Hartree, force 1e-7 Hartree/Bohr and at most 5%
median endpoint regression in each declared group; numeric peaks may not grow.

All 48 final groups pass with six observations per binary/group. The maximum
median regression is 1.416342%, energy error 4.618528e-14 Hartree and force error
1.052275e-10 Hartree/Bohr. Default routes and iteration/Fock work agree throughout.
Both actual queue-reset and allocator-fallback memcheck runs report zero errors.
NVML process peaks are 8026/8038 MiB: retain that 12-MiB non-ledger overhead rather
than presenting the unchanged numeric peaks as an unchanged total-VRAM result.

- 708 focused host tests pass, including real packing/arena/profile consumers,
  synthetic targets, RHF/UHF batch counts, invalid identities, memory boundaries,
  overflow, topology invalidation and resource-only admission. Compiler and SCF
  ownership audits, the tuning-inventory checker, Ruff and clang-format pass.
- RTX 5090 job 2040 on authorized SSH host `n5` / Slurm `node5` uses `main`,
  `gpu:5090:1`, an exclusive finite 20-minute allocation and Slurm's unchanged
  `CUDA_VISIBLE_DEVICES=0`. The full production-profile test independently
  qualifies spherical 2/18-AO RHF/UHF energy/force, single/batched, cold/warm/
  moved endpoints, native/cuBLAS choices and reduced runtime domains.
- The energy-only 32-AO s/p/d/f allocation fixture has a 606640-byte direct
  numeric peak versus 8651336 resident bytes. A 4628988-byte ledger limit
  rejects the optional tensor once, completes the direct CUDA endpoint against
  PySCF, reuses its retained fallback and releases all tracked storage.

A capacity fixture must explicitly request energy only: default `ScfOptions`
requests forces, which correctly disables residency and otherwise makes the two
measured routes identical. The test asserts its actual selected topology before
interpreting capacity evidence.

Both timing binaries use Release, CUDA 12.9.86, sm_120 and the same explicit
fast-compile mode. Timers surround the complete cached CUDA bucket call,
including packing, target admission, allocation and finalization; fixture
construction and process launch are excluded. Each cold owner is fresh; warm
and moved calls consume its preceding physical density. Two fresh-process pairs
alternate baseline/candidate order, with three owner repetitions each. These
observations qualify the declared workload/build population, not every molecule,
another GPU, the non-fast-compile build or a new autotune winner.

Compilation initially encounters a full system disk. The final builds use
verified ccache 4.5.1 and the existing `/data/jzzeng/ccache`, with both CXX/CUDA
launchers explicit and temporary/build files on the data disk. Before/after
statistics are retained; they are shared-cache observations, not exclusive
task-attributed hit rates. No cache is cleared or correctness setting weakened.

## References

- [Earlier topology admission](2026-11-08-direct-jk-topology-profile-admission.md).
- [Current policy contract](../../../../docs/maintainer/performance_engineering.md#direct-jk-policy-ownership).
- [Qualification bundle](../../../../benchmarks/results/issue597-direct-jk-policy-20261114/README.md).
