# Named KS correction resource qualification

Baseline: `c413cde9f` (master including #2835). The candidate production sources
are reconstructed by applying `source.patch` to that revision. `identities.json`
records the patch, source files, binaries and retained receipts. This is a scoped
#2227 R2/R5/R6 cutover, not completion of all non-R3 work or broader force admission.

## Endpoints and gates

The timing matrix uses PBE-D3(BJ) and PBE0-D3(0), restricted neutral and unrestricted
singly charged systems, energy and analytic energy/force requests, STO-3G, and
`GridSpec(12, 6, 12)`. Scalar H2 includes preparation, resource planning, execution,
publication and teardown. A ragged H2/water fleet includes preparation/cold,
two frozen replays, moved coordinates, two moved replays, and teardown. Separate
fleet protocol totals include all these costs. This small discrete-grid matrix
is not a basis/grid-limit, large-system or general-method performance claim.

Two source-isolated persistent Python processes load independent baseline and
candidate libraries. Both prime model/artifact caches before 8 ABBA/BAAB groups;
each measured protocol constructs fresh owners and cold densities. Each backend
has **1,792 complete scalar/batch calls**, plus separately timed complete fleet
protocol totals. Every measured row retains energies, forces, correction outputs,
SCF iterations, Fock/grid/occupation and J/K semantic work. The gate compares every
time series, requires matching work, E within 2e-9 Eh and forces within 2e-8 Eh/bohr.

Final CPU maximum candidate/baseline median ratio: **1.017847212813969**.
Final CUDA ratio: **1.0105351835299938**. Both satisfy the <=1.05 gate.
CPU observables match exactly. CUDA maximum E difference is 2.8422e-14 Eh;
maximum force difference is 9.7700e-15 Eh/bohr. These are mechanical parity
results, supplemented by independent numerical references below.

`cpu-samples.json.xz` and `cuda-samples.json.xz` retain every sample and binary
identity. Slurm job IDs and unchanged assigned CUDA visibility are inside the
receipts. GPU execution uses n1 RTX 5090; no GPU work ran on n3.

## Scientific, ABI and lifetime checks

- 126 CPU CTests passed after compilation. Focused final CPU tests: 170 passed,
  25 subtests passed, 20 CUDA-labelled cases deselected.
- Final source-matched CUDA-library suite: 98 passed under Slurm. It includes
  CPU/CUDA complete named programs, D3 damping/ATM variants, RKS/UKS, ragged
  replay, moved geometry, item failure isolation, recovery, and energy-only
  publication. Selected native failures cannot retry Python corrections.
- A complete budgeted water PBE-D3(BJ) endpoint is compared with independent
  PySCF SCF energies and central differences at the declared moving quadrature,
  plus the pinned simple-dftd3 correction fixture. Gates: 2e-8 Eh, 3e-6 Eh/bohr;
  correction E/gradient gates are 2e-14 / 3e-13 respectively. Both CPU and CUDA
  pass. This does not certify convergence to an infinite grid.
- The installed-header `sdk.cpp` executes named scalar and ragged E/F with
  `env -i PATH=/nonexistent`. CUDA keeps Slurm's assigned visibility. Caller
  resource handles are destroyed before replay; retained diagnostic handles
  prove allocation/provider charges reach zero after destruction. CPU rejects
  CUDA resource attachment explicitly.
- Native D3 device tests cover the exact actual allocation capacity, one byte
  below, partial-setup rollback, replay failures and destruction outside the
  submitting scope. The CMake-built probe passed compute-sanitizer with zero errors and zero
  leaked bytes (Slurm 14824).
- Global/local capacity, native query transactionality, old-library dispatch,
  current owner inventory and optional CPU compatibility storage are exercised
  by `test_named_ks_resources.py` and existing resource tests.

## GPU work and memory

`profile-summary.json` and the retained Nsight CSV summaries describe an untimed
complete matrix (56 calls), profiled separately from timing. Raw `.nsys-rep` and
SQLite traces remain in ignored `.artifacts/non-r3/` and the n1 qualification
workspace. Actual kernel, transfer and allocation counts are not inferred from
SCF iterations. The report distinguishes explicit CUDA allocations from driver,
module/static, stack, allocator-pool and other opaque memory.

Native per-item corrections retain duplicated tables and rebuild them with a
changed geometry. The final matched trace (Slurm 14826) measured 30,784 -> 30,832 kernels,
H2D 5,180,520 -> 9,843,976 bytes, unchanged D2H bytes 2,053,872 (more small copies),
and explicit live peak 26,757,490 -> 27,048,950 bytes. Both processes retire all
observed dynamic device allocations and issue no graph API calls. These costs are
included in the passing complete endpoint gate; this is not a lower-work claim.

## Reproduction

Use `qualify.py --baseline <checkout> --candidate <checkout>
--baseline-library <library> --candidate-library <library> --backend cpu|cuda
--groups 8 --output <samples.json.xz>`. Run the CUDA command through finite `srun`
on a permitted node. `endpoint_worker.py --once` is the complete profiler workload.
The profiler uses Nsight Systems 2025.1.3, CUDA tracing and memory-usage events,
with CPU sampling/context-switch tracing disabled.

Both builds use Release, GCC 11.4, scalar CPU linear algebra and verified
sccache 0.16.0. CUDA uses toolkit 12.9.86 and sm_120; AOT defaults are unchanged.
Compiler commands and shared-cache before/after snapshots remain in ignored
artifacts. The latter are shared-cache statistics, not exclusive task hit counts.

Early remote attempts lacked the checkout root marker/native compiler inputs
and failed during Python model provenance preflight, before numerical execution.
Their logs are retained separately; successful runs used complete source inputs.
Initial fixture/import and SDK type-name mistakes were corrected, with subsequent
full passing runs. No scientific tolerance, work gate or unsupported domain was
relaxed to obtain acceptance.
