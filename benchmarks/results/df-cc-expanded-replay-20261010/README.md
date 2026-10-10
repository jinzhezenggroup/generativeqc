# Independently packed expanded DF CCSD replay

This bundle preserves a scientifically qualified lowering and its conditional
complete-endpoint observations. It is not a universal speedup claim or a passing
shared formal performance-promotion envelope. The measured parent is the exact
`b315d4056` master revision in `provenance.json`; the reconstruction patch and
frozen executable/library hashes identify the candidate.

## Scope and settings

Fresh-process native molecular energy-only DF CCSD(T), ethane230, 230 spherical
AOs, 9 occupied / 221 virtual orbitals and 488 auxiliary functions. All electrons
are active. Settings are FP64, pedantic matrix execution, DIIS history 8,
Q tile limit 8 and a 64-GiB correlation budget. Each process includes RHF and its
DF preconvergence guess, correlation-source construction, CCSD, independently
expanded physical residual replay and standard (T). Forces/Lambda/orbital
response are not requested; their unrequested diagnostic fields are not results.

Finite n2 Slurm job 2798 uses one RTX PRO 6000 Blackwell Workstation GPU, CUDA
12.9.1, sm_120 Release, portable CUDA AOT profile and one thread per numerical
library. The order is baseline/candidate/candidate/baseline. Preserve Slurm's
device visibility; no GPU visibility override or production oracle is used.

The native helper includes inner phase-owner teardown. Complete process wall
also includes startup and final caller/reference cleanup. All external oracle
checks occur after timing. Existing iteration/replay timers add no extra fences.

## Complete observations

| Median | Baseline | Candidate |
| --- | ---: | ---: |
| Complete process wall, s | 195.912753 | 187.624613 |
| Native helper, s | 195.582392 | 187.311250 |
| Complete CCSD solver, s | 99.831593 | 91.389140 |
| Primary iteration timer, s | 81.555446 | 81.560398 |
| Expanded physical replay timer, s | 18.127537 | 9.678092 |
| Logical packing bytes | 5,578,493,904,992 | 6,263,154,422,624 |
| CCSD numeric capacity, bytes | 4,130,767,416 | 4,130,816,982 |

Complete process wall decreases 4.23% (1.04417x); CCSD decreases 8.46% and replay
46.61%. The complete-wall baseline spread is 0.453081 s, versus an 8.288141-s
median difference. Both candidate observations are retained; none are trimmed.
The copy/capacity tradeoff is included rather than hidden by replay-only timing.
Packing counters are logical reads/writes, not measured physical DRAM traffic.

The existing compiler packs the original expanded one-Q graph; it does not
consume primal reduced cuts. All original Q slices and core replay remain.
Primary iterations/evaluations, tile size, physical Q scope and accumulation
work match. Each replay Q adds 27 prepared GEMMs, 9,030,980,907 matrix summands,
29 operations and 1,402,992,864 logical packing bytes. Common-subexpression
elimination saves exactly 439,569 scalar summands per Q, or 214,509,672 for
the complete physical replay. The driver checks every resulting counter delta.

## Scientific and compatibility gates

All four observations pass the pinned independent PySCF 2.14.0 total-energy /
separate-(T) gates of `1e-8` / `1e-10` Eh and independent physical residual norms
at most `1e-10`. Maximum energy / (T) errors are `2.203e-12` / `8.448e-14` Eh;
maximum singles / doubles replay norms are `5.316e-13` / `2.621e-13`.
Slurm 2797 passes 33 native solver and 17 CUDA action regressions; Slurm 2798
passes six frozen-library ABI, capacity, ordered-sum and sticky-error checks.
Host graph tests pass 28 cases and native CPU solver tests pass 22 cases.

Optional replay admission follows primary tile selection. Exact and one-byte
short budgets, unchanged primary tiles, scalar replay ablation, determinant
oracles and early-Q overflow are covered. Allocation rejection drops replay
packing first; execution failures do not rerun partially computed work.

Master #2163 changes force/Lambda policy after the energy-only return. #2153
changes SCF but leaves this CC lowering unchanged. Their source is integrated
without repeating the full GPU suite merely because master advanced. Relevant
CC generation is byte-identical, and the endpoint/CUDA solver compile with
the latest headers. These frozen observations do not claim latest-master SCF
timings. Two samples per selection, absent timed per-iteration histories and
unmeasured isolated compile cost do not satisfy the shared formal promotion
envelope; `validation.json` marks it not run and accepts only numerical scope.

## Reproduction and retained artifacts

- `samples.json.gz`: all four complete records, exact argv, walls and gates.
- `summary.json` / `provenance.json`: medians, work deltas and content identities.
- `measured-source.patch.gz`: exact candidate delta against the frozen parent.
- `base-timers.patch.gz`: baseline-only exposure of existing phase timers.
- `measure.py.gz` / `build.sh.gz`: exact endpoint driver and ccache-enabled build.
- `qualify.sh.gz`: the finite-allocation frozen-library qualification wrapper.
- `qualified-native-tests.py.gz` / `replay-host-tests.py.gz`: exact qualification fixtures.
- `validation.json` / `publication.json`: scoped gates and checked inventory.

Decompress the build/driver files and relocate their explicit machine-local
`ROOT` / dependency paths if necessary. Decompress the qualification fixtures
as `test_df_cc_native_solver.py` and `test_df_cc_matrix_replay.py` in the same
experiment root expected by the build/wrapper scripts. Reconstruct the pinned parent, build
the baseline, then apply the retained candidate patch and build it in the same
source directory, freezing matching libraries, endpoint objects, generated
headers and `.so.0` SONAME links. Inputs and the independent oracle remain in
`benchmarks/results/rccsd-diis-ring-1900/ethane230.input` and
`benchmarks/results/df-lambda-cost-2136-20261009/oracle-energy.json`.

Run real-device tests and measurements through a finite allocation, for example
`srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:30:00 bash qualify.sh`
on n2. `qualify.sh` selects the frozen-library checks and then runs the retained
driver; its exact controls are preserved in each sample. Full trees, binaries,
raw logs and verified compiler-cache receipts remain ignored at
`n2:/data/jzzeng/qc-cc-replay-20261010/` and locally under
`.artifacts/df-cc-replay-20261010/`. No Release or external backup is created.
