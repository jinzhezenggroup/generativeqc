# Decision: prune unused Mulliken batch implementations and duplicate MINAO data

Status: implemented
Date: 2026-11-08

## Problem

The `d52d2ae2a` master snapshot still compiled two uncalled GFN2 CPU Mulliken
entry points: `evaluate_mulliken_population_cpu` and
`add_mulliken_hamiltonian_cpu`. Their 601 implementation lines duplicated the
population and Hamiltonian science already consumed through the per-system
interfaces. The complete production SCC driver calls only
`evaluate_mulliken_population_system_cpu` and
`add_mulliken_hamiltonian_system_cpu`, including restricted and spin-packed work.

`src/scf/initial_guess/minao_data.hpp` also retained a 46,425-byte copy of the
primitive, shell and element-offset data owned by
`src/integrals/minao_basis_data.hpp`. No native source included the old table;
only the initial-guess benchmark's source-provenance inventory referenced it.

## Decision

Delete the uncalled batch implementations and their private declarations,
without adding compatibility wrappers. Keep both per-system implementations,
their validation, scratch ownership, failure-atomic publication and selected
kernel tables unchanged. Keep the charge/magnetization contract documented on
the retained interface and describe workspace publication at per-system scope.

Delete the duplicate MINAO header and bind future benchmark evidence to the
actual integral-owned MINAO source, interface and data. Do not rewrite historical
benchmark receipts or remove their independent scientific reference inputs.

## Evidence and boundaries

- A repository-wide reference audit found only the definitions and declarations
  of the removed Mulliken entry points, and only the benchmark inventory for the
  removed MINAO path.
- A one-off comparison verified that the two MINAO primitive/shell/element
  payloads were identical; only include guards and namespaces differed. This is
  migration evidence, not a permanent production-source fingerprint test.
- Qualification uses the existing independent tblite energy/force fixtures,
  retained-runtime identity/error-recovery tests, and MINAO ensemble-admission
  oracle tests. No numerical, safety or lifecycle test is removed.
- This is CPU-source pruning, not a CUDA kernel, algorithm, precision, AOT
  inventory or public C ABI change. No GPU performance claim follows from it.

Qualification on GCC 11.4.0, Release, CPU-only, with stationary CPU force AOT
disabled and the verified sccache 0.16.0 launcher produced these same-build
artifact comparisons:

| Artifact | Before (bytes) | After (bytes) |
| --- | ---: | ---: |
| `mulliken.cpp.o` | 129,280 | 96,160 |
| Shared library, unstripped | 23,010,872 | 22,989,528 |
| Shared library, stripped | 22,104,016 | 22,083,536 |

All 362 exported `generativeqc_` C symbols were unchanged. A one-off comparison
of seven tblite-fixture molecules plus triplet water, doublet OH and water
dication found exactly equal energies, forces and iteration counts before and
after pruning. The unchanged production paths retain their independent-oracle
and lifecycle tests; artifact sizes are not a complete-build timing claim.

## Consequences

The checkout and compiler inputs shrink without changing production numerical
work. Object/library sizes and compiler-cache evidence must be distinguished from
complete clean-build timing; an incremental rebuild is not a clean-build speedup.
Deleting tracked files does not reclaim their historical Git objects. Benchmark
evidence, independent references and vendored license notices are not disposable
repository-size savings.

## Revisit when

Add a whole-batch internal interface only if a real consumer needs it, with an
explicit publication contract and executable numerical/failure coverage. Prefer
sharing the current scientific kernels over restoring duplicate scalar loops.

## Follow-up

The broader audit and cumulative measurements are recorded in
[the native pruning audit](2026-11-08-native-pruning-audit.md). The table above
records the initial Mulliken/MINAO change, not the cumulative follow-up result.
