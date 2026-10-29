# AOT reuse audit, 2026-10-27

This is source/build evidence, not a molecular performance result. The baseline
is `a16443baced95f5434c31b74cd993c970c39927b`. The candidate is explicitly recorded
as a dirty worktree, with compiler-input and changed build-input hashes. No GPU
execution, numerical admission expansion or kernel-count reduction is claimed.

The [decision note](../../../.agents/notes/implemented/architecture/2026-10-27-aot-source-and-execution-identity.md)
explains the mathematical and execution-ABI boundaries.

| Inventory | Finding |
| --- | --- |
| Curated Stationary, 64 CUDA units | 1 shared s/p primitive + 23 shared s/p/d shards + 28 legacy wrappers + 12 native wrappers |
| Staged Stationary geometry | 3 exact-source groups; including spin, weights, source inventory and reduction separates all 14 profile contracts |
| DF, 80 ordered shell classes | 2,100 axis-cache demands, 100 distinct power triples/expressions, 80 distinct ordered layouts |
| Orbital exchange | 50 unordered angular orbits; no implemented/qualified kernel alias |
| Source preservation | 64 scientific source records, 45,970,400 bytes, matching baseline SHA256 and size per record |
| Compiled target identity | Actual SM80 and SM120 CUDA objects and shared libraries have distinct keys and binary hashes; replay skips compilation |
| Combined image request | CMake preserves both `120-real` and `120-virtual`; cuobjdump lists SM120 cubin and PTX |

`census.json.gz` retains all profile contracts, per-class cache/schedule data and
polynomial expression hashes. `source-comparison.json.gz` retains the one-off
comparison; native rows cover scientific wrapper bodies, excluding outer binding
provenance. These historical digests are not permanent required-CI fingerprints.

`compile-validation.json` retains real artifact identities, toolkit versions,
changed input hashes and cuobjdump output. `validation-summary.json` retains the
actual sccache compile command, both generate-code arguments, parsed compiler-cache
counters and test outcomes, with original local-log hashes. These cache counters
describe the CMake probe, separately from the private JIT cache slots used by the
runtime compiler adapter. Original transient logs remain ignored locally. The probe
used sccache 0.16.0 and NVCC 12.9.86. Compiler-cache use is not a claim of hits.
`cache-dependency-validation.json` records evaluation of the actual backend
macro: all five scientific generation edges preserve a cache path containing
spaces and include their complete recipe input closure.

The source-cache/dependency/CMake regression command recorded under `test_suites.pytest.log` in `validation-summary.json` is:

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_stationary_aot_source_cache.py \
  tests/python/test_stationary_aot_build_dependencies.py \
  tests/python/test_stationary_semantic_cache.py \
  tests/python/test_stationary_aot_generation.py \
  tests/python/test_semantic_source_cache.py \
  tests/python/test_cmake_ownership.py
```

It passed 130 tests. The subsequently added automatic shared-package cache case
is included under `test_suites.aot-cache-tests.log` (five cases including the
existing four).
Two legacy mock checks were replaced by output checks sensitive to the selected
profile; their 18-case targeted rerun also passed.

Real compilation qualification (11 tests, including the host-only CMake cases):

```bash
GENERATIVEQC_NVCC=/group/software/cuda-12.9.1/bin/nvcc PYTHONPATH=python:. \
  python -m pytest -q tests/python/test_cuda_aot_architecture.py
```

Further native/DF regression coverage passed 951 tests, recorded in
`test_suites.native-df-tests.log` in the same summary:

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_native_stationary_cuda_binding.py \
  tests/python/test_native_stationary_shared_kernels.py \
  tests/python/test_native_stationary_graph_inventory.py \
  tests/python/test_native_stationary_build_options.py \
  tests/python/test_df_shell_units.py
```

The audit is reproducible without a GPU:

```bash
python tools/audit_aot_reuse.py --output /tmp/aot-reuse.json
python benchmarks/results/aot-reuse-audit-20261027/source-comparison.py \
  baseline /tmp/aot-baseline.json
python benchmarks/results/aot-reuse-audit-20261027/source-comparison.py \
  candidate /tmp/aot-candidate.json
cmp /tmp/aot-baseline.json /tmp/aot-candidate.json
```

The historical comparison refuses compiler inputs other than those audited.
Source reuse and mathematical commonality do not establish endpoint speedups;
additional device-execution sharing requires independent numerical and complete
endpoint qualification.

PR preparation integrated the subsequent #2835 packaging/host-validation fix
`c3ddb8861e65a6903c39098eb75cad9f90e7f6a8`. Its scientific compiler inputs are
unchanged. The existing packaged-AOT preflight digest was refreshed for the
reviewed cache wiring; 427 capacity/point-contract tests passed, recorded under
`test_suites.pr-base-compatibility.log` in the structured summary.
