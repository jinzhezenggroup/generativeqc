# Decision: keep UCCSD host evaluators outside the CUDA translation unit

Status: implemented
Date: 2026-11-16

## Problem

The generated UCCSD CUDA owner includes the CPU residual and response headers
for shared shape, capacity, feed/output metadata and semantic work contracts.
Those headers also contain large ordinary C++ evaluators. NVCC parsed these
unused host arithmetic bodies for every architecture even though CUDA dispatch
uses its own generated kernels. Kernel pooling alone does not remove this work.

## Decision and invariants

Guard only the CPU primal, triples and response evaluators, including their
response dispatcher, with `#ifndef __CUDACC__`. Keep the shared types, metadata,
required-capacity and work functions, equation identities and error behavior
visible to both backends. Ordinary C++ retains every existing evaluator.

Do not combine different numerical programs, change output inventories or CMake
ownership, remove methods or reduce wheel architectures. This boundary avoids
unnecessary parsing rather than altering device execution or scientific work.
The CUDA source itself is unchanged. A physical-GPU timing claim would not
follow from reduced host preprocessing.

## Evidence

Both complete UCCSD CUDA objects compile using NVCC 12.9.86, verified sccache
0.16.0, `sm_80` and the production compilation options. Actual NVCC preprocessed
input shrinks from 9,862,040 to 7,173,110 bytes: 2,688,930 bytes, about 27.3%.
The generated CUDA source remains byte-identical at 2,992,096 bytes.

Extracted PTX and CUBIN from both complete objects are byte-identical:

- PTX SHA-256: `63cd1bf397b63fafb5ecb8f9d8ee1b4623226eed11959cbf4ab575e49e490a36`.
- CUBIN SHA-256: `45941dc3cfccf8f5d3e380bfee5df23e266c0b12cbf69b96b18d82091a99942d`.

The object sizes are 10,902,568 and 10,899,400 bytes. Observed compilation times
are 165 and 135 seconds, but the baseline has four CUDA-stage misses while the
new build has three misses and one CUBIN hit. Other work overlaps these builds.
These timings therefore do **not** establish a cold-build speedup. The one-off
device-image comparison is audit evidence, not a permanent source-hash test.

The executable regression compiles actual generated headers in ordinary C++ and
host-only CUDA-preprocessor modes. It checks the backend API boundary, matching
metadata/work/capacity behavior, unknown-program and overflow failures, and a
CPU residual energy against the independent TensorIR interpreter. The latter
mode does not execute CUDA or qualify a physical GPU.

The boundary, Libxc catalog, RCCSD(T) dependency and CC scalar-reduction suites
pass 94 tests in 73.62 seconds. Compiler dependency checking reports 604 modules
and zero errors. Ruff checks and formatting pass.

## Associated CI audit refresh

PR run 43744336993 fails its core-a preflight because the retained Libxc native
catalog snapshot no longer lists all changed code-generation inputs. Refresh
the snapshot with the official `check_libxc_native_catalog.py --write` command;
do not weaken its acceptance projection. Its capability summary, qualified and
observed endpoints, evidence receipts, baseline, reasons and product scopes
remain unchanged. Only source identities and the changed-input lists of the
already-stale revalidation plan change. No previous qualification is promoted
to a current result. The subsequent native-CLI prerequisite failure is a
consequence of the failed preflight, not independent compiler evidence.

## Rejected alternatives and revisit conditions

Splitting the public header inventory or merging all generated sources is
unnecessary for this boundary and risks larger translation-unit working sets.
Deleting shared metadata would break backend contracts. Disabling catalog
checks would hide stale qualification rather than fix the CI dependency audit.

Full three-architecture wheel acceptance and the earlier Direct arithmetic
change's physical-device scientific/runtime qualification remain separate gates.
Revisit if CUDA intentionally acquires a host-replay consumer or profiling shows
another shared header contains large backend-exclusive implementations.
