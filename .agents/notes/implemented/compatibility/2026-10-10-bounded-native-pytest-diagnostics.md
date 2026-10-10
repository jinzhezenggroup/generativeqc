# Decision: Bounded native fault evidence for core-b

Status: implemented
Date: 2026-10-10

## Problem

Four hosted core-b workers faulted in PySCF's first H2 LDA MINAO `NPdgemm`
call, including two unrelated CI-only PRs. The unchanged #2194 retry passed.
Existing Python tracebacks establish the call boundary, but neither the faulting
native instruction/provider nor the exact crashed worker's preceding tests.
JUnit aggregates do not reconstruct xdist's work-stealing history. A fifth
older documentation head (#2187) faulted in the same H2 reference through
`libcvhf`'s incore Coulomb path, rather than MINAO GEMM; the library/provider
and preceding-corruption alternatives therefore remain open.

## Decision

Explicitly opt core-b into a pytest-only diagnostic plugin, compiled separately
from the scientific library. Keep its root-level `.artifacts/native-diagnostics`
files under the existing failure-only upload and 14-day retention. No reruns,
quarantines, tolerance changes, test-selection changes, or scientific fix are
part of this change. The master wheel trigger remains intact.

The plugin flushes test IDs, outcomes, and allowlisted package versions to a
4 MiB maximum history per worker process. It retains only executable mapping
metadata for allowlisted numerical/runtime library basenames, at most 128 entries
per snapshot. Mapping files are replaced, not appended. Native snapshot storage
is bounded at 256 immutable tables; reaching the cap retains the last table and
records the limitation. Newly loaded unmapped modules remain unresolved.

PySCF loads inside the failing test. The passive CPython `ctypes.dlsym` event for
`NPdgemm`, `CVHFnrs4_incore_drv`, and `CVHFnrs8_incore_drv` refreshes
mappings after the private BLAS/VHF load and before the corresponding call. It replaces no function or ctypes signature. A reentrant lock and
`PyDLL` serialize helper publication without releasing the GIL. Immutable tables
are published through a lock-free atomic pointer.

The SIGSEGV/SIGABRT handler writes at most one 512-byte record containing the
signal/code and mapped library/file offset. Absolute instruction/fault
addresses and mapping ranges remain in memory and are not written to disk. It uses bounded formatting, lock-free atomics, and async-signal-safe
signal/write operations. It does not unwind, call the loader/stdio/allocator,
read arbitrary memory, discover files, or collect environment values. The
saved Python faulthandler receives the original context. Repeated active installs
are rejected, and teardown preserves a handler replaced by the test. Returning handlers and
same/cross-signal recursion cannot convert the first fatal signal into success.
A malicious or independently hanging prior handler is outside this guarantee.

Raw cores are disabled by changing only the current opted-in worker's soft core
limit. Startup rollback and normal teardown restore its prior value. Controllers
and other shards are not intercepted. Runtime diagnostic I/O failures are
best-effort and cannot replace a scientific test outcome; configuration/build
failures remain explicit preflight failures.

## Rejected alternatives

- More blind retries: an unchanged pass establishes intermittence, not a repair
- Blaming or pinning PySCF/BLAS without a faulting native module or reproducer
- Unsafe in-handler `backtrace`, `dladdr`, or formatted stdio
- Raw core dumps, whole environment dumps, and unbounded worker logs
- Adding the contract file under `tests/python`: it would shift the alternating
  core-a/core-b file partition. The new contracts run separately before core-b

## Evidence and limitations

Exact Python/reference package versions, active coverage, a source-qualified
incremental native reconstruction, all 697 core-b module imports, and the 444
previously omitted Git-HEAD audit cases did not reproduce the crash locally.
Full-collection DFT passed 22 CPU cases; adding the audits passed 466 cases with
20 CUDA skips. This is not an exact hosted worker replay or clean hosted build.

Subprocess contracts enforce original SIGSEGV/SIGABRT exit signals, Python
traceback retention, returning/recursive handler chaining, late-provider mapping,
unresolved modules, bounded output, worker-attributed crash/assertion evidence,
external termination, concurrent recording, and failure-safe cleanup. The new
instrumentation must obtain hosted evidence before a root-cause repair is chosen.

## Revisit when

Use the next captured module-relative fault PC and crashed worker history to
build a bounded reproducer. Remove or narrow instrumentation once the verified
cause is repaired and its regression gate exists.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38016849726
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38016687381
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38016660738
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38019818279
