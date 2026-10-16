# Decision: atomically poll concurrent native CUDA GMRES failure status

Status: implemented
Date: 2026-10-16

## Problem

The native GMRES vector kernels can read `State::code` while another lane,
warp or block reports an arithmetic failure through `atomicCAS`. A plain entry
read is a CUDA C++ data race even when later stream completion and true-residual
checks prevent publication of the failed result. Successful-solve memcheck or
shared-memory racecheck results do not establish global flag synchronization.

## Decision and invariants

Use a zero-value device-scope compare-and-swap to poll the status at entry to
normalize, precondition, orthogonalize, candidate and residual. It has the same
atomic scope as the existing failure writer. A helper records this concurrency
contract without changing floating-point arithmetic or reduction order.

The flag does not publish another data payload: other state is consumed through
the existing same-stream kernel ordering and block barriers. One-block norm/dot
collectives already bracket their relevant status accesses with barriers, and
single-thread scalar kernels remain ordered by the stream. Do not blanket-add
atomics to numerical reductions or reinterpret this polling operation as a
general acquire/release publication protocol.

## Executed evidence

The source boundary is PR #2527 head
`1cd0987fc232850b5fef7822640c3ee6519703c2` plus the local repair. The generated
header SHA-256 is
`92bcd108b69cdd3ad95b0b71c288d061621a372d66394c178ac533bd3a4b6879`.

- A host concurrency control compiles the actual old/new emitted kernels with
  CUDA lane indices represented by thread-local state and `atomicCAS` emulated
  by `std::atomic_ref`. Sixty-four threads exercise arithmetic failures in all
  five vector kernels. The old body produces a ThreadSanitizer data-race report
  and exits 66; the repaired body runs all five cases with exit 0 and no report.
  The fixed binary's first launch instead failed in ThreadSanitizer startup
  with an unexpected-memory-mapping fatal error. That log remains separate; a
  subsequent compatible launch produced the clean five-case result. This is
  host memory-model evidence, not CUDA execution or a GPU racecheck claim.
- Finite Slurm job 8819 on n1/node1, RTX 5090, preserves assigned
  `CUDA_VISIBLE_DEVICES=2`. NVCC 12.9.86 uses strict `--fmad=false`, sm_120 and
  verified sccache 0.16.0. The standalone native owner/probe passes all 32
  existing/expanded Python tests, including independent NumPy solve/residual
  oracles and unchanged failure/publication checks.
- Three additional memcheck executions cover preconditioner overflow at 3,
  257 and 1025 elements: all pass and each reports zero memory errors. The
  latter sizes cross block boundaries and include a partial final block.
- The remote generated header, owner/header, probe source and Python tests are
  hash-checked against the local source after execution. Local generation and
  compiler-structure checks pass (553 modules, zero dependency errors), as do
  Ruff/format and patch whitespace checks.

Raw host controls, startup failure, GPU logs/JUnit, source manifests, build
scripts and cache snapshots remain under ignored
`.artifacts/gmres-status-20261016/`; the standalone GPU build remains on n1 at
`/data/jzzeng/qc-gmres-status-20261016.gCr9T5`. Cache counters are shared daemon
observations, not exclusive compilation counts. No cache was cleared or stopped.

## Scope

This repairs one concurrency defect in the generic solver. It does not complete
or qualify native molecular Hessian/HVP assembly, coupled response integration,
the Python-free CLI runtime, scientific endpoints or complete endpoint speed.
Those independent PR acceptance requirements remain unchanged.
