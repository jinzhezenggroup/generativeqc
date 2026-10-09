# Decision: stage a physical Gram ring for ordinary CUDA HF and KS DIIS

Status: implemented as an explicit opt-in; default policy remains unchanged
Date: 2026-10-09

## Problem

Ordinary CUDA HF and KS rebuild every live residual-history dot for every DIIS
solve. With live history `h` and residual length `N`, each insertion repeats
`h*h*N` scalar products, although only one residual enters the ring. The
existing DF cooperative path has a separate schedule and is outside this change.

## Decision

Keep a physical `capacity*capacity` raw FP64 Gram cache per system, separate
from the existing `(capacity+1)^2` augmented solve scratch. Before inserting a
pending residual, one same-stream shared history kernel updates its row and
column against live old slots and computes its self norm. The existing SCF
kernel then inserts Fock and residual, assembles the small solve from physical
Gram slots, and retains its chronology, normalization, retirement and
Fock-combination policy. The new-row contraction and strict FP64 scalar steps
come from the checked SCF TensorIR template; the shared tensor CUDA primitive
owns physical slot addressing and its 256-lane reduction tree.

`GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM=1` selects this route when a plan is
prepared. The ordinary serial-dot instantiation remains the default; DF and CC
callers are unchanged. The tree has a different addition order from ordinary
serial HF/KS dots, so the measured opt-in result does not justify a default
policy change.

## Invariants and work

- Counts and heads live on device. A pending row reads only the `count` live
  physical slots and itself, before the same stream overwrites the pending
  slot. Reset needs only count/head reset; stale Gram entries are unreachable.
- At old live count `k` and capacity `h`, a valid insertion computes
  `min(k+1,h)` full-vector dots, or `min(k+1,h)*N` scalar products, and writes
  `2*min(k,h-1)+1` raw Gram scalars. At capacity, the overwritten slot is no
  longer a live old vector. A full history of eight costs `8*N` products per
  insertion instead of the old
  `64*N` per solve, with further old-work repetition on dependent retries.
  The bounded `capacity`-block grid still launches inactive blocks.
- Raw Gram and augmented solve scratch may not alias. Resource estimates and
  arena partitioning charge the optional `capacity^2` FP64 storage.
- Invalid device count/head skips the pending reduction; the incremental
  update clears that system's history and returns its current Fock. This is a
  recovery path, not acceptance of arbitrary nonfinite residuals.

## Rejected alternatives

Reusing the augmented solve buffer as raw storage would let Gaussian
elimination destroy cached old-old entries. Keeping the legacy full rebuild
avoids the reduction-order change but retains quadratic history-pair work.
An unconditional library dot route would require separate numerical and
complete-endpoint evidence, and would cross the provider-neutral tensor
ownership boundary.

## Evidence

The source-matched H200 build used base commit
`d5a3173cc89b399ed105b0750124473eb782e2e5`, archive SHA-256
`c9303e0617b5375fdcb481323e892feb37f681bb398a98c85fc8090ab5ee4809`,
and worktree patch SHA-256
`2206301f1c0a421873def3f4090d26398f5b6148ccbe5a527f4e3014484fad2a`.
It used CUDA 12.9.86 on H200 `sm_90`, Python 3.11.16, CMake 3.31.10 and
`sccache` 0.16.0. Generated C++ and CUDA compile commands invoked the verified
launcher. The native library SHA-256 was
`bb164cc9d434a8c07fb85cefa0be3385ae3035edc0bba40473451dba496092a1`.

The initial `generativeqc_cuda_diis_tests` H200 Notebook run set
`SLURM_JOB_ID=notebook-manual-issue1874` to bypass the test's Slurm-only
allocation guard. It was a real-device run, but not a Slurm allocation or a
valid guard pass. The test now also admits a Notebook only when its Jupyter and
Kubernetes context is present, `MY_POD_NAME` matches the system hostname, and
the mounted Kubernetes namespace is readable. The platform reported the
`issue-1874-gram-h200-20261009` Notebook RUNNING with 1x H200 in project
`原子级化学反应基座模型2.0`. With `SLURM_JOB_ID` absent, the rebuilt target passed
CTest 1/1 and printed pod
`issue-1874-gram-h200-20261009--235b7fdaf387-flfwrdz4db`, namespace
`chemicalreaction`, and `NVIDIA H200`. Removing both allocation identifiers
made the binary return the configured skip code 77. The test source SHA-256 was
`9c8f2b7b726c56d445f76e0c22bc03b40d45b76323b128989bb3c242ee124a17`,
the test executable SHA-256 was
`9bad77651907860a3b76a5a500555c71cd1266350bbb9b52205c6a124270792f`,
and the library hash above was unchanged. The target covers ring wrap, reset,
inactive/ragged systems, invalid count/head, nonfinite pending values and
normalized dependent-vector retirement. The relevant Python
TensorIR/codegen tests passed 30/30. With the opt-in enabled, independent HF
reference tests passed 5/5, KS energy/commutator reference tests passed 8/8,
LDA RKS/UKS public forces matched independent gradients 2/2, and two DF
shared-caller regressions passed 2/2. Final HF/KS density matrix A/B errors
were at most `9.1e-14` across RHF, UHF, RKS and UKS.

Matched complete endpoint A/B used three calls per mode and the median of
the last two, including the requested output and public `Calculator` work.
The ratios below are incremental/default; values near one do not establish
a material speedup.

| Endpoint | Default ms | Incremental ms | Ratio | Iterations default/incremental |
| --- | ---: | ---: | ---: | ---: |
| RHF water def2-svp, energy+force | 142.7 | 142.1 | 0.996 | 17/17 |
| UHF OH def2-svp, energy+force | 157.7 | 156.5 | 0.993 | 19/19 |
| LDA RKS water sto-3g, energy+force | 1138.6 | 1134.6 | 0.996 | 9/9 |
| LDA UKS water cation sto-3g, energy+force | 1179.9 | 1185.8 | 1.005 | 12/12 |
| LDA RKS water def2-svp, energy | 227.5 | 227.7 | 1.001 | 12/12 |
| LDA UKS OH def2-svp, energy | 638.9 | 604.5 | 0.946 | 57/56 |

The last row changed iteration count, so its timing difference is not a
per-iteration kernel speedup measurement. Baseline/candidate energy errors
were at most `4.3e-14` Hartree and force errors at most `1.3e-14`
Hartree/Bohr in this matrix. The raw endpoint JSON SHA-256 is
`29149f61cf899f58535fc04b9ed0b6eb0e8fd9e5d652ef9d5efce75e197124b2`
under `/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1874-gram-20261009/endpoint-ab.json`.

## Consequences and revisit when

This opt-in changes reduction order and adds one stream launch plus bounded
`capacity^2` storage. It removes old-old vector dot work, but the measured
complete endpoints show no consistent benefit at these small and medium
domains. The default should change only after broader numerical and endpoint
qualification establishes a useful crossover. Parent #1874 still owns
Fock-history provider/crossover work and runtime diagnostics for live history,
executed reductions and vector work; this Gram slice does not close it.

## References

Issue #1874; `src/tensor/cuda_history.cuh`;
`src/scf/cuda/scf_diis_kernels.cu`;
`tests/native/test_cuda_diis.cpp`.
