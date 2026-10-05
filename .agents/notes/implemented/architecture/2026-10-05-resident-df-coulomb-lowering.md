# Decision: canonical resident DF Coulomb execution for DFT

Status: implemented
Date: 2026-10-05

## Problem

The real DF-KS Fock owner still submitted the resident charge and J contractions
through direct cuBLAS calls. This bypassed #1886's canonical semantic operation,
admitted precision, candidate legality and prepared execution boundary.

## Decision

Express dense `bijq,bij->bq` / `bijq,bq->bij` and packed `bpq,bp->bq` /
`bpq,bq->bp` in the existing TensorIR. The generated adapter projects the same
`LoweringRequest` and two executable FP64 candidates into the shared native
selector. The generic `CudaVectorContraction` provider owns library submission.
The DF scientific owner selects its actual physical storage only.

Retain the incumbent algorithms: two strided batched GEMMs for dense tensors,
and two GEMVs per serialized system for packed tensors. Unknown complete costs
are negative selection evidence, not zero-cost estimates. GEMV rejects actual
batches greater than one. No precision or performance promotion is implied.

The binding borrows the existing exclusively owned handle/stream. It introduces
no numerical buffer or second library handle. DF plan host diagnostics charge
both heap binding objects, in addition to the existing plan object. Preparation
retains actual cuBLAS/runtime versions, compute capability, resolved shapes,
canonical identities, selection/rejection reasons and elapsed preparation time.
Library-internal borrowed allocations remain explicitly unknown. Existing
handle lifetime and mode invariants apply until both bindings are destroyed.

## Invariants

- Packed off-diagonal density is `D_ij+D_ji`, including nonsymmetric inputs;
  diagonal entries are counted once. Existing pack/scatter kernels own this.
- Preserve matrix orientation, auxiliary order, FP64 arithmetic, batch strides,
  and the existing fitted metric/threshold and exchange calculations.
- Preparation rejects capture before performing handle/provider queries.
  Replay submits only existing device work; the graph owner counts replays.
- Output storage is disjoint from both inputs; partial aliasing fails before
  submission. Immutable shape/device compatibility is checked by the binding.
- Bindings die before their borrowed handle. Generated metadata has static life.
- Streamed raw/metric Coulomb panels remain unchanged migration debt for #1890.

## Rejected alternatives

Routing dense batch-one calls through GEMV would change the incumbent recipe
without endpoint evidence. Allocating independent handles/workspace would add
resources and potentially affect capture. Reusing a generic owning matrix
executor would likewise change the original vector submission/layout recipe.

## Evidence

The dedicated host gate compares dense/packed TensorIR with an independent
four-center construction and uses nonsymmetric densities. The native gate checks
both executable candidates, multiple batches/tails, long-double scalar oracles,
changed-input graph replay, alias rejection, handle mode admission and capture
preparation rejection. Complete existing DF and KS suites qualify the actual
production consumers.

Qualification on n1, Slurm `main`, `--gres=gpu:5090:1`, CUDA 12.9 / cuBLAS
120901, source `dd6a2eac1` (includes the separate source-capacity fix #1956):

- 48 host algebra/selection tests, focused type check and all ownership/prek checks.
- Ten native vector cases: dense/packed, both executable candidates, nbf 1/7/24,
  naux 1/13/73, batches 1/2/3, three changed-input replays each. Scalar
  long-double gates are `2e-12 * (1 + abs(reference))` for charge and J.
- Complete native DF suite, including 48 weighted finite differences, source and
  resident/streamed J/K oracles, and complete native KS suite pass.
- Eight CUDA public DF-KS energy/geometry/ragged/resource gates pass (one CPU-only
  selection skipped), plus 31 DF resource tests.
- Six complete FP64 DF-KS timing cases: dense/packed-single PBE-RKS and PBE0-RKS
  water (STO-3G, 7 AOs), PBE-UKS Li (STO-3G, 5 AOs); explicit def2-SVP auxiliary
  basis and Cartesian representation. All actual plan diagnostics are resident.
  Each case includes cold, two warm, and changed-geometry execution; maximum
  independent PySCF energy error is `4.13e-13` Hartree against a `1e-8` gate.
- Native binding memcheck/initcheck report zero errors. Both prepared objects
  occupy 3,728 retained host bytes in this build; explicit numeric workspace is
  zero, while borrowed cuBLAS internal allocations remain unknown.

Representative complete current-path times (not a before/after comparison):

| Endpoint / storage | Prepare (s) | Cold (s) | Warm (s) | Cold/warm Fock builds |
| --- | ---: | ---: | ---: | ---: |
| PBE-RKS / dense | 0.2980 | 0.6346 | 0.0716 | 9 / 1 |
| PBE-RKS / packed | 0.0549 | 0.6301 | 0.0714 | 9 / 1 |
| PBE0-RKS / dense | 0.0458 | 0.6364 | 0.0714 | 9 / 1 |
| PBE0-RKS / packed | 0.0547 | 0.6310 | 0.0715 | 9 / 1 |
| PBE-UKS / dense | 0.2684 | 0.1720 | 0.0491 | 7 / 2 |
| PBE-UKS / packed | 0.0249 | 0.1686 | 0.0489 | 7 / 2 |

Configure, preparation, cold/warm and geometry phases are retained separately;
configuration was 0.0013–0.0025 s. These small cases establish complete execution,
not a speedup or algorithm promotion. Optional 25-AO water/def2-SVP-auxiliary and
moved OH stress references failed to converge in independent PySCF (including a
water Newton attempt). Their logs remain negative fixture evidence and do not
qualify those endpoints. No production fallback or tolerance was changed.

The full DF suite initially exposed the pre-existing three-term source-capacity
underestimate. It reproduced with an independent pre-migration library; #1956
corrects its six-term public record bound separately. Initial test compilation
failures (CUDA 12.9 signatures) and all failed oracle logs are retained.

Evidence and reproducible scripts are under
`/data/jzzeng/qc-dft-coulomb-lowering-1890/.artifacts/1890-coulomb/` on the editing
host and n1: source/binary hashes, verified ccache commands and before/after
statistics, `qualify*.sh`, `host.sh`, `endpoint.py`, and all gate logs. The final
qualified library SHA-256 is
`fffaec2af66fdb6b28ee9602dfaf7aa3f48abd8a5b0cdb7adf4b15fcedffad9e`.

## Revisit when

Measured complete DF-KS endpoints justify replacing an incumbent, or streamed
panel/beta-update contracts migrate through the same shared boundary.

## References

- #1886 canonical provider boundary; #1890 production-callsite migration.
- `tests/python/test_df_coulomb_lowering.py`
- `tests/native/test_cuda_vector_contraction.cpp`
- `tests/python/test_dft_df_public.py`
