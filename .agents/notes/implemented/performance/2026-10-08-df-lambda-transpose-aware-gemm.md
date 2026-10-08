# Decision: transpose-aware operands for DF Lambda matrix actions

Status: implemented
Date: 2026-10-08
Base: master `4385f72751b829883407c01106186917c344317b`

## Problem

The DF matrix packer forced every binary contraction into NN operand layouts,
although the native ordinary and strided-batched GEMM bindings already support
transpose flags. Lambda reverse-mode graphs also contain permutation views that
can be expressed by relabeling an operand, instead of allocating and launching
another permutation. These buffers are regenerated for response seeds and Q
batches even though they add no scientific work.

## Decision

Compose operand permutation chains into their einsum labels. Borrow the original
buffer when its physical axes are exactly batch/rows/columns or
batch/columns/rows. The existing native lowering derives N/T flags and explicit
matrix strides from the resulting contraction. Otherwise materialize the
canonical packed order as a normal, arena-budgeted TensorIR transpose.

The change belongs to the shared DF compiler packer, not to a handwritten
Lambda equation or native runtime special case. DF residual and parameter/factor
consumers use the same layout policy. Primal, staged scalar and expanded
derivative graphs remain unchanged; optimized generated layout identities
deliberately change.

## Invariants

- Preserve exact index identity, rational coefficients, contraction trees, and
  the index order within each flattened row/column/reduction group.
- Q batches remain leading and retain their ordered accumulation and tail rules.
- Equal extents never permit exchanging distinct labels or inferring symmetry.
- Repeated labels, one-sided reductions, n-ary contractions and nonadmitted
  batches retain their original lowering.
- Independently consumed/shared permutation outputs remain live.
- No additional virtual axes, omitted integral blocks, provider resources or
  scientific fallback are introduced. Runtime capacity queries still charge
  the complete optimized arena; fewer copies do not guarantee a smaller peak.

## Evidence

For the generator's representative `(nocc,nvir)=(2,3)` and symbolic Q batch 3:

| Stage | Materialized transpose nodes before | After |
| --- | ---: | ---: |
| staged core transpose | 148 | 118 |
| staged auxiliary transpose | 37 | 30 |
| staged factor response | 64 | 53 |
| staged auxiliary primal | 23 | 20 |

These are compiler graph counts, not measured GPU traffic or endpoint speedups.
The native operation/packing counters and capacity queries derive from these
same lowered graphs. No full energy/force speedup is claimed.

Regression gates cover NN/NT/TN/TT ordinary and batched layouts, asymmetric
dimensions, nested non-involutive permutations, grouped rows, interleaved Q and
reduction axes, equal-sized distinct reduction labels, shared outputs and
nonadmitted fallback paths. Existing DF residual/Lambda tests compare with the
expanded equations and independently written NumPy derivatives. The native
contraction projection test validates every generated request at nonrepresentative
runtime dimensions and verifies descriptor corruption is rejected.

Initial validation on this patch:

- Focused DF packing, Lambda/reduction, solver-policy/lifetime and native
  contraction-projection suite: 65 passed, 57 environment-gated tests skipped.
  The skipped cases include full native DF solver/Lambda execution; they are
  not counted as qualified by the standalone contraction gate.
- `test_native_contraction_binding.py`: 2 passed under Slurm job 6658 on node1,
  `main`, `gpu:5090:1`, with a five-minute time limit. The real-device gate covers
  typed ordinary/batched transpose dispatch, numerical failures and capture
  boundaries. Slurm assigned `CUDA_VISIBLE_DEVICES=2`, preserved unchanged.
  Compilation used verified ccache 4.5.1 and CUDA 12.9.1 without clearing caches.
- Compiler structure check: 494 modules, zero dependency errors. Focused Ruff
  checks/format checks and `git diff --check` pass.

Run the real-device gate with `GENERATIVEQC_DF_CC_CUDA_TEST=1`,
`PYTHONPATH=python:.`, the CUDA toolchain/library paths configured, and
`python -m pytest -q tests/python/test_native_contraction_binding.py` inside an
explicit finite `srun` allocation. This does not replace complete native Lambda
or molecular energy/force qualification.

### Complete native qualification and measured phase benefits

Subsequent Slurm job 2705 on node2 qualifies the complete candidate library:
109 tests passed and 10 conditional cases skipped, including native Lambda/DF
solver execution and independent complete-force finite differences. Matched
frozen-master/candidate complete E+force observations in job 2706 retain six
interleaved pairs each for water24 and methane34. CCSD median times fall
3.51% and 0.93%; methane Lambda falls 0.66%. Those phase improvements have
positive descriptive paired-bootstrap intervals in this run.

Complete E+force changes and water Lambda's point estimate overlap observed
variation; they are not qualified speedups. The third case's baseline
qualification process times out at 300 seconds before publication, so no large
candidate comparison or large-domain claim is made. This change is retained for
bounded materialization/phase benefits, not as a remedy for the dominant native
nuclear response. Full data, numerical gates, unchanged semantic work and
reproduction constraints are in
`benchmarks/results/df-lambda-transpose-actions/README.md` and its retained JSON.

## Rejected alternatives

Always packing NN wastes buffers and launches already representable by GEMM.
Unconditionally treating a tensor permutation as a matrix transpose would be
wrong for interleaved Q axes or internally permuted flattened groups. Choosing a
different reduction-label order merely to find more direct layouts is outside
this change: it would alter summation order and require separate qualification.

## Revisit when

Measured complete energy/force endpoint timings identify remaining permutation
traffic as a bottleneck, or a typed affine-view lowering can remove interleaved
packing with independent scientific and capacity gates.

## References

- `python/generativeqc_compiler/cc/df_gemm.py`
- `tests/python/test_df_gemm.py`
- `tests/python/test_df_lambda_matrix.py`
- `tests/python/test_native_contraction_binding.py`
- `docs/developer/df_ccsdt_gradient.md`
