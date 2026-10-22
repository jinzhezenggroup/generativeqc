# Decision: conservative FP32 block rejection, with separate integer experiments

Status: implemented (explicit opt-in)
Date: 2026-10-23

## Problem

RTX 5090 has relatively weak FP64 throughput, but Direct screening precision is
independent of integral/contraction precision. The useful question is whether
cheaper admission removes complete endpoint time without changing any accepted
shell quartet. Replacing the three-double density record also changes the ABI
shared with generated shell streams, and removing its original values would
make exact near-threshold FP64 decisions impossible without reconstruction.

## Decision

Keep `GENERATIVEQC_DIRECT_FP32_BLOCK_SCREENING` default-off. Capture it in the
borrowed batch view before launch/capture and include it in HF resource/restart
identity. The three existing bounded block consumers may reject using an FP32
upper enclosure, then run the unchanged FP64 block and exact shell predicates
for survivors. No new allocation, kernel launch, synchronization or quartet
materialization is introduced by the production experiment. Compatibility
compilers without the qualified NVCC intrinsic contract retain FP64 admission.

The full bound/density data stays FP64. Neither Schwarz initialization nor
ERI, J/K, energy or force arithmetic is downgraded. Generated resident streams
and class-page gates do not consume this experiment. This deliberately limits
the possible speedup to the bounded consumers actually reached by a workload.

## Numerical invariants

For each nonnegative input, `__double2float_ru` encloses its FP64 value.
Each sequential `__fmul_ru` result encloses the corresponding FP64 rounded
product, including intermediate overflow. Thresholds use downward conversion;
strict `<` preserves equality. Positive values/products smaller than the FP32
normal range are raised to `FLT_MIN`, making the proof valid even with FTZ.
Negative/nonfinite inputs become infinity for coarse purposes; unordered
products such as zero times infinity never authorize a new rejection. The
original predicate remains authoritative on every uncertain candidate.

The CUDA arithmetic test checks both admission purposes against sequential
host FP64, and every early rejection against extended-precision products.
It includes zero, signed zero, negative values, quiet/signaling NaNs, infinity,
FP32/FP64 subnormal and overflow regions, random binary64 bit patterns, and
exact/adjacent thresholds. Both `--ftz=false` and `--ftz=true` are required.
The tests compare externally observable admission, not source spelling.

## INT8 and Tensor Core experiment

`benchmarks/direct_screening_probe.cu` is a standalone synthetic experiment.
Each sorted 16-bound tile uses an exact power-of-two scale with upward-rounded
signed INT8 codes in [0,64]. Unsupported tiles fail open. INT32 products are
exact and reconstructed with conservative FP32 scaling and density products;
all survivors receive FP64 refinement. Scalar INT8 and WMMA use the same codes.
The tiled scalar control also uses the same warp, tile and refinement mapping,
so its comparison with WMMA isolates the matrix instruction from scheduling.
The WMMA outer product has one nonzero lane out of 16 along K: screening does
not naturally supply a long dot product. Its packing, padding, scale storage
and refinement stay in the timer. A prepared-FP32 arm separately measures
amortizing bound conversion across a matrix of candidate blocks.

Every emitted task identity is checked against FP64. Integer experiments do
not enter production; a dense square mask is unsuitable as an unbounded
molecular quartet queue. A future integration must charge bounded staging,
density refresh and geometry invalidation, and include actual compaction and
integral consumers. Synthetic throughput cannot establish endpoint benefit.

## Evidence

The [source-matched receipts](../../../../benchmarks/results/direct-screening-20261023/README.md)
retain accepted timing/error comparisons, reproduction scripts, selected kernel
rows and a summary reader. Full arrays, logs and excluded trials stay in local
artifacts; scalar errors were computed from the complete arrays before compaction. The final n1/RTX 5090 campaign contains 18 fresh
processes and 78 complete E/F calls: three cold and nine frozen warm samples per
arm/system, plus one changed-geometry call per arm/system. PBE0 and r2SCAN-3c
use 96-atom water; C7 is QM7 C7H16 Direct RHF/def2-SVP, not the CCSD(T)/CBS
endpoint. Median cold/warm wall times show no meaningful molecular speedup,
so the experiment stays default-off.

Every molecular sample passes independent tighter GPU4PySCF energy/force gates,
including D4/gCP corrections for r2SCAN-3c. Maximum baseline/candidate differences
are 1.82e-12 Eh and 1.62e-11 Eh/Bohr. DFT force census fields are identical:
92,223,676 admitted quartets for PBE0 and 243,123,642 for r2SCAN-3c. PBE0 rejects
no pages at the block gate; r2SCAN-3c rejects 30,112. The census does not cover
the C7 HF owner, so no C7 quartet-count claim is made. Changed gates are fused
with compaction/integral consumers; profiles cannot isolate their instruction
time without additional instrumentation.

The arithmetic fixture passes 3,061,731 cases per FTZ setting with zero task-set
mismatches. Independent UHF force regression and Compute Sanitizer memcheck also
pass. The final synthetic experiment checks every task identity in all six arms,
at 256/2048/8192 bounds, for sparse and all-survive thresholds. At 8192 bounds
the force screening timer is 0.131 ms for WMMA versus 0.154 ms for tiled INT8
in the sparse case; all-survive times are 0.543 and 0.541 ms respectively.
Prepared FP32 saves conversion overhead in the sparse case but loses to FP64
when all candidates survive. Small integer tiles lose to packing/launch costs.
These results support further prepared-metadata experiments, not production
INT8/WMMA selection or a claim of faster molecular endpoints.

## Revisit when

Promote only after repeated complete cold/warm/moved endpoints improve for the
intended domains and independently pass numerical/task-set gates. A direct
FP32 conversion on every block visit can cost more than the few saved FP64
multiplies. Prepared FP32 metadata is the next measurable option; retain original
FP64 bounds or a correct reconstruction path for exact refinement. Shrinking
the 24-byte generated density ABI to 12 bytes requires its own producer,
consumer, lifetime and near-threshold qualification.
